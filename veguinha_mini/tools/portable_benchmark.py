#!/usr/bin/env python3
"""Benchmark portátil Veguinha. Use --help; não instala nada nem treina.

Esta é a fonte do lançador. tools/package_benchmark.py gera o .py único
com o snapshot do projeto embutido para enviar como documento no WhatsApp.
"""
import argparse
import base64
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import statistics
import signal
import subprocess
import sys
import tempfile
import time
import zipfile

PAYLOAD = ""  # Replaced by package_benchmark.py, not downloaded at runtime.
PAYLOAD_SHA256 = ""


def memory_mb():
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024 if sys.platform == "darwin" else 1024)


def environment():
    versions = {}
    for package in ("numpy", "brian2", "Cython", "mujoco"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return dict(system=platform.system(), machine=platform.machine(),
                python=platform.python_version(), cpu_count=os.cpu_count(),
                available_cpus=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                versions=versions,
                thread_environment={key: os.environ.get(key) for key in
                                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})


def run_bounded(cmd, timeout):
    """Stop the worker and its compiler children together on POSIX."""
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=os.name == "posix",
                               env={**os.environ, "PYTHONUNBUFFERED": "1"})
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.communicate()
        raise
    return subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)


def worker(args):
    # Each worker imports a fresh, identical snapshot, never the local checkout.
    sys.path.insert(0, args._root)
    import numpy as np
    from brian2 import ms
    from veguinha_mini.config import Config
    from veguinha_mini.mujoco_body import MujocoBody
    config = Config(seed=42, scale=args.scale, backend=args.backend,
                    body_model=args.body_model, block_ms=20, phase5_enabled=False)
    start = time.perf_counter()
    metadata = {}
    if args._worker == "rede":
        from veguinha_mini.network import build_network
        model = build_network(config, record_spikes=False)
        advance = lambda duration: model.network.run(duration * 1000 * ms)
        metadata.update(neurons=len(model.group), synapses=len(model.synapses))
    elif args._worker == "fisica":
        body = MujocoBody(dt=.002, variant=args.body_model)
        torque = np.zeros_like(body.angles)
        def advance(duration):
            for _ in range(round(duration / .02)):
                body.step(.02, torque)
        metadata.update(joints=body.model.nu, mass_kg=float(body.model.body_mass.sum()))
        metadata["physics_workload"] = "passive body, zero torque; 20 ms blocks and 2 ms substeps"
    else:
        from veguinha_mini.sim import Simulation
        sim = Simulation(config, headless=True, physics="mujoco", body_substep_ms=2)
        sim.plasticity = False
        advance = sim.run_headless
        metadata.update(neurons=len(sim.model.group), synapses=len(sim.model.synapses),
                        joints=sim.body.model.nu, mass_kg=float(sim.body.model.body_mass.sum()))
    build_s = time.perf_counter() - start
    start = time.perf_counter()
    advance(args.warmup)
    warmup_s = time.perf_counter() - start
    samples = []
    # Physics is measured for longer because it is far faster than the network.
    seconds = args.physics_seconds if args._worker == "fisica" else args.seconds
    for index in range(args.repeats):
        start = time.perf_counter()
        advance(seconds)
        samples.append(time.perf_counter() - start)
        print(f"  janela {index + 1}/{args.repeats}: {samples[-1]:.3f} s reais", flush=True)
    median = statistics.median(samples)
    metadata.update(mode=args._worker, simulated_s_per_window=seconds,
                    wall_s_windows=samples, build_s=build_s, compile_and_warmup_s=warmup_s,
                    warmup_simulated_s=args.warmup, wall_s_per_sim_s=median / seconds,
                    real_time_percent=100 * seconds / median,
                    real_s_per_simulated_minute=60 * median / seconds,
                    peak_rss_mb=memory_mb())
    if args._worker == "rede":
        metadata["spikes_total"] = int(model.monitor.num_events)
    elif args._worker == "malha":
        metadata.update(spikes_total=int(sim.model.monitor.num_events),
                        motor_torque_rms=float(np.sqrt(np.mean(sim.motor.torques ** 2))),
                        physical_time_s=float(sim.body.data.time))
    Path(args._result).write_text(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Benchmark curto da mesma Veguinha nos dois aparelhos. Sem viewer e sem aprendizado.")
    parser.add_argument("--backend", choices=("cython", "numpy"), default="cython")
    parser.add_argument("--body-model", choices=("mk1", "control"), default="mk1")
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--seconds", type=float, default=1, help="segundos simulados por janela de rede/malha (padrão 1)")
    parser.add_argument("--physics-seconds", type=float, default=5)
    parser.add_argument("--warmup", type=float, default=.2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--modes", nargs="+", choices=("rede", "fisica", "malha"), default=["rede", "fisica", "malha"])
    parser.add_argument("--output", type=Path, help="arquivo JSON de saída")
    parser.add_argument("--timeout", type=int, default=180, help="limite total real, até 240 s")
    parser.add_argument("--_worker", choices=("rede", "fisica", "malha"), help=argparse.SUPPRESS)
    parser.add_argument("--_root", help=argparse.SUPPRESS)
    parser.add_argument("--_result", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (not .05 <= args.scale <= 10 or not 1 <= args.repeats <= 5
            or not 10 <= args.timeout <= 240):
        parser.error("scale 0.05–10; repeats 1–5; timeout 10–240")
    for value in (args.seconds, args.physics_seconds, args.warmup):
        if not .02 <= value <= 10 or abs(value / .02 - round(value / .02)) > 1e-8:
            parser.error("durações devem ser múltiplos de 0.02 s, entre 0.02 e 10")
    if args._worker:
        worker(args)
        return
    if not PAYLOAD:
        parser.error("este é o fonte: gere o arquivo portátil com tools/package_benchmark.py")
    output = args.output or Path.cwd() / ("veguinha_benchmark_" + time.strftime("%Y%m%d_%H%M%S") + ".json")
    output = output.resolve()
    report = dict(snapshot_sha256=PAYLOAD_SHA256, environment=environment(),
                  settings=dict(seed=42, body_model=args.body_model, scale=args.scale,
                                backend=args.backend, neural_dt_ms=1, physics_dt_ms=2,
                                block_ms=20, viewer=False, learning=False,
                                repeats=args.repeats, warmup_sim_s=args.warmup,
                                network_window_sim_s=args.seconds, physics_window_sim_s=args.physics_seconds),
                  results=[], errors=[])
    print("Veguinha — comparação de velocidade (100% = tempo real)", flush=True)
    print(f"Corpo: {args.body_model}; escala: {args.scale}; backend: {args.backend}", flush=True)
    print("Primeiro uso pode compilar Cython. Compilação/aquecimento ficam fora da velocidade medida.", flush=True)
    start = time.perf_counter()
    archive = base64.b85decode(PAYLOAD)
    if hashlib.sha256(archive).hexdigest() != PAYLOAD_SHA256:
        raise RuntimeError("snapshot embutido danificado")
    with tempfile.TemporaryDirectory(prefix="veguinha-bench-") as temp:
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            for name in z.namelist():
                path = Path(name)
                if path.is_absolute() or ".." in path.parts:
                    raise RuntimeError("caminho inválido no snapshot")
            z.extractall(temp)
        for mode in args.modes:
            remaining = args.timeout - (time.perf_counter() - start)
            if remaining <= 0:
                report["errors"].append(dict(mode=mode, error="limite total atingido"))
                break
            result_path = Path(temp) / (mode + ".json")
            cmd = [sys.executable, str(Path(__file__).resolve()), "--_worker", mode,
                   "--_root", temp, "--_result", str(result_path), "--backend", args.backend,
                   "--body-model", args.body_model, "--scale", str(args.scale),
                   "--seconds", str(args.seconds), "--physics-seconds", str(args.physics_seconds),
                   "--warmup", str(args.warmup), "--repeats", str(args.repeats)]
            print(f"\nTestando {mode}...", flush=True)
            try:
                result = run_bounded(cmd, min(90, remaining))
                if result.returncode:
                    error = (result.stderr or result.stdout)[-1800:]
                    report["errors"].append(dict(mode=mode, error=error))
                    print(error, flush=True)
                else:
                    row = json.loads(result_path.read_text())
                    report["results"].append(row)
                    print(f"{row['real_time_percent']:.1f}% do tempo real | {row['wall_s_per_sim_s']:.3f} s real/s simulado | pico {row['peak_rss_mb']:.0f} MB", flush=True)
            except subprocess.TimeoutExpired:
                report["errors"].append(dict(mode=mode, error="tempo esgotado (compilação incluída no limite; cache Cython preservado)"))
                print("Limite do teste atingido; resultado parcial será salvo.", flush=True)
            report["total_wall_s"] = time.perf_counter() - start
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    report["total_wall_s"] = time.perf_counter() - start
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nResultado para compartilhar: {output}", flush=True)
    if report["errors"]:
        print("Houve testes incompletos; veja errors no JSON. Nenhum backend foi trocado automaticamente.")


if __name__ == "__main__":
    main()
