"""Ensaio curto de travamento de fase, sem corpo ou aprendizado.

Cada seed usa a mesma rede e o mesmo ruído em todas as forças. Com força zero,
as duas pernas são dinamicamente independentes: seus períodos são intrínsecos.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import tempfile
import time

import numpy as np

SEEDS = tuple(range(42, 52))
FORCES = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0)
SAMPLE_DT = .05


def _kernel_kinds(kinds, cross_timescale):
    if cross_timescale not in ("slow", "fast"):
        raise ValueError("escala temporal inválida")
    routed = np.asarray(kinds, dtype=np.int8).copy()
    if cross_timescale == "fast":
        routed[routed == 7] = 0
    return routed


def _period(trace):
    trace = np.asarray(trace, dtype=float)
    trace = trace - trace.mean()
    if np.std(trace) < 1e-8:
        return None
    padded = 8 * len(trace)
    spectrum = np.fft.rfft(trace * np.hanning(len(trace)), n=padded)
    frequency = np.fft.rfftfreq(padded, SAMPLE_DT)
    band = (frequency >= 1.0) & (frequency <= 3.33)
    peak = np.flatnonzero(band)[np.argmax(abs(spectrum[band]) ** 2)]
    return float(1 / frequency[peak])


def phase_metrics(counts, first=0, second=1, target="anti"):
    """Mede a fase circular ao longo do fim da janela, não em um instante."""
    if target not in ("anti", "same"):
        raise ValueError("alvo de fase inválido")
    if counts.shape[-1] < 200:
        raise ValueError("a medição de fase precisa de pelo menos 10 s")
    traces = np.asarray([counts[leg, 0, 0] - counts[leg, 0, 1]
                         for leg in (first, second)], dtype=float)
    periods = [_period(trace[round(2 / SAMPLE_DT):]) for trace in traces]
    traces -= traces.mean(axis=1, keepdims=True)
    spectrum = np.fft.fft(traces, axis=1)
    frequency = np.fft.fftfreq(traces.shape[1], SAMPLE_DT)
    band = (frequency >= 1.0) & (frequency <= 3.33)
    peak = np.flatnonzero(band)[np.argmax((abs(spectrum[:, band]) ** 2).sum(axis=0))]
    center = frequency[peak]
    positive = ((frequency >= max(1.0, center - .8))
                & (frequency <= min(3.5, center + .8)))
    analytic = np.fft.ifft(spectrum * (2 * positive), axis=1)
    values = analytic[:, round(2 / SAMPLE_DT):-round(1 / SAMPLE_DT)]
    amplitude = abs(values[0]) * abs(values[1])
    if not all(periods) or amplitude.max() < 1e-8:
        return {"locked": False, "phase_deg": None, "phase_sd_deg": None,
                "plv": 0.0, "period_s": periods}
    phase = np.angle(values[0] * values[1].conjugate())
    valid = amplitude >= np.percentile(amplitude, 20)
    phasor = np.exp(1j * phase[valid]).mean()
    plv = float(abs(phasor))
    mean = float(np.angle(phasor, deg=True))
    dispersion = float(np.rad2deg(np.sqrt(-2 * np.log(max(plv, 1e-12)))))
    locked = (abs(mean) >= 135 if target == "anti" else abs(mean) <= 45)
    locked = locked and dispersion <= 35
    return {"locked": bool(locked), "phase_deg": round(mean, 2),
            "phase_sd_deg": round(dispersion, 2), "plv": round(plv, 4),
            "period_s": [round(p, 4) for p in periods]}


def _save_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_pair(kind, seconds, forces, budget_s, output, cross_timescale="slow",
             cross_tau_ms=150.0):
    if kind not in ("contralateral", "neighbor"):
        raise ValueError("tipo de par inválido")
    if not 10 <= seconds <= 20 or not 0 < budget_s <= 240:
        raise ValueError("duração ou orçamento inválido")
    if not forces or forces[0] != 0 or any(not math.isfinite(x) or x < 0 for x in forces):
        raise ValueError("forças inválidas; o controle zero deve vir primeiro")
    if cross_timescale not in ("slow", "fast"):
        raise ValueError("escala temporal inválida")
    if not math.isfinite(cross_tau_ms) or cross_tau_ms <= 1:
        raise ValueError("constante de decaimento inválida")
    import pyximport
    from .fast_motor import build_probe
    from .search_fast import LOCAL_OSCILLATOR

    pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)
    from .fast_kernel import simulate

    started = time.perf_counter()
    root = Path(__file__).resolve().parent
    output = Path(output)
    report = {"kind": kind, "seeds": list(SEEDS), "forces": list(forces),
              "simulated_seconds": seconds, "budget_s": budget_s,
              "cross_timescale": cross_timescale,
              "cross_tau_ms": cross_tau_ms,
              "rows": [], "status": "running",
              "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                for name in ("phase_lock_audit.py", "fast_motor.py", "fast_kernel.pyx")}}
    previous_handler = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise TimeoutError("orçamento de tempo real atingido")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, budget_s)
    try:
        _save_atomic(output, report)
        for seed in SEEDS:
            probe = build_probe(legs=2, seed=seed, pair_kind=kind, backend="cython",
                                coupling=0, mutual_inhibition=0)
            order = np.argsort(probe.pre, kind="stable")
            starts = np.r_[0, np.cumsum(np.bincount(probe.pre, minlength=256))].astype(np.int32)
            targets = probe.post[order].astype(np.int32)
            kinds = probe.edge_kinds[order]
            kernel_kinds = _kernel_kinds(kinds, cross_timescale)
            degrees = probe.edge_degrees[order]
            base = np.asarray(probe.recurrent.w[:], dtype=float)[order]
            cross = kinds == 7
            for force in forces:
                row_started = time.perf_counter()
                weights = base.copy()
                weights[cross] = -force / np.sqrt(degrees[cross])
                counts, _, _, _, state = simulate(
                    starts, targets, weights, 2, round(seconds * 1000), seed,
                    LOCAL_OSCILLATOR["drive"], LOCAL_OSCILLATOR["proprio"], 2.5,
                    return_state=True, edge_kinds=kernel_kinds,
                    cross_tau_ms=cross_tau_ms,
                    inhibitory_bias=LOCAL_OSCILLATOR["inhibitory_bias"],
                    motor_bias=LOCAL_OSCILLATOR["motor_bias"],
                    adaptation_a=LOCAL_OSCILLATOR["adaptation_a"],
                    adaptation_d=LOCAL_OSCILLATOR["adaptation_d"])
                phase = phase_metrics(counts)
                totals = np.asarray(state["total"]).reshape(2, 128)
                motor_rate = (totals[:, :32].sum(axis=1) / 32 / seconds).round(3).tolist()
                premotor_rate = (totals[:, 32:80].sum(axis=1) / 48 / seconds).round(3).tolist()
                inhibitory_rate = (totals[:, 80:120].sum(axis=1) / 40 / seconds).round(3).tolist()
                periods = phase["period_s"]
                row = {"seed": seed, "force": force, **phase,
                       "period_difference_pct": (round(200 * abs(periods[0] - periods[1])
                                                       / sum(periods), 2)
                                                 if all(periods) else None),
                       "motor_rate_hz": motor_rate,
                       "premotor_rate_hz": premotor_rate,
                       "inhibitory_rate_hz": inhibitory_rate,
                       "wall_s": round(time.perf_counter() - row_started, 3)}
                report["rows"].append(row)
                report["wall_s"] = round(time.perf_counter() - started, 3)
                _save_atomic(output, report)
                print(f"{kind} seed={seed} force={force:g} locked={phase['locked']} "
                      f"phase={phase['phase_deg']} sd={phase['phase_sd_deg']}", flush=True)
        report["status"] = "complete"
    except TimeoutError as exc:
        report["status"] = "timeout"
        report["error"] = str(exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        report["wall_s"] = round(time.perf_counter() - started, 3)
        _save_atomic(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("contralateral", "neighbor"), required=True)
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--forces", type=float, nargs="+", default=FORCES)
    parser.add_argument("--budget-s", type=float, default=240)
    parser.add_argument("--cross-timescale", choices=("slow", "fast"), default="slow")
    parser.add_argument("--cross-tau-ms", type=float, default=150.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or Path(__file__).parent / "results" / f"phase_lock_{args.kind}.json"
    result = run_pair(args.kind, args.seconds, tuple(args.forces), args.budget_s,
                      output, cross_timescale=args.cross_timescale,
                      cross_tau_ms=args.cross_tau_ms)
    print(f"{result['status']}: {len(result['rows'])} ensaios em {result['wall_s']}s", flush=True)
    return 124 if result["status"] == "timeout" else 0


if __name__ == "__main__":
    raise SystemExit(main())
