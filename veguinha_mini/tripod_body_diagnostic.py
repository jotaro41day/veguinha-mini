"""Diagnóstico pareado da ponte rede-motor e da mecânica do tripé."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import tempfile
import time
from types import SimpleNamespace

import numpy as np

from .body import Body, LEG_NAMES
from .config import Config
from .metrics import Metrics
from .motor import MotorDecoder, POOL_NAMES
from .phase_lock_audit import _save_atomic
from .search_fast import FastEvaluator, LOCAL_OSCILLATOR
from .world import World


PERIOD_S = 0.38
DT_S = 0.02
FINAL_S = 20
TRIPOD_PHASE = np.array([0, math.pi, 0, math.pi, 0, math.pi])
CONDITIONS = ("rede18", "controle", "embaralhado", "imposto0", "jitter10",
              "jitter20", "jitter35")


def _phase_from_counts(counts):
    """Fase da diferença premotora flexor-extensor, 20 ms por amostra."""
    traces = counts[:, :, 0, 0] - counts[:, :, 0, 1]
    traces = traces.astype(float) - traces.mean(axis=0, keepdims=True)
    spectrum = np.fft.fft(traces, axis=0)
    freq = np.fft.fftfreq(len(traces), DT_S)
    band = (freq >= 1) & (freq <= 3.33)
    power = (np.abs(spectrum[band]) ** 2).sum(axis=1)
    if power.max() < 1e-8:
        return None
    center = freq[np.flatnonzero(band)[np.argmax(power)]]
    positive = (freq >= max(1.0, center - .8)) & (freq <= min(3.5, center + .8))
    analytic = np.fft.ifft(spectrum * (2 * positive[:, None]), axis=0)
    return np.angle(analytic)


def _phase_contact_r(phase, contact):
    if np.std(contact) < 1e-8:
        return 0.0
    design = np.column_stack((np.sin(phase), np.cos(phase), np.ones(len(phase))))
    fitted = design @ np.linalg.lstsq(design, contact.astype(float), rcond=None)[0]
    if np.std(fitted) < 1e-8:
        return 0.0
    return float(np.clip(np.corrcoef(fitted, contact)[0, 1], 0, 1))


def _contact_coherence(contact):
    pairs = ((0, 2), (0, 4), (2, 4), (3, 1), (3, 5), (1, 5))
    values = [float(np.corrcoef(contact[:, a], contact[:, b])[0, 1])
              if np.std(contact[:, a]) and np.std(contact[:, b]) else 0.0
              for a, b in pairs]
    all_three = np.column_stack((np.all(contact[:, (0, 2, 4)], axis=1),
                                 np.all(contact[:, (3, 1, 5)], axis=1)))
    return float(np.mean(values)), np.mean(all_three, axis=0).tolist()


def _diagnostics(metrics, body, torques, contacts, impulses, phases, counts=None):
    snapshot = metrics.snapshot(body)
    # Ignore the ends of the Fourier window in measurements of phase.
    usable = slice(round(2 / DT_S), -round(1 / DT_S))
    phase = (_phase_from_counts(counts) if counts is not None else phases)
    contact = contacts[usable]
    selected_phase = None if phase is None else phase[usable]
    relative = []
    spread = []
    phase_contact = []
    for leg in range(6):
        if selected_phase is None:
            relative.append(None); spread.append(None); phase_contact.append(None)
            continue
        delta = selected_phase[:, leg] - selected_phase[:, 0]
        phasor = np.exp(1j * delta).mean()
        relative.append(round(float(np.angle(phasor, deg=True)), 2))
        spread.append(round(float(np.rad2deg(np.sqrt(-2 * np.log(max(abs(phasor), 1e-12))))), 2))
        phase_contact.append(round(_phase_contact_r(selected_phase[:, leg], contact[:, leg]), 4))
    coherence, all_three = _contact_coherence(contact)
    cycles = FINAL_S / PERIOD_S
    return {"net_path": snapshot["net_path"],
            "net_energy": round(snapshot["displacement"] / max(snapshot["energy_spent"], 1e-9), 4),
            "displacement": snapshot["displacement"],
            "path_length": snapshot["path_length"],
            "energy_spent": snapshot["energy_spent"],
            "phase_deg_vs_lf": relative,
            "phase_sd_deg_vs_lf": spread,
            "phase_contact_r": phase_contact,
            "contact_fraction_by_leg": np.mean(contact, axis=0).round(4).tolist(),
            "same_tripod_contact_r": round(coherence, 4),
            "all_three_contact_fraction": np.round(all_three, 4).tolist(),
            "impulse_forward_per_cycle_by_leg":
                np.round(np.sum(impulses, axis=0) / cycles, 5).tolist(),
            "impulse_forward_per_cycle": round(float(np.sum(impulses) / cycles), 5),
            "torque_rms_by_joint": np.sqrt(np.mean(torques ** 2, axis=(0, 1))).round(5).tolist(),
            "torque_rms_by_leg": np.sqrt(np.mean(torques ** 2, axis=0)).round(5).tolist(),
            "realized_relative_phase_sd_deg":
                round(float(np.std(np.angle(np.exp(1j *
                    ((phases[usable, 1:] - phases[usable, :1]) - TRIPOD_PHASE[1:])))
                    * 180 / math.pi)), 2)
                if counts is None else None}


def _save_trace(output, seed, condition, **arrays):
    target = output.parent / f"tripod_diag_trace_seed{seed}_{condition}.npz"
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".npz", delete=False) as f:
        temporary = Path(f.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target.name


def _finish_run(body, metrics, torque_log, contact_log, impulse_log, phase_log,
                count_log, output, seed, condition):
    torques = np.asarray(torque_log)
    contacts = np.asarray(contact_log, dtype=bool)
    impulses = np.asarray(impulse_log)
    phases = None if not phase_log else np.asarray(phase_log)
    counts = None if not count_log else np.asarray(count_log)
    result = _diagnostics(metrics, body, torques, contacts, impulses, phases, counts)
    result["trace"] = _save_trace(output, seed, condition, torques=torques,
                                   contacts=contacts, forward_impulses=impulses,
                                   phases=np.empty((0, 6)) if phases is None else phases,
                                   premotor_counts=np.empty((0, 6, 2, 2)) if counts is None else counts)
    return result


def _network_run(seed, condition, output, seconds):
    strength = 0 if condition == "controle" else 18
    evaluator = FastEvaluator(seed=seed, cross_force=strength,
                              normalize_neighbors=True, cross_tau_ms=40)
    weights = evaluator.build_weights({**LOCAL_OSCILLATOR, "coupling": 0.0})
    model = SimpleNamespace(group=evaluator.probe.neurons,
                            motor_pools={leg: {name: evaluator.probe.pools[index]
                                                .reshape(4, 8)[pool]
                                                for pool, name in enumerate(POOL_NAMES)}
                                         for index, leg in enumerate(LEG_NAMES)})
    decoder = MotorDecoder(model, Config(motor_gain=.18))
    world = World(); body = Body(world); metrics = Metrics(body)
    state = None; previous = np.zeros(768, dtype=np.int64)
    rng = np.random.default_rng(seed + 9000)
    permutation = rng.permutation(6)
    while np.any(permutation == np.arange(6)):
        permutation = rng.permutation(6)
    torque_log, contact_log, impulse_log, count_log = [], [], [], []
    for step in range(round(seconds / DT_S)):
        contacts = body.angles[:, 1] < 0
        _, _, _, _, state = evaluator.simulate(
            evaluator.starts, evaluator.targets, weights, 6, 20, seed,
            LOCAL_OSCILLATOR["drive"], LOCAL_OSCILLATOR["proprio"], 2.5,
            state=state, external_angles=body.angles.ravel(), external_contacts=contacts.astype(float),
            return_state=True, edge_kinds=evaluator.kinds, cross_tau_ms=40,
            inhibitory_bias=LOCAL_OSCILLATOR["inhibitory_bias"],
            motor_bias=LOCAL_OSCILLATOR["motor_bias"],
            adaptation_a=LOCAL_OSCILLATOR["adaptation_a"],
            adaptation_d=LOCAL_OSCILLATOR["adaptation_d"])
        delta = state["total"] - previous
        previous = state["total"].copy()
        torque = decoder.update(state["total"], DT_S)
        if condition == "embaralhado":
            torque = torque[permutation]
        heading = body.heading
        body.step(DT_S, torque, contacts, world)
        metrics.update(body, torque, contacts)
        if step >= round((seconds - FINAL_S) / DT_S):
            torque_log.append(torque)
            contact_log.append(contacts)
            forward = np.array([math.cos(heading), math.sin(heading)])
            impulse_log.append(body.ground_forces @ forward * DT_S)
            count_log.append(delta[evaluator.probe.premotor].sum(axis=-1))
    result = _finish_run(body, metrics, torque_log, contact_log, impulse_log, [],
                         count_log, output, seed, condition)
    result["permutation"] = permutation.tolist() if condition == "embaralhado" else None
    result["calibration_amp"] = (np.sqrt(2) * np.sqrt(np.mean(np.asarray(torque_log) ** 2,
                                                           axis=(0, 1)))).round(5).tolist()
    return result


def _imposed_run(seed, condition, amplitude, output, seconds):
    sigma_deg = 0 if condition == "imposto0" else int(condition.removeprefix("jitter"))
    # The target is the dispersion between two legs. Independent per-leg
    # offsets each need target/sqrt(2) to reproduce that pair dispersion.
    sigma = math.radians(sigma_deg) / math.sqrt(2)
    alpha = math.exp(-DT_S / PERIOD_S)
    rng = np.random.default_rng(seed + 5000)
    jitter = rng.normal(0, sigma, 6)
    world = World(); body = Body(world); metrics = Metrics(body)
    torque_log, contact_log, impulse_log, phase_log = [], [], [], []
    amp = np.asarray(amplitude, dtype=float)
    for step in range(round(seconds / DT_S)):
        if sigma:
            jitter = alpha * jitter + math.sqrt(1 - alpha * alpha) * sigma * rng.normal(size=6)
        phase = 2 * math.pi * step * DT_S / PERIOD_S + TRIPOD_PHASE + jitter
        torque = np.column_stack((amp[0] * np.sin(phase), amp[1] * np.cos(phase)))
        torque = np.clip(torque, -1, 1)
        contacts = body.angles[:, 1] < 0
        heading = body.heading
        body.step(DT_S, torque, contacts, world)
        metrics.update(body, torque, contacts)
        if step >= round((seconds - FINAL_S) / DT_S):
            torque_log.append(torque)
            contact_log.append(contacts)
            forward = np.array([math.cos(heading), math.sin(heading)])
            impulse_log.append(body.ground_forces @ forward * DT_S)
            phase_log.append(phase)
    result = _finish_run(body, metrics, torque_log, contact_log, impulse_log,
                         phase_log, [], output, seed, condition)
    result["command_amplitude"] = amp.tolist()
    result["jitter_target_deg"] = sigma_deg
    return result


def run(output, seeds, budget_s=240, seconds=120):
    if seconds != 120 or not 0 < budget_s <= 240:
        raise ValueError("diagnóstico requer 120 s simulados e até 240 s reais por lote")
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    report = (json.loads(output.read_text()) if output.exists() else
              {"seconds": seconds, "final_window_s": FINAL_S, "period_s": PERIOD_S,
               "phase_legs": "LF LM LH RF RM RH; A=LF/RM/LH; B=RF/LM/RH",
               "conditions": list(CONDITIONS), "rows": [], "status": "running",
               "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                 for name in ("body.py", "motor.py", "fast_kernel.pyx",
                                              "search_fast.py", "tripod_body_diagnostic.py")}})
    started = time.perf_counter()
    previous_handler = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise TimeoutError("orçamento de tempo real atingido")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, budget_s)
    try:
        for seed in seeds:
            for condition in CONDITIONS:
                if any(row["seed"] == seed and row["condition"] == condition
                       for row in report["rows"]):
                    continue
                row_started = time.perf_counter()
                if condition in ("rede18", "controle", "embaralhado"):
                    result = _network_run(seed, condition, output, seconds)
                else:
                    calibration = next(row["calibration_amp"] for row in report["rows"]
                                       if row["seed"] == seed and row["condition"] == "rede18")
                    result = _imposed_run(seed, condition, calibration, output, seconds)
                report["rows"].append({"seed": seed, "condition": condition, **result,
                                       "wall_s": round(time.perf_counter() - row_started, 3)})
                report["status"] = "running"
                _save_atomic(output, report)
                print(f"seed={seed} condition={condition} path={result['net_path']:.4f} "
                      f"energy={result['net_energy']:.2f}", flush=True)
        report["status"] = "complete" if len(report["rows"]) == 70 else "partial"
    except TimeoutError as exc:
        report["status"] = "timeout"; report["error"] = str(exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        report["wall_s_last_batch"] = round(time.perf_counter() - started, 3)
        _save_atomic(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--budget-s", type=float, default=240)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "results" / "tripod_body_diagnostic_v1.json")
    args = parser.parse_args()
    report = run(args.output, args.seeds, args.budget_s)
    print(f"{report['status']}: {len(report['rows'])}/70 ensaios", flush=True)
    return 124 if report["status"] == "timeout" else 0


if __name__ == "__main__":
    raise SystemExit(main())
