"""Porta curta de seis pernas: dois tripés, sem corpo e sem aprendizado."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import time

import numpy as np

from .phase_lock_audit import SEEDS, _save_atomic, phase_metrics


# Ordem anatômica: LF, LM, LH, RF, RM, RH.
TRIPOD_A = (0, 4, 2)  # LF, RM, LH
TRIPOD_B = (3, 1, 5)  # RF, LM, RH
EDGE_PAIRS = ((0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5))
SAME_PAIRS = ((0, 2), (0, 4), (2, 4), (3, 1), (3, 5), (1, 5))


def phase_series(counts, window_s=10, stride_s=5):
    """Fase de cada perna relativa a LF em janelas sucessivas."""
    bins_per_second = 20
    window = round(window_s * bins_per_second)
    stride = round(stride_s * bins_per_second)
    if window < 200 or stride < 1 or counts.shape[0] != 6:
        raise ValueError("série precisa de seis pernas e janelas de ao menos 10 s")
    rows = []
    for start in range(0, counts.shape[-1] - window + 1, stride):
        segment = counts[..., start:start + window]
        phases = [phase_metrics(segment, 0, leg) for leg in range(1, 6)]
        rows.append({"time_s": round((start + window / 2) / bins_per_second, 2),
                     "phase_deg": [0.0] + [item["phase_deg"] for item in phases],
                     "phase_sd_deg": [0.0] + [item["phase_sd_deg"] for item in phases]})
    return rows


def neighbor_normalizers():
    degree = np.zeros(6, dtype=float)
    for first, second in EDGE_PAIRS:
        degree[first] += 1
        degree[second] += 1
    return 1 / np.sqrt(degree)


def _tripod_metrics(counts, totals, seconds):
    edges = [phase_metrics(counts, *pair) for pair in EDGE_PAIRS]
    same = [phase_metrics(counts, *pair, target="same") for pair in SAME_PAIRS]
    grouped = np.stack((counts[list(TRIPOD_A)].mean(axis=0),
                        counts[list(TRIPOD_B)].mean(axis=0)))
    group_phase = phase_metrics(grouped)
    motor_rate = (totals.reshape(6, 128)[:, :32].sum(axis=1) / 32 / seconds).round(3)
    active = bool(np.all(motor_rate >= 1.0))
    return {"tripod_locked": (all(row["locked"] for row in edges)
                              and all(row["locked"] for row in same)
                              and group_phase["locked"] and active),
            "anti_edges_locked": sum(row["locked"] for row in edges),
            "same_pairs_locked": sum(row["locked"] for row in same),
            "group_phase_deg": group_phase["phase_deg"],
            "group_phase_sd_deg": group_phase["phase_sd_deg"],
            "motor_rate_hz": motor_rate.tolist(),
            "active": active,
            "edge_phases": [row["phase_deg"] for row in edges],
            "edge_phase_sd_deg": [row["phase_sd_deg"] for row in edges],
            "same_phases": [row["phase_deg"] for row in same],
            "same_phase_sd_deg": [row["phase_sd_deg"] for row in same]}


def run_tripod(seconds, forces, budget_s, output, cross_tau_ms=40.0,
               normalize_neighbors=False, analysis_seconds=None):
    analysis_seconds = seconds if analysis_seconds is None else analysis_seconds
    if not 10 <= analysis_seconds <= seconds <= 60 or not 0 < budget_s <= 240:
        raise ValueError("duração ou orçamento inválido")
    if not forces or forces[0] != 0 or any(not math.isfinite(x) or x < 0 for x in forces):
        raise ValueError("o controle zero deve vir primeiro")
    if not math.isfinite(cross_tau_ms) or cross_tau_ms <= 1:
        raise ValueError("constante de decaimento inválida")
    import pyximport
    from .fast_motor import build_probe
    from .search_fast import LOCAL_OSCILLATOR

    pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)
    from .fast_kernel import simulate
    started = time.perf_counter()
    output = Path(output)
    root = Path(__file__).resolve().parent
    report = {"stage": "tripod", "seeds": list(SEEDS), "forces": list(forces),
              "simulated_seconds": seconds, "cross_tau_ms": cross_tau_ms,
              "analysis_seconds": analysis_seconds,
              "normalize_neighbors": bool(normalize_neighbors),
              "budget_s": budget_s, "rows": [], "status": "running",
              "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                for name in ("tripod_audit.py", "phase_lock_audit.py",
                                             "fast_motor.py", "fast_kernel.pyx")}}
    previous_handler = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise TimeoutError("orçamento de tempo real atingido")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, budget_s)
    try:
        _save_atomic(output, report)
        for seed in SEEDS:
            probe = build_probe(legs=6, seed=seed, backend="cython",
                                coupling=0, mutual_inhibition=0,
                                cross_tau_ms=cross_tau_ms)
            order = np.argsort(probe.pre, kind="stable")
            starts = np.r_[0, np.cumsum(np.bincount(probe.pre, minlength=768))].astype(np.int32)
            targets = probe.post[order].astype(np.int32)
            kinds = probe.edge_kinds[order]
            degrees = probe.edge_degrees[order]
            cross = kinds == 7
            cross_scale = (neighbor_normalizers()[targets[cross] // 128]
                           if normalize_neighbors else 1.0)
            base = np.asarray(probe.recurrent.w[:], dtype=float)[order]
            for force in forces:
                row_started = time.perf_counter()
                weights = base.copy()
                weights[cross] = -force * cross_scale / np.sqrt(degrees[cross])
                kernel_args = dict(return_state=True, edge_kinds=kinds,
                                   cross_tau_ms=cross_tau_ms,
                                   inhibitory_bias=LOCAL_OSCILLATOR["inhibitory_bias"],
                                   motor_bias=LOCAL_OSCILLATOR["motor_bias"],
                                   adaptation_a=LOCAL_OSCILLATOR["adaptation_a"],
                                   adaptation_d=LOCAL_OSCILLATOR["adaptation_d"])
                warmup = seconds - analysis_seconds
                state = None
                warm_counts = None
                previous_total = None
                if warmup:
                    warm_counts, _, _, _, state = simulate(
                        starts, targets, weights, 6, round(warmup * 1000), seed,
                        LOCAL_OSCILLATOR["drive"], LOCAL_OSCILLATOR["proprio"], 2.5,
                        **kernel_args)
                    previous_total = state["total"].copy()
                counts, _, _, _, state = simulate(
                    starts, targets, weights, 6, round(analysis_seconds * 1000), seed,
                    LOCAL_OSCILLATOR["drive"], LOCAL_OSCILLATOR["proprio"], 2.5,
                    state=state, **kernel_args)
                totals = state["total"] if previous_total is None else state["total"] - previous_total
                result = _tripod_metrics(counts, totals, analysis_seconds)
                if warm_counts is not None:
                    result["phase_series"] = phase_series(np.concatenate((warm_counts, counts), axis=-1))
                report["rows"].append({"seed": seed, "force": force, **result,
                                       "wall_s": round(time.perf_counter() - row_started, 3)})
                report["wall_s"] = round(time.perf_counter() - started, 3)
                _save_atomic(output, report)
                print(f"seed={seed} force={force:g} tripod={result['tripod_locked']} "
                      f"edges={result['anti_edges_locked']}/7 "
                      f"same={result['same_pairs_locked']}/6", flush=True)
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
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--analysis-seconds", type=float)
    parser.add_argument("--forces", type=float, nargs="+", default=(0, 10, 15, 20, 25))
    parser.add_argument("--cross-tau-ms", type=float, default=40)
    parser.add_argument("--normalize-neighbors", action="store_true")
    parser.add_argument("--budget-s", type=float, default=240)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "results" / "tripod_phase_v1.json")
    args = parser.parse_args(argv)
    result = run_tripod(args.seconds, tuple(args.forces), args.budget_s,
                        args.output, args.cross_tau_ms, args.normalize_neighbors,
                        args.analysis_seconds)
    print(f"{result['status']}: {len(result['rows'])} ensaios em {result['wall_s']}s", flush=True)
    return 124 if result["status"] == "timeout" else 0


if __name__ == "__main__":
    raise SystemExit(main())
