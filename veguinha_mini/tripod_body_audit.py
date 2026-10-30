"""Ensaio corporal curto do andaime de seis pernas, sem plasticidade."""

import argparse
from pathlib import Path
import signal
import time

import numpy as np

from .phase_lock_audit import _save_atomic
from .search_fast import FastEvaluator, LOCAL_OSCILLATOR


def run(output, seconds=120, force=18, budget_s=240):
    if seconds != 120 or not 0 < budget_s <= 240:
        raise ValueError("este ensaio exige 120 s e orçamento real de até 240 s")
    started = time.perf_counter()
    output = Path(output)
    report = {"seconds": seconds, "force": force, "normalize_neighbors": True,
              "cross_tau_ms": 40, "seeds": [42, 43, 44], "status": "running",
              "rows": []}
    previous_handler = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise TimeoutError("orçamento real excedido")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, budget_s)
    try:
        _save_atomic(output, report)
        for seed in report["seeds"]:
            for condition, strength in (("sem acoplamento", 0), ("tripé", force)):
                row_started = time.perf_counter()
                evaluator = FastEvaluator(seed=seed, cross_force=strength,
                                          normalize_neighbors=True, cross_tau_ms=40)
                result = evaluator.evaluate({**LOCAL_OSCILLATOR, "coupling": 0.0}, seconds)
                report["rows"].append({"seed": seed, "condition": condition,
                                       "net_path": result["net_path"],
                                       "net_energy": result["net_energy"],
                                       "alternation": round(float(-np.mean(result["phase"])), 4),
                                       "phase_by_edge": result["phase"],
                                       "premotor_rate_min_hz": result["rate_min"],
                                       "premotor_rate_max_hz": result["rate_max"],
                                       "saturation": result["saturation"],
                                       "wall_s": round(time.perf_counter() - row_started, 3)})
                report["wall_s"] = round(time.perf_counter() - started, 3)
                _save_atomic(output, report)
                print(f"seed={seed} {condition} ratio={result['net_path']:.4f}", flush=True)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "results" / "tripod_body_120s_v1.json")
    parser.add_argument("--budget-s", type=float, default=240)
    args = parser.parse_args()
    report = run(args.output, budget_s=args.budget_s)
    print(f"{report['status']}: {len(report['rows'])} ensaios em {report['wall_s']}s")
    return 0 if report["status"] == "complete" else 124


if __name__ == "__main__":
    raise SystemExit(main())
