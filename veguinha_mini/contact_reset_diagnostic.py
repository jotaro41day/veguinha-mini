"""One-leg local-contact phase-reset assay before testing the six-leg body."""

import csv
from pathlib import Path
import time

import numpy as np

from .fast_motor import build_probe
from .phase_lock_audit import _period
from .search_fast import LOCAL_OSCILLATOR


def run(seed, mode, seconds=20, reset_gain=2.0):
    import pyximport
    pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)
    from .fast_kernel import simulate

    started = time.perf_counter()
    probe = build_probe(legs=1, seed=seed, backend="cython", coupling=0,
                        mutual_inhibition=0, cross_tau_ms=40)
    order = np.argsort(probe.pre, kind="stable")
    starts = np.r_[0, np.cumsum(np.bincount(probe.pre, minlength=128))].astype(np.int32)
    counts, angles, contacts, _, state = simulate(
        starts, probe.post[order].astype(np.int32),
        np.asarray(probe.recurrent.w[:], dtype=float)[order], 1,
        round(seconds * 1000), seed, LOCAL_OSCILLATOR["drive"],
        LOCAL_OSCILLATOR["proprio"], 2.5,
        contact_gain=1 if mode == "tonic" else 0,
        contact_phase_reset_gain=reset_gain if mode == "reset" else 0,
        return_state=True, edge_kinds=probe.edge_kinds[order], cross_tau_ms=40,
        inhibitory_bias=LOCAL_OSCILLATOR["inhibitory_bias"],
        motor_bias=LOCAL_OSCILLATOR["motor_bias"],
        adaptation_a=LOCAL_OSCILLATOR["adaptation_a"],
        adaptation_d=LOCAL_OSCILLATOR["adaptation_d"])
    first = round(5 / .05)
    hip = counts[0, 0, 0, first:] - counts[0, 0, 1, first:]
    knee = counts[0, 1, 0, first:] - counts[0, 1, 1, first:]
    def antagonist(joint):
        flex, extend = counts[0, joint, :, first:]
        return float(np.corrcoef(flex, extend)[0, 1]) if np.std(flex) and np.std(extend) else 0
    return dict(seed=seed, mode=mode, reset_gain=reset_gain if mode == "reset" else 0,
                hip_period_s=round(_period(hip), 3),
                knee_period_s=round(_period(knee), 3),
                hip_antagonist=round(antagonist(0), 3),
                knee_antagonist=round(antagonist(1), 3),
                foot_contact=round(float(contacts[1, first:].mean()), 3),
                hip_angle_sd=round(float(angles[0, first:].std()), 3),
                knee_angle_sd=round(float(angles[1, first:].std()), 3),
                motor_hz=round(float(state["total"][:32].sum() / (32 * seconds)), 3),
                wall_s=round(time.perf_counter() - started, 2))


def main():
    path = Path(__file__).parent / "results" / "contact_reset_1leg.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = None
        for seed in range(42, 52):
            for mode in ("tonic", "reset"):
                row = run(seed, mode)
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                handle.flush()
                print(row, flush=True)


if __name__ == "__main__":
    main()
