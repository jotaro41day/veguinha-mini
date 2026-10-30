"""Mede alternância neural e movimento do andaime no circuito isolado."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from brian2 import Clock, NetworkOperation, ms

from .isolated import LEGS, isolated_simulation


def antagonist_correlation(events):
    """Correlação de disparos flexor/extensor após janela causal de 100 ms."""
    events = np.asarray(events, dtype=float)
    if events.ndim != 2 or events.shape[1] != 2 or len(events) < 5:
        raise ValueError("série deve ter pelo menos cinco blocos flexor/extensor")
    smoothed = [np.convolve(events[:, index], np.ones(5) / 5, mode="valid")
                for index in range(2)]
    if min(np.std(signal) for signal in smoothed) < 1e-9:
        return 0.0
    return float(np.corrcoef(*smoothed)[0, 1])


def measure(seed, seconds=30, tonic_drive=2, plasticity=False, backend="cython",
            **config_changes):
    sim = isolated_simulation(seed=seed, tonic_drive=tonic_drive, backend=backend,
                              plasticity=plasticity, motor_scaffold=True,
                              **config_changes)
    model = sim.model
    pool_indices = np.asarray([
        [[model.half_centers[leg][f"{joint}_{action}_e"]
          for action in ("flex", "extend")]
         for joint in ("hip", "knee")]
        for leg in LEGS
    ], dtype=np.int32)
    previous = np.zeros(len(model.group), dtype=np.int64)
    samples = []

    def record():
        nonlocal previous
        count = np.asarray(model.monitor.count[:], dtype=np.int64)
        delta = count - previous
        previous = count.copy()
        samples.append(delta[pool_indices].sum(axis=-1))

    model.network.add(NetworkOperation(record, clock=Clock(dt=sim.config.block_ms * ms),
                                       when="end", order=1))
    started = time.perf_counter()
    observed = sim.run_headless(seconds)
    wall = time.perf_counter() - started
    events = np.asarray(samples, dtype=np.float32)
    antagonist = [[antagonist_correlation(events[:, leg, joint, :])
                   for joint in range(2)] for leg in range(6)]
    phase = observed["neighbor_phase_correlation"]
    contact = observed["contact_correlation"]
    return {
        "seed": seed, "simulated_seconds": seconds, "tonic_drive": tonic_drive,
        "plasticity": plasticity, "motor_gain": sim.config.motor_gain,
        "plastic_fraction": sim.config.motor_plastic_fraction,
        "plasticity_hebb_rate": sim.config.vnc_hebb_rate,
        "neurons": len(model.group),
        "synapses": len(model.synapses), "plastic_edges": len(model.plastic_edges),
        "wall_seconds": round(wall, 3),
        "realtime_percent": round(100 * seconds / wall, 2),
        "antagonist_correlation": np.round(antagonist, 4).tolist(),
        "mean_antagonist_correlation": round(float(np.mean(antagonist)), 4),
        "negative_antagonist_fraction": round(float(np.mean(np.asarray(antagonist) < 0)), 4),
        "net_path": observed["net_path"],
        "displacement": observed["displacement"],
        "path_length": observed["path_length"],
        "energy_spent": observed["energy_spent"],
        "net_energy": round(observed["displacement"] / max(observed["energy_spent"], 1e-9), 4),
        "mean_forward_flow": observed["mean_forward_flow"],
        "mean_abs_lateral_flow": observed["mean_abs_lateral_flow"],
        "mean_abs_rotation": observed["mean_abs_rotation"],
        "contralateral_phase": round(float(np.mean(phase[4:])), 4),
        "contralateral_contact_correlation": round(float(np.mean(contact[4:])), 4),
        "motor_saturation": observed["motor_saturation"],
        "motor_rates": observed["motor_rates"],
        "dn_rate_last_window": sim.rates[sim.names.index("dn")],
        "dan_rate_last_window": sim.rates[sim.names.index("dan")],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--tonic-drive", type=float, default=2)
    parser.add_argument("--backend", choices=("cython", "numpy"), default="cython")
    parser.add_argument("--plasticity", action="store_true")
    parser.add_argument("--motor-gain", type=float)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "results" / "scaffold_eval.json")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds deve ser positivo")
    rows = []
    for seed in args.seeds:
        changes = {"motor_gain": args.motor_gain} if args.motor_gain is not None else {}
        row = measure(seed, args.seconds, args.tonic_drive, args.plasticity,
                      args.backend, **changes)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
