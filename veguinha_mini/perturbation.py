"""Mede readaptação após retirar torque e apoio de uma perna."""

import argparse
import csv
from pathlib import Path

import numpy as np
from brian2 import ms

from .body import Body
from .config import Config
from .metrics import Metrics
from .sensory import inject_sensors
from .sim import Simulation


def measure(checkpoint, leg="lf", seconds=180, interval_s=30, baseline_s=30, plasticity=True,
            seed=42, scale=1.0, backend="cython"):
    if seconds <= 0 or interval_s <= 0:
        raise ValueError("duração inválida")
    sim = Simulation(Config(seed=seed, scale=scale, backend=backend,
                            phase5_enabled=True, block_ms=20), headless=True)
    sim.load_checkpoint(checkpoint)
    sim.body = Body(sim.world)
    sim.metrics = Metrics(sim.body)
    sim.contacts[:] = True
    inject_sensors(sim.model, sim.body.snapshot(sim.world, sim.contacts), config=sim.config)
    sim.plasticity = False
    rows = []

    def run_window(chunk, elapsed, damaged):
        start_position = sim.body.position.copy()
        start_path = sim.metrics.path_length
        start_energy = sim.metrics.energy_spent
        start_phase = sim.metrics.phase_sum.copy()
        start_phase_count = sim.metrics.phase_count.copy()
        start_saturated = sim.motor.saturation_samples
        start_motor_samples = sim.motor.samples
        sim.model.network.run(chunk * 1000 * ms)
        net = float(np.linalg.norm(sim.body.position - start_position))
        path = sim.metrics.path_length - start_path
        energy = sim.metrics.energy_spent - start_energy
        phase_sum = sim.metrics.phase_sum[4:] - start_phase[4:]
        phase_count = sim.metrics.phase_count[4:] - start_phase_count[4:]
        rows.append({"leg": leg, "plasticity": plasticity, "damaged": damaged,
                     "elapsed_s": round(elapsed, 2),
                     "net_path": round(net / max(path, 1e-9), 6),
                     "net_energy": round(net / max(energy, 1e-9), 4),
                     "path": round(path, 4), "energy": round(energy, 5),
                     "motor_saturation": round((sim.motor.saturation_samples - start_saturated)
                                               / max(1, sim.motor.samples - start_motor_samples), 5),
                     "weight_p95": round(sim.motor_learning.summary()["p95"], 4),
                     "contralateral_phase": round(float(np.mean(
                         phase_sum / np.maximum(phase_count, 1))), 4)})

    if baseline_s > 0:
        run_window(baseline_s, 0, False)
    sim.plasticity = plasticity
    sim.submit({"cmd": "damage", "leg": leg, "value": True})
    elapsed = 0.0
    while elapsed < seconds - 1e-6:
        chunk = min(interval_s, seconds - elapsed)
        run_window(chunk, elapsed + chunk, True)
        elapsed += chunk
    return rows


def main():
    parser = argparse.ArgumentParser(description="Perturbação de uma perna após treino")
    parser.add_argument("checkpoint", type=Path, help="prefixo do checkpoint .pkl")
    parser.add_argument("--leg", choices=("lf", "lm", "lh", "rf", "rm", "rh"), default="lf")
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--interval-s", type=float, default=30)
    parser.add_argument("--baseline-s", type=float, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--backend", choices=("cython", "numpy"), default="cython")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = []
    for plasticity in (True, False):
        rows.extend(measure(args.checkpoint, args.leg, args.seconds, args.interval_s,
                            args.baseline_s,
                            plasticity, args.seed, args.scale, args.backend))
    destination = args.output or args.checkpoint.with_name(args.checkpoint.name + "_perturbation.csv")
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
