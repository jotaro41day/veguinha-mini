"""Short reduced Brian2-to-MuJoCo check, with no motor plasticity."""

import csv
import time
from pathlib import Path

from .config import Config
from .metrics import Metrics
from .mujoco_probe import run_probe
from .sim import Simulation


def compare(seeds=(42, 43, 44), seconds=9.5, scale=.05):
    rows = []
    for seed in seeds:
        for mode in ("random", "tripod"):
            probe = run_probe(seed, mode, seconds=seconds)
            rows.append({"seed": seed, "mode": mode, "neurons": 0,
                         "velocity_m_s": probe["velocity_m_s"],
                         "support": probe["support"], "falls": probe["falls"],
                         "wall_s": probe["wall_s"]})
        start = time.perf_counter()
        simulation = Simulation(
            Config(seed=seed, scale=scale, backend="cython", block_ms=20,
                   phase5_enabled=False), headless=True, body_substep_ms=2,
            physics="mujoco")
        neurons = len(simulation.model.group)
        simulation.run_headless(2)
        simulation.window_metrics = Metrics(simulation.body)
        simulation.run_headless(seconds - 2)
        score = simulation.window_metrics.snapshot(simulation.body)
        rows.append({"seed": seed, "mode": "rede_sem_plasticidade",
                     "neurons": neurons, "velocity_m_s": score["mean_forward_flow"],
                     "support": sum(score["contact_fraction"]) / 6,
                     "falls": int(simulation.falls_seen),
                     "wall_s": time.perf_counter() - start})
    return rows


def main():
    rows = compare()
    destination = Path("veguinha_mini/results/mujoco_network_short_v1.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row, flush=True)


if __name__ == "__main__":
    main()
