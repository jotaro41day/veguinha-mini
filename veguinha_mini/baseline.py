"""Linha de base da fase 4: 300 s simulados, uma seed por processo."""

import argparse
import csv
import json
from pathlib import Path
import time

from .config import Config
from .metrics import net_path_ratio
from .sim import Simulation


def run(seed, seconds=300, output_dir=None):
    directory = Path(output_dir or Path(__file__).parent / "results")
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    sim = Simulation(Config(seed=seed), headless=True,
                     metrics_csv=directory / f"baseline_seed{seed}.csv")
    result = sim.run_headless(seconds)
    result.update(seed=seed, simulated_seconds=seconds,
                  wall_seconds=round(time.perf_counter() - started, 2))
    (directory / f"baseline_seed{seed}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def summarize(directory):
    directory = Path(directory)
    records = [json.loads((directory / f"baseline_seed{seed}.json").read_text())
               for seed in (42, 43, 44)]
    for record in records:
        record["displacement_per_energy"] = round(
            record["displacement"] / max(record["energy_spent"], 1e-9), 5)
        record["net_over_path"] = round(record.get(
            "net_path", net_path_ratio(record["displacement"], record["path_length"])), 5)
    keys = ("displacement", "path_length", "mean_speed", "energy_spent",
            "mean_abs_torque", "motor_saturation", "displacement_per_energy",
            "net_over_path")
    means = {key: round(sum(item[key] for item in records) / len(records), 5)
             for key in keys}
    with (directory / "baseline_summary.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["seed", *keys])
        for record in records:
            writer.writerow([record["seed"], *(record[key] for key in keys)])
        writer.writerow(["mean", *(means[key] for key in keys)])
    return means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "results"))
    args = parser.parse_args()
    result = summarize(args.output_dir) if args.summary else run(args.seed, args.seconds, args.output_dir)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
