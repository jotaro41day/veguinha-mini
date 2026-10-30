"""Treino headless, controles pareados e checkpoints retomáveis da fase 5."""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
from brian2 import ms

from .config import Config
from .isolated import isolated_config
from .metrics import Metrics
from .sim import Simulation


def checkpoint_metrics(sim, wall_s, condition, measured_sim_s=None):
    observed = sim.metrics.snapshot(sim.body)
    window = (sim.window_metrics.snapshot(sim.body)
              if sim.window_metrics is not None else observed)
    simulated_s = sim.simulated_ms / 1000
    window_start_s = 0 if simulated_s < 900 else 900 if simulated_s < 2400 else 2400
    phases = observed["neighbor_phase_correlation"]
    weights = sim.motor_learning.summary()
    return {
        "condition": condition, "seed": sim.config.seed,
        "neurons": len(sim.model.group), "synapses": len(sim.model.synapses),
        "t_s": round(sim.simulated_ms / 1000, 3),
        "wall_s_per_sim_min": round(60 * wall_s / max(measured_sim_s or sim.simulated_ms / 1000, 1e-9), 2),
        "displacement": observed["displacement"],
        "path_length": observed["path_length"],
        "net_path": observed["net_path"],
        "energy_spent": observed["energy_spent"],
        "net_energy": round(observed["displacement"] / max(observed["energy_spent"], 1e-9), 6),
        "window_start_s": window_start_s,
        "window_displacement": window["displacement"],
        "window_path_length": window["path_length"],
        "window_energy_spent": window["energy_spent"],
        "window_net_path": window["net_path"],
        "window_net_energy": round(window["displacement"] /
                                   max(window["energy_spent"], 1e-9), 6),
        "window_contralateral_phase": round(float(np.mean(
            window["neighbor_phase_correlation"][4:])), 4),
        "window_contralateral_contact_correlation": round(float(np.mean(
            window["contact_correlation"][4:])), 4),
        "neighbor_phase": round(float(np.mean(phases[:4])), 4),
        "contralateral_phase": round(float(np.mean(phases[4:])), 4),
        "contact_fraction": round(float(np.mean(observed["contact_fraction"])), 4),
        "contralateral_contact_correlation": round(float(np.mean(
            observed["contact_correlation"][4:])), 4),
        "motor_saturation": round(sim.motor.saturation_fraction(), 6),
        "mean_forward_flow": observed["mean_forward_flow"],
        "mean_abs_lateral_flow": observed["mean_abs_lateral_flow"],
        "mean_abs_rotation": observed["mean_abs_rotation"],
        "wall_collisions": observed["wall_collisions"],
        "dan_hz": sim.rates[sim.names.index("dan")],
        "flow_hz": sim.rates[sim.names.index("flow")],
        "weight_min": round(weights["min"], 5),
        "weight_p05": round(weights["p05"], 5),
        "weight_p50": round(weights["p50"], 5),
        "weight_p95": round(weights["p95"], 5),
        "weight_max": round(weights["max"], 5),
        "weight_floor_fraction": round(weights["floor_fraction"], 6),
        "weight_ceiling_fraction": round(weights["ceiling_fraction"], 6),
        "dopamine": round(weights["dopamine"], 5),
        "octopamine_hz": round(weights["octopamine_hz"], 4),
    }


def train_one(seed, condition, seconds, checkpoint_s=60, destination=None,
              scale=1.0, backend="cython", resume=True, learning_rate=None,
              vnc_hebb_rate=None, value_rotation_scale=None,
              proprio_projection_strength=None, isolated=False,
              motor_scaffold=False, tonic_drive=2.0):
    if condition not in ("learning", "off", "shuffled"):
        raise ValueError("condição inválida")
    if seconds <= 0 or checkpoint_s <= 0:
        raise ValueError("duração inválida")
    destination = Path(destination or Path(__file__).parent / "results" / "phase5")
    destination.mkdir(parents=True, exist_ok=True)
    config_options = dict(seed=seed, scale=scale, backend=backend,
                          phase5_enabled=True, block_ms=20)
    if learning_rate is not None:
        config_options["learning_rate"] = learning_rate
    if vnc_hebb_rate is not None:
        config_options["vnc_hebb_rate"] = vnc_hebb_rate
    if value_rotation_scale is not None:
        config_options["value_rotation_scale"] = value_rotation_scale
    if proprio_projection_strength is not None:
        config_options["proprio_projection_strength"] = proprio_projection_strength
    if isolated:
        config_options.pop("scale")
        config_options.pop("phase5_enabled")
        config_options.pop("block_ms")
        config_options["motor_scaffold"] = motor_scaffold
        config = isolated_config(tonic_drive=tonic_drive, **config_options)
    else:
        config_options["motor_scaffold"] = motor_scaffold
        config = Config(**config_options)
    sim = Simulation(config, headless=True,
                     reward_mode="shuffled" if condition == "shuffled" else "coupled")
    sim.window_metrics = Metrics(sim.body) if condition != "off" else None
    sim.plasticity = condition != "off"
    stem = destination / f"{condition}_seed{seed}"
    if resume and stem.with_suffix(".pkl").exists():
        sim.load_checkpoint(stem)
    if condition == "off":
        sim.window_metrics = None
    rows_file = stem.with_suffix(".csv")
    existing = rows_file.exists() and rows_file.stat().st_size > 0 and resume
    if existing:
        with rows_file.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(checkpoint_metrics(sim, 0.0, condition))
            retained = [record for record in reader
                        if float(record["t_s"]) <= sim.simulated_ms / 1000 + 1e-6]
        if sim.simulated_ms and (not retained or float(retained[-1]["t_s"]) < sim.simulated_ms / 1000 - 1e-6):
            recovered = checkpoint_metrics(sim, 0.0, condition)
            recovered["wall_s_per_sim_min"] = ""
            retained.append(recovered)
        with rows_file.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows({key: record.get(key, "") for key in fieldnames}
                             for record in retained)
    started = time.perf_counter()
    start_sim_s = sim.simulated_ms / 1000
    with rows_file.open("a" if existing else "w", newline="", encoding="utf-8") as handle:
        writer = None
        while sim.simulated_ms / 1000 < seconds - 1e-6:
            remaining = seconds - sim.simulated_ms / 1000
            chunk = min(checkpoint_s, remaining)
            sim.model.network.run(chunk * 1000 * ms)
            row = checkpoint_metrics(sim, time.perf_counter() - started, condition,
                                     sim.simulated_ms / 1000 - start_sim_s)
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if not existing:
                    writer.writeheader()
            sim.save_checkpoint(stem)
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
            print(json.dumps(row, ensure_ascii=False), flush=True)
    return row if writer else checkpoint_metrics(sim, time.perf_counter() - started, condition)


def summarize(destination, seeds, seconds, baseline_file=None):
    """Compara cada treino ao controle sem plasticidade da mesma seed."""
    destination = Path(destination)
    reports = []
    for seed in seeds:
        conditions = {}
        for name in ("learning", "off", "shuffled"):
            path = destination / f"{name}_seed{seed}.csv"
            if not path.exists():
                continue
            with path.open(newline="", encoding="utf-8") as handle:
                matches = [row for row in csv.DictReader(handle)
                           if abs(float(row["t_s"]) - seconds) < 1e-6]
            if matches:
                conditions[name] = matches[-1]
        report = {"seed": seed, "t_s": seconds, "available": sorted(conditions)}
        for name, row in conditions.items():
            report[f"{name}_net_path"] = float(row["net_path"])
            report[f"{name}_net_energy"] = float(row["net_energy"])
            report[f"{name}_contralateral_contact_correlation"] = float(
                row.get("contralateral_contact_correlation", 0))
        if "off" in conditions:
            report["off_net_energy"] = float(conditions["off"]["net_energy"])
        if "learning" in conditions and "off" in conditions:
            report["energy_gain_vs_off"] = (float(conditions["learning"]["net_energy"])
                                            / max(report["off_net_energy"], 1e-9))
            report["net_path_gain_vs_off"] = (float(conditions["learning"]["net_path"])
                                              / max(float(conditions["off"]["net_path"]), 1e-9))
        if len(conditions) == 3:
            learned = conditions["learning"]
            report["gait_evidence"] = (float(learned["contralateral_phase"]) < -.1
                                       and float(learned["contralateral_contact_correlation"]) < -.1)
            report["pass"] = (float(learned["net_path"]) > .2
                              and report["gait_evidence"]
                              and all(float(learned["net_path"]) > float(conditions[name]["net_path"])
                                      and float(learned["net_energy"]) > float(conditions[name]["net_energy"])
                                      for name in ("off", "shuffled")))
        else:
            report["pass"] = None
        reports.append(report)
    for report in reports:
        print(json.dumps(report, ensure_ascii=False))
    print(json.dumps({"all_five_pass": len(reports) == 5 and all(
        report["pass"] is True for report in reports)}, ensure_ascii=False))
    return reports


def main():
    parser = argparse.ArgumentParser(description="Treino motor pareado da Veguinha mini")
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--checkpoint-s", type=float, default=60)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--conditions", nargs="+", choices=("learning", "off", "shuffled"),
                        default=["learning", "off", "shuffled"])
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--backend", choices=("numpy", "cython"), default="cython")
    parser.add_argument("--isolated", action="store_true")
    parser.add_argument("--motor-scaffold", action="store_true")
    parser.add_argument("--tonic-drive", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--vnc-hebb-rate", type=float)
    parser.add_argument("--value-rotation-scale", type=float)
    parser.add_argument("--proprio-projection-strength", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--summary", action="store_true",
                        help="compara checkpoints pareados e encerra")
    args = parser.parse_args()
    if args.summary:
        summarize(args.output or Path(__file__).parent / "results" / "phase5",
                  args.seeds, args.seconds)
        return
    for seed in args.seeds:
        for condition in args.conditions:
            train_one(seed, condition, args.seconds, args.checkpoint_s, args.output,
                      args.scale, args.backend, not args.restart, args.learning_rate,
                      args.vnc_hebb_rate, args.value_rotation_scale,
                      args.proprio_projection_strength, args.isolated,
                      args.motor_scaffold, args.tonic_drive)


if __name__ == "__main__":
    main()
