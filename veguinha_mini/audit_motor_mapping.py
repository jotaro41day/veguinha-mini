"""Diagnóstico de 20 s: sinal neural → torque → ângulo em cada junta."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .body import LEG_NAMES
from .config import Config
from .sim import Simulation


def measure(seed=42, seconds=20, scale=1, backend="cython", destination=None,
            physics="legacy"):
    sim = Simulation(Config(seed=seed, scale=scale, backend=backend,
                            block_ms=20, phase5_enabled=False),
                     headless=True, body_substep_ms=2, physics=physics)
    samples = []
    original = sim.metrics.update

    def observe(body, torques, contacts):
        original(body, torques, contacts)
        samples.append((np.array(torques, copy=True), np.array(body.angles, copy=True),
                        np.array(sim.motor.filtered_hz, copy=True)))

    sim.metrics.update = observe
    sim.run_headless(seconds)
    torque = np.stack([x[0] for x in samples])
    angle = np.stack([x[1] for x in samples])
    rate = np.stack([x[2] for x in samples])
    rows = []
    for index, leg in enumerate(LEG_NAMES):
        for joint, names in enumerate((("hip_flex", "hip_extend"),
                                        ("knee_flex", "knee_extend"))):
            command_rms = float(np.sqrt(np.mean(torque[:, index, joint] ** 2)))
            gear = (float(sim.body.model.actuator_gear[index * 2 + joint, 0])
                    if physics == "mujoco" else 1.0)
            rows.append({
                "leg": leg, "joint": "hip" if joint == 0 else "knee",
                "torque_rms": round(command_rms, 4),
                "torque_rms_nm": round(command_rms * gear, 5),
                "angle_p95_p5_rad": round(float(np.percentile(angle[:, index, joint], 95) -
                                                np.percentile(angle[:, index, joint], 5)), 4),
                "flex_hz": round(float(rate[:, index, joint * 2].mean()), 2),
                "extend_hz": round(float(rate[:, index, joint * 2 + 1].mean()), 2),
            })
    if destination:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seconds", type=int, default=20)
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--backend", choices=("cython", "numpy"), default="cython")
    parser.add_argument("--physics", choices=("legacy", "mujoco"), default="legacy")
    parser.add_argument("--output", default="veguinha_mini/results/motor_mapping_20s.csv")
    args = parser.parse_args()
    print(json.dumps(measure(args.seed, args.seconds, args.scale, args.backend,
                             args.output, args.physics), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
