"""Resume ensaios pareados de corpo/tripé já concluídos; não simula nada."""

import csv
import json
from pathlib import Path

import numpy as np

from .phase_lock_audit import _save_atomic
from .tripod_audit import EDGE_PAIRS
from .tripod_body_diagnostic import (_phase_contact_r, _phase_from_counts,
                                      CONDITIONS)


KEYS = ("net_path_pct", "net_energy", "contact_alternation", "phase_contact_r",
        "phase_torque_r", "same_tripod_contact_r", "all_three_contact_fraction",
        "mean_contact_fraction", "impulse_forward_per_cycle", "torque_rms",
        "realized_relative_phase_sd_deg")


def _mean_sd(values):
    data = np.asarray(values, dtype=float)
    return {"mean": round(float(data.mean()), 4),
            "sd": round(float(data.std(ddof=1)), 4)}


def _correlation(first, second):
    return float(np.corrcoef(first, second)[0, 1]) if np.std(first) and np.std(second) else 0.0


def report(input_path):
    input_path = Path(input_path)
    source = json.loads(input_path.read_text())
    rows = source["rows"]
    expected = {(seed, condition) for seed in range(42, 52) for condition in CONDITIONS}
    if {(row["seed"], row["condition"]) for row in rows} != expected or len(rows) != 70:
        raise ValueError("os 70 ensaios pareados precisam estar completos")
    flat = []
    for row in rows:
        data = np.load(input_path.parent / row["trace"])
        contact = data["contacts"][100:-50]
        edge_r = [_correlation(contact[:, a], contact[:, b]) for a, b in EDGE_PAIRS]
        premotor = data["premotor_counts"]
        neural_torque = None
        if len(premotor):
            phase = _phase_from_counts(premotor)[100:-50]
            torque = data["torques"][100:-50]
            neural_torque = float(np.mean([_phase_contact_r(phase[:, leg], torque[:, leg, joint])
                                           for leg in range(6) for joint in range(2)]))
        flat.append({"seed": row["seed"], "condition": row["condition"],
                     "net_path_pct": round(100 * row["net_path"], 4),
                     "net_energy": row["net_energy"],
                     "contact_alternation": round(float(-np.mean(edge_r)), 4),
                     "phase_contact_r": round(float(np.mean(row["phase_contact_r"])), 4),
                     "phase_torque_r": None if neural_torque is None else round(neural_torque, 4),
                     "same_tripod_contact_r": row["same_tripod_contact_r"],
                     "all_three_contact_fraction": round(float(np.mean(row["all_three_contact_fraction"])), 4),
                     "mean_contact_fraction": round(float(np.mean(row["contact_fraction_by_leg"])), 4),
                     "impulse_forward_per_cycle": row["impulse_forward_per_cycle"],
                     "torque_rms": round(float(np.mean(row["torque_rms_by_joint"])), 5),
                     "realized_relative_phase_sd_deg": row["realized_relative_phase_sd_deg"]})
    summary = {}
    for condition in CONDITIONS:
        selected = [row for row in flat if row["condition"] == condition]
        summary[condition] = {key: _mean_sd([row[key] for row in selected if row[key] is not None])
                              for key in KEYS if any(row[key] is not None for row in selected)}
    per_seed = {seed: {row["condition"]: row for row in flat if row["seed"] == seed}
                for seed in range(42, 52)}
    summary["paired"] = {
        "imposed_above_control": sum(per_seed[s]["imposto0"]["net_path_pct"] >
                                     per_seed[s]["controle"]["net_path_pct"] for s in per_seed),
        "jitter35_below_imposed": sum(per_seed[s]["jitter35"]["net_path_pct"] <
                                        per_seed[s]["imposto0"]["net_path_pct"] for s in per_seed),
        "jitter35_above_control": sum(per_seed[s]["jitter35"]["net_path_pct"] >
                                        per_seed[s]["controle"]["net_path_pct"] for s in per_seed),
        "network_above_control": sum(per_seed[s]["rede18"]["net_path_pct"] >
                                     per_seed[s]["controle"]["net_path_pct"] for s in per_seed),
        "shuffled_above_network": sum(per_seed[s]["embaralhado"]["net_path_pct"] >
                                       per_seed[s]["rede18"]["net_path_pct"] for s in per_seed),
        "imposed_minus_control_pct_points": _mean_sd([
            per_seed[s]["imposto0"]["net_path_pct"] - per_seed[s]["controle"]["net_path_pct"]
            for s in per_seed]),
        "jitter35_minus_imposed_pct_points": _mean_sd([
            per_seed[s]["jitter35"]["net_path_pct"] - per_seed[s]["imposto0"]["net_path_pct"]
            for s in per_seed])}
    summary["by_leg"] = {}
    for condition in ("rede18", "imposto0"):
        selected = [row for row in rows if row["condition"] == condition]
        for leg, name in enumerate(("LF", "LM", "LH", "RF", "RM", "RH")):
            phasor = np.exp(1j * np.deg2rad([row["phase_deg_vs_lf"][leg] for row in selected])).mean()
            phase_sd = np.rad2deg(np.sqrt(-2 * np.log(max(abs(phasor), 1e-12))))
            summary["by_leg"][f"{condition}_{name}"] = {
                "phase_mean_deg": round(float(np.angle(phasor, deg=True)), 2),
                "phase_sd_across_seeds_deg": round(float(phase_sd), 2),
                "contact_fraction": _mean_sd([row["contact_fraction_by_leg"][leg] for row in selected]),
                "phase_contact_r": _mean_sd([row["phase_contact_r"][leg] for row in selected]),
                "impulse_forward_per_cycle": _mean_sd([
                    row["impulse_forward_per_cycle_by_leg"][leg] for row in selected]),
                "torque_hip_rms": _mean_sd([row["torque_rms_by_leg"][leg][0] for row in selected]),
                "torque_knee_rms": _mean_sd([row["torque_rms_by_leg"][leg][1] for row in selected])}
    result_path = input_path.with_name("tripod_body_diagnostic_summary_v2.json")
    _save_atomic(result_path, summary)
    csv_path = input_path.with_name("tripod_body_diagnostic_per_seed_v2.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("seed", "condition", *KEYS))
        writer.writeheader(); writer.writerows(flat)
    print(result_path, csv_path, sep="\n")
    return summary


if __name__ == "__main__":
    report(Path(__file__).parent / "results" / "tripod_body_diagnostic_v2.json")
