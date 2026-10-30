"""Verificação de 30 s simulados em janelas de 5 s."""

import json

import numpy as np
from brian2 import ms

from .config import Config
from .learning import kenyon_homeostasis
from .network import build_network, set_odor_pulse


def kenyon_fraction_per_50ms(times, neurons, kenyon, start_ms, end_ms):
    bins = round((end_ms - start_ms) / 50)
    mask = ((times >= start_ms) & (times < end_ms) &
            (neurons >= kenyon[0]) & (neurons <= kenyon[-1]))
    if not mask.any():
        return 0.0
    bucket = ((times[mask] - start_ms) // 50).astype(np.int32)
    cell = neurons[mask] - kenyon[0]
    unique = np.unique(bucket * len(kenyon) + cell)
    return len(unique) / (bins * len(kenyon))


def verify(config=Config()):
    model = build_network(config)
    previous = np.zeros(len(model.group), dtype=np.int64)
    rows = []
    glomeruli = model.regions["al_glom"][:max(1, len(model.regions["al_glom"]) // 2)]
    for window in range(6):
        set_odor_pulse(model, 12 if window == 2 else 0)
        model.network.run(5000 * ms)
        current = np.asarray(model.monitor.count[:], dtype=np.int64)
        delta = current - previous
        previous = current
        rates = {name: round(float(delta[indices].sum() / len(indices) / 5), 2)
                 for name, indices in model.regions.items()}
        motor_rates = {}
        for leg, pools in model.motor_pools.items():
            motor_indices = np.concatenate(tuple(pools.values()))
            motor_rates[leg] = round(float(delta[motor_indices].sum() /
                                           len(motor_indices) / 5), 2)
        times = np.asarray(model.monitor.t[:] / ms)
        neurons = np.asarray(model.monitor.i[:], dtype=np.int32)
        recent = neurons[times >= (window + 1) * 5000 - 50]
        last_sparse = round(float(np.isin(model.regions["kc"], recent).mean()), 4)
        sparse = round(kenyon_fraction_per_50ms(times, neurons, model.regions["kc"],
                                                 window * 5000, (window + 1) * 5000), 4)
        odor_rate = round(float(delta[glomeruli].sum() / len(glomeruli) / 5), 2)
        rows.append({"window_s": [window * 5, (window + 1) * 5], "odor": window == 2,
                     "kc_active_50ms_mean": sparse, "kc_last_50ms": last_sparse,
                     "odor_glom_hz": odor_rate,
                     "motor_hz": motor_rates,
                     "rates_hz": rates})
        kenyon_homeostasis(model, rates["kc"])
        print(json.dumps(rows[-1], separators=(",", ":")), flush=True)
    baseline = (rows[0]["odor_glom_hz"] + rows[1]["odor_glom_hz"]) / 2
    pulse_ratio = rows[2]["odor_glom_hz"] / baseline
    projection_baseline = (rows[0]["rates_hz"]["al_proj"] + rows[1]["rates_hz"]["al_proj"]) / 2
    kenyon_baseline = (rows[0]["rates_hz"]["kc"] + rows[1]["rates_hz"]["kc"]) / 2
    projection_ratio = rows[2]["rates_hz"]["al_proj"] / projection_baseline
    kenyon_ratio = rows[2]["rates_hz"]["kc"] / kenyon_baseline
    stable = all((0 <= rate < 80 if name == "kc" and not row["odor"] else 0.1 < rate < 80)
                 for row in rows for name, rate in row["rates_hz"].items())
    sparse = all((0.05 <= row["kc_active_50ms_mean"] <= 0.10 if row["odor"]
                  else row["kc_active_50ms_mean"] <= 0.02) for row in rows)
    passed = stable and sparse and pulse_ratio >= 1.3 and projection_ratio >= 1.1 and kenyon_ratio >= 1.05
    summary = {"pass": passed, "stable": stable, "sparse": sparse,
               "odor_pulse_ratio": round(pulse_ratio, 2),
               "projection_pulse_ratio": round(projection_ratio, 2),
               "kenyon_pulse_ratio": round(kenyon_ratio, 2), "neurons": len(model.group),
               "synapses": len(model.synapses)}
    print(json.dumps(summary, separators=(",", ":")), flush=True)
    return summary


if __name__ == "__main__":
    raise SystemExit(0 if verify()["pass"] else 1)
