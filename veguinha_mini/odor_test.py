"""Discriminação sensorial de quatro odores, sem corpo e sem treinamento."""

import json

import numpy as np
from brian2 import ms

from .config import Config
from .network import build_network, set_odor_pulse
from .verification import kenyon_fraction_per_50ms


def measure(config=Config()):
    model = build_network(config)
    kenyon = model.regions["kc"]
    rows = []
    codes = []
    for odor in (None, 0, 1, 2, 3):
        set_odor_pulse(model, 0 if odor is None else 12, odor or 0)
        first_spike = int(model.monitor.num_spikes)
        model.network.run(1000 * ms)
        end_ms = float(model.network.t / ms)
        last_spike = int(model.monitor.num_spikes)
        times = np.asarray(model.monitor.t[first_spike:last_spike] / ms)
        cells = np.asarray(model.monitor.i[first_spike:last_spike])
        is_kenyon = np.isin(cells, kenyon)
        recent = cells[is_kenyon & (times >= end_ms - 200)]
        codes.append(set(recent.tolist()))
        rows.append({
            "odor": "sem odor" if odor is None else odor,
            "kenyon_hz": round(float(is_kenyon.sum() / len(kenyon)), 3),
            "active_50ms": round(kenyon_fraction_per_50ms(
                times, cells, kenyon, end_ms - 500, end_ms), 4),
        })
    pairs = {}
    for first in range(4):
        for second in range(first + 1, 4):
            left, right = codes[first + 1], codes[second + 1]
            pairs[f"{first}-{second}"] = round(len(left & right) / max(1, len(left | right)), 3)
    baseline = rows[0]["kenyon_hz"]
    passed = (all(score < 0.20 for score in pairs.values())
              and all(row["kenyon_hz"] > max(0.5, baseline * 5) for row in rows[1:])
              and all(0.02 <= row["active_50ms"] <= 0.10 for row in rows[1:]))
    result = {"pass": passed, "odors": rows, "jaccard_200ms": pairs}
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")), flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(0 if measure()["pass"] else 1)
