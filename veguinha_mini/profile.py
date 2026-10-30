"""Profile reproduzível da mesma rede com atraso por sinapse ou atraso fixo."""

import argparse
import json
import time

import numpy as np
from brian2 import ms

from .config import Config
from .network import build_network


def measure(variable_delays=False):
    model = build_network(Config())
    if variable_delays:
        rng = np.random.default_rng(31)
        model.synapses.delay = rng.integers(1, 6, len(model.synapses)) * ms
    model.network.run(100 * ms)
    start = time.perf_counter()
    model.network.run(1000 * ms, profile=True)
    wall = time.perf_counter() - start
    result = {"delay": "1–5 ms por sinapse" if variable_delays else "2 ms fixo",
              "neurons": len(model.group), "synapses": len(model.synapses),
              "wall_s": round(wall, 3), "real_time_percent": round(100 / wall, 1),
              "components_s": {name: round(float(value / ms) / 1000, 3)
                               for name, value in model.network.profiling_info}}
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variable-delays", action="store_true")
    args = parser.parse_args()
    measure(args.variable_delays)
