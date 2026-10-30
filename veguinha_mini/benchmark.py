"""Benchmark curto, com um processo limpo por escala."""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import replace

from brian2 import ms

from .config import Config
from .network import build_network


def rss_mb():
    with open("/proc/self/status", encoding="utf-8") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    raise RuntimeError("VmRSS indisponível")


def measure(scale):
    config = replace(Config(), scale=scale, backend="cython")
    start = time.perf_counter()
    model = build_network(config)
    built = time.perf_counter()
    model.network.run(100 * ms)
    warmed = time.perf_counter()
    model.network.run(1000 * ms)
    elapsed = time.perf_counter() - warmed
    assert model.monitor.num_spikes > 0
    return {
        "scale": scale,
        "neurons": len(model.group),
        "synapses": len(model.synapses),
        "rss_mb": round(rss_mb(), 1),
        "build_s": round(built - start, 2),
        "warmup_s": round(warmed - built, 2),
        "wall_per_sim_s": round(elapsed, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float)
    args = parser.parse_args()
    if args.scale is not None:
        print(json.dumps(measure(args.scale)))
    else:
        for scale in (1, 2, 5, 10):
            result = subprocess.run(
                [sys.executable, "-m", "veguinha_mini.benchmark", "--scale", str(scale)],
                check=True, capture_output=True, text=True,
            )
            print(result.stdout.strip(), flush=True)


if __name__ == "__main__":
    main()
