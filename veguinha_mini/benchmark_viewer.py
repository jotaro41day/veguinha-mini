"""Compare identical 20 ms Brian2/body runs with mjviser drawing off and on.

The Viser server stays open in both intervals; only pose publication changes.
This measures renderer overhead without resetting the network between intervals.
"""

import argparse
import time

from .config import Config
from .mjviser_bridge import MjviserBridge
from .sim import Simulation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=.05)
    parser.add_argument("--backend", choices=("cython", "numpy"), default="cython")
    parser.add_argument("--wall-seconds", type=float, default=6)
    parser.add_argument("--port", type=int, default=8012)
    args = parser.parse_args()
    sim = Simulation(Config(scale=args.scale, backend=args.backend,
                            block_ms=20, phase5_enabled=False), body_substep_ms=2,
                     physics="mujoco")
    sim.speed = 20
    bridge = MjviserBridge(sim, port=args.port)
    bridge.active = False
    bridge.force_render = True
    sim.start()
    bridge.start()
    try:
        time.sleep(3)  # Discard Brian2 startup and Cython cache effects.
        for label, enabled in (("desligado", False), ("ligado", True),
                               ("desligado_repetição", False)):
            bridge.active = enabled
            before_ms = sim.simulated_ms
            before_frames = bridge.frames
            before_render_s = bridge.render_seconds
            start = time.perf_counter()
            time.sleep(args.wall_seconds)
            wall = time.perf_counter() - start
            simulated = (sim.simulated_ms - before_ms) / 1000
            frames = bridge.frames - before_frames
            render_s = bridge.render_seconds - before_render_s
            print(f"{label:22} {simulated:7.3f} s simulados / {wall:.3f} s reais "
                  f"= {100*simulated/wall:6.1f}% do tempo real; "
                  f"{frames/wall:4.1f} fps; {1000*render_s/max(frames,1):.1f} ms/quadro",
                  flush=True)
    finally:
        bridge.stop()
        sim.stop()


if __name__ == "__main__":
    main()
