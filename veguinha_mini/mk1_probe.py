"""Short mechanics checks and optional viewer for the articulated mk-1."""
import argparse
import json
import time
import threading
from pathlib import Path

import mujoco
import numpy as np
from .mujoco_body import MujocoBody
from .world import World


def validate():
    rows = []
    for variant in ("control", "mk1"):
        for dt in (.004, .002, .001):
            b = MujocoBody(dt=dt, variant=variant)
            start = time.perf_counter()
            for _ in range(250):
                b.step(.02, np.zeros_like(b.angles))
            rows.append(dict(body=variant, dt_ms=1000 * dt,
                             wall_s_per_sim_s=(time.perf_counter() - start) / 5,
                             mass_kg=float(b.model.body_mass.sum()),
                             tilt_degrees=float(np.degrees(np.arccos(np.clip(b.upright, -1, 1)))),
                             height_m=float(b.data.qpos[2]), loads_N=b.normal_load.tolist(),
                             com_m=b.data.subtree_com[b.body_id].tolist()))
    pulses = []
    for leg in range(6):
        for joint in range(3):
            b = MujocoBody(variant="mk1", dt=.001)
            b.model.opt.gravity[:] = 0
            b.model.jnt_stiffness[:] = 0
            b.model.dof_damping[:] = 0
            b.data.qpos[2] += .5  # Remove floor reactions in this sign-only test.
            command = np.zeros((6, 3))
            command[leg, joint] = .1
            b.step(.01, command)
            pulses.append(float(b.angles[leg, joint]))
    assert min(pulses) > 0, "positive torque produced negative joint motion"
    b = MujocoBody(variant="mk1")
    b.data.qpos[3:7] = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]
    for _ in range(100):
        b.step(.02, np.zeros((6, 3)))
    report = dict(passive=rows, positive_torque_angle_rad=pulses,
                  forced_fall_detected=bool(b.fell), forced_fall_upright=b.upright)
    destination = Path(__file__).parent / "results/mk1_stage3/validation.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def view():
    from .mjviser_bridge import MjviserBridge
    world = World()

    class Preview:
        def __init__(self):
            self.body = MujocoBody(world, variant="mk1")
            self.seq = 0
            self.lock = threading.Lock()

        def latest(self):
            with self.lock:
                return dict(body=self.body.snapshot(world), world=world.snapshot(),
                            epoch=0, seq=self.seq, speed=1)

    preview = Preview()
    bridge = MjviserBridge(preview)
    bridge.start()
    try:
        while True:
            start = time.perf_counter()
            with preview.lock:
                preview.body.step(.02, np.zeros((6, 3)))
                preview.seq += 1
            time.sleep(max(0, .02 - (time.perf_counter() - start)))
    except KeyboardInterrupt:
        pass
    finally:
        bridge.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", action="store_true")
    args = parser.parse_args()
    view() if args.view else validate()
