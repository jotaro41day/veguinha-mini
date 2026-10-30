import unittest
import time

import numpy as np

from veguinha_mini.config import Config
from veguinha_mini.learning import kenyon_homeostasis
from veguinha_mini.network import build_network, set_odor_pulse
from veguinha_mini.sim import Simulation


class NetworkControlsTests(unittest.TestCase):
    def test_simulation_thread_handles_pause_and_ablation(self):
        sim = Simulation(Config(scale=0.05, backend="numpy"))
        self.assertEqual(sim.latest()["epoch"], 0)
        sim.start()
        try:
            sim.submit({"cmd": "ablate", "region": "kc", "value": True})
            sim.submit({"cmd": "pause", "value": True})
            deadline = time.monotonic() + 5
            while not sim.latest()["paused"] and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(sim.latest()["paused"])
            self.assertIn("kc", sim.latest()["ablated"])
            frozen = sim.latest()["t"]
            time.sleep(0.15)
            self.assertEqual(sim.latest()["t"], frozen)
            sim.submit({"cmd": "reset"})
            deadline = time.monotonic() + 5
            while sim.latest()["epoch"] == 0 and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(sim.latest()["epoch"], 1)
            self.assertEqual(sim.latest()["ablated"], [])
        finally:
            sim.stop()

    def test_odor_input_and_homeostasis(self):
        model = build_network(Config(scale=0.1, backend="numpy"))
        self.assertTrue(np.all(np.asarray(model.group.spike_gate[:]) == 1))
        set_odor_pulse(model, 12)
        odor = np.asarray(model.group.I_odor[:])
        self.assertGreater(np.count_nonzero(odor), 0)
        self.assertEqual(np.count_nonzero(odor[model.regions["kc"]]), 0)
        before = np.asarray(model.group.I_base[:])[model.regions["kc"]].copy()
        kenyon_homeostasis(model, observed_hz=10)
        after = np.asarray(model.group.I_base[:])[model.regions["kc"]]
        self.assertTrue(np.all(after < before))
        set_odor_pulse(model, 0)
        self.assertEqual(np.count_nonzero(np.asarray(model.group.I_odor[:])), 0)
        model.network.run(0 * model.ms)


if __name__ == "__main__":
    unittest.main()
