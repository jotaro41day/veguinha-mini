import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from brian2 import ms

from veguinha_mini.config import Config
from veguinha_mini.motor import MotorDecoder
from veguinha_mini.network import build_network
from veguinha_mini.sensory import inject_sensors
from veguinha_mini.sim import Simulation


class Phase4Tests(unittest.TestCase):
    def test_motor_decoder_uses_pool_differences_and_limits(self):
        config = Config(scale=.05, backend="numpy")
        model = build_network(config, record_spikes=False)
        decoder = MotorDecoder(model, config)
        counts = np.zeros(len(model.group), dtype=int)
        counts[model.motor_pools["lf"]["hip_flex"]] = 1
        result = decoder.update(counts, .01)
        self.assertGreater(result[0, 0], 0)
        self.assertEqual(result[0, 1], 0)
        self.assertTrue(np.all(np.abs(result) <= config.motor_limit))
        counts[model.motor_pools["lf"]["hip_flex"]] = 100
        result = decoder.update(counts, .01)
        self.assertEqual(result[0, 0], config.motor_limit)
        self.assertGreater(decoder.saturation_fraction(), 0)

    def test_sensors_address_correct_leg_and_comparator(self):
        config = Config(scale=.1, backend="numpy", proprio_gain=4,
                        efference_gain=3, odor_gain=5)
        model = build_network(config, record_spikes=False)
        sample = {"angles": [[.5, -.2]] + [[0, 0]] * 5,
                  "contacts": [1, 0, 0, 0, 0, 0], "touch": True,
                  "energy": .5, "antenna_odor": [.6, .2]}
        signal = np.zeros((6, 2)); signal[0, 0] = 1
        inject_sensors(model, sample, odor_type=2, config=config, efference=signal)
        currents = np.asarray(model.group.I_sense[:])
        self.assertTrue(np.allclose(currents[model.motor_pools["lf"]["hip_extend"]], 2))
        self.assertTrue(np.allclose(currents[model.motor_pools["lm"]["hip_extend"]], 0))
        self.assertGreater(currents[model.regions["dn"]].mean(), 0)
        self.assertEqual(currents[np.array_split(model.regions["comparator"], 12)[0]].mean(), 3)
        odor = np.asarray(model.group.I_odor[:])
        left, right = np.array_split(np.array_split(model.regions["al_glom"], 4)[2], 2)
        self.assertTrue(np.allclose(odor[left], 3))
        self.assertTrue(np.allclose(odor[right], 1))

    def test_closed_loop_babbles_and_silent_network_stays_still(self):
        config = Config(scale=.05, backend="numpy")
        active = Simulation(config, headless=True)
        weights_before = np.asarray(active.model.synapses.w[:]).copy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            active.metrics_csv = path
            result = active.run_headless(2)
            self.assertGreater(result["displacement"], .05)
            self.assertGreater(result["mean_abs_torque"], .01)
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertGreaterEqual(len(rows), 1)
            self.assertIn("rate_vnc_lf", rows[0])
            self.assertIn("variance_lf_hip", rows[0])
        np.testing.assert_array_equal(np.asarray(active.model.synapses.w[:]), weights_before)
        silent = Simulation(config, headless=True)
        for name in silent.model.regions:
            silent.model.group.spike_gate[silent.model.regions[name]] = 0
        result = silent.run_headless(1)
        self.assertEqual(result["displacement"], 0)
        self.assertEqual(result["mean_abs_torque"], 0)
        self.assertEqual(result["path_length"], 0)
        self.assertTrue(np.isfinite(np.asarray(silent.model.group.v[:])).all())
        with self.assertRaises(ValueError):
            silent.submit({"cmd": "motor", "leg": "lf", "joint": 0, "value": 1})

    def test_live_mode_records_metrics_and_publishes_spikes(self):
        with tempfile.TemporaryDirectory() as directory:
            sim = Simulation(Config(scale=.05, backend="numpy"),
                             metrics_csv=Path(directory) / "live.csv")
            sim._open_live_metrics()
            try:
                sim.model.network.run(1010 * ms)
                self.assertGreater(sim.latest()["t"], 0)
                self.assertIn("metrics", sim.latest())
                self.assertGreater(sim.model.monitor.num_spikes, 0)
            finally:
                sim._close_metrics()
            with (Path(directory) / "live_epoch0.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["t_s"], "1.0")


if __name__ == "__main__":
    unittest.main()
