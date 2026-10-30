import unittest
import tempfile
import pickle
from pathlib import Path

import numpy as np

from veguinha_mini.isolated import isolated_config, isolated_simulation
from veguinha_mini.network import build_network
from veguinha_mini.training import train_one


class IsolatedModeTests(unittest.TestCase):
    def test_off_control_skips_unused_window_observer(self):
        with tempfile.TemporaryDirectory() as directory:
            train_one(42, "off", .02, checkpoint_s=.02,
                      destination=directory, backend="numpy",
                      isolated=True, motor_scaffold=True)
            with (Path(directory) / "off_seed42.pkl").open("rb") as handle:
                saved = pickle.load(handle)
            self.assertIsNone(saved["window_metrics"])

    def test_scaffold_training_checkpoint_can_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            first = train_one(42, "learning", .04, checkpoint_s=.02,
                              destination=directory, backend="numpy",
                              isolated=True, motor_scaffold=True)
            self.assertAlmostEqual(first["t_s"], .04)
            self.assertEqual(first["neurons"], 869)
            resumed = train_one(42, "learning", .06, checkpoint_s=.02,
                                destination=directory, backend="numpy",
                                isolated=True, motor_scaffold=True)
            self.assertAlmostEqual(resumed["t_s"], .06)
            self.assertEqual(len((Path(directory) / "learning_seed42.csv").read_text().splitlines()), 4)

    def test_motor_core_is_small_and_descending_drive_is_tunable(self):
        quiet = isolated_config(seed=42, backend="numpy", tonic_drive=0)
        driven = isolated_config(seed=42, backend="numpy", tonic_drive=3)
        self.assertEqual(sum(region.cells for region in driven.regions)
                         + 96 + 48 + 32, 869)
        model_quiet = build_network(quiet, record_spikes=False)
        model_driven = build_network(driven, record_spikes=False)
        self.assertEqual(len(model_driven.group), 869)
        self.assertEqual(sum(len(model_driven.regions[f"vnc_{leg}"])
                             for leg in ("lf", "lm", "lh", "rf", "rm", "rh"))
                         + len(model_driven.regions["dn"]), 640)
        self.assertTrue(np.allclose(
            np.asarray(model_driven.group.I_base[:])[model_driven.regions["dn"]]
            - np.asarray(model_quiet.group.I_base[:])[model_quiet.regions["dn"]], 3))

    def test_scaffold_defaults_to_only_reward_modulated_motor_edges(self):
        config = isolated_config(seed=42, motor_scaffold=True)
        self.assertEqual(config.vnc_hebb_rate, 0)
        self.assertLessEqual(config.motor_plastic_fraction, .25)

    def test_isolated_loop_uses_body_metrics_and_silencing_stops_motion(self):
        active = isolated_simulation(seed=42, backend="numpy", tonic_drive=2)
        result = active.run_headless(1)
        self.assertGreater(result["mean_abs_torque"], 0)
        self.assertGreater(result["path_length"], 0)
        self.assertIn("neighbor_phase_correlation", result)
        silent = isolated_simulation(seed=42, backend="numpy", tonic_drive=2)
        silent.model.group.spike_gate = 0
        stopped = silent.run_headless(.5)
        self.assertEqual(stopped["path_length"], 0)
        self.assertEqual(stopped["mean_abs_torque"], 0)

    def test_homeostasis_continues_when_reward_plasticity_is_off(self):
        sim = isolated_simulation(seed=42, backend="numpy", plasticity=False)
        kc = sim.model.regions["kc"]
        before = np.asarray(sim.model.group.I_base[:])[kc].copy()
        sim._sample_rates(np.zeros(len(sim.model.group), dtype=np.int64), 5000)
        after = np.asarray(sim.model.group.I_base[:])[kc]
        self.assertGreater(float(after.mean()), float(before.mean()))


if __name__ == "__main__":
    unittest.main()
