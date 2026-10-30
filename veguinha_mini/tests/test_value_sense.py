import inspect
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from veguinha_mini import sensory
from veguinha_mini.body import Body
from veguinha_mini.isolated import isolated_config
from veguinha_mini.isolated import isolated_simulation
from veguinha_mini.network import build_network
from veguinha_mini.metrics import Metrics
from veguinha_mini.training import checkpoint_metrics
from veguinha_mini.world import World


class ValueSenseTests(unittest.TestCase):
    def test_flow_memory_smooths_a_brief_forward_pulse(self):
        self.assertTrue(hasattr(sensory, "ValueFlowFilter"))
        value_filter = sensory.ValueFlowFilter(200)
        first = value_filter.update([2, 0, 0], .02)
        second = value_filter.update([0, 0, 0], .02)
        self.assertAlmostEqual(first[0], 2 * (1 - math.exp(-.1)), places=5)
        self.assertGreater(second[0], 0)
        self.assertLess(second[0], first[0])

    def test_value_and_dan_receive_filtered_flow_current(self):
        self.assertIn("value_flow", inspect.signature(sensory.inject_sensors).parameters)
        config = isolated_config(seed=42, backend="numpy")
        model = build_network(config, record_spikes=False)
        sample = Body(World()).snapshot(World(), np.ones(6, dtype=bool))
        sample["flow"] = [2.0, 0.0, 0.0]
        sensory.inject_sensors(model, sample, config=config)
        raw = np.asarray(model.group.I_sense[:]).copy()
        sensory.inject_sensors(model, sample, config=config, value_flow=[.2, 0, 0])
        filtered = np.asarray(model.group.I_sense[:]).copy()
        for region in ("value", "dan"):
            self.assertLess(filtered[model.regions[region]].mean(),
                            raw[model.regions[region]].mean())
        np.testing.assert_allclose(filtered[model.regions["flow"]],
                                   raw[model.regions["flow"]])

    def test_checkpoint_preserves_recent_sensory_flow(self):
        sim = isolated_simulation(seed=42, backend="numpy", motor_scaffold=True)
        self.assertTrue(hasattr(sim, "value_filter"))
        sim.value_filter.update([2, 1, .2], .02)
        expected = sim.value_filter.state.copy()
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / "checkpoint"
            sim.save_checkpoint(stem)
            resumed = isolated_simulation(seed=42, backend="numpy", motor_scaffold=True)
            resumed.load_checkpoint(stem)
            np.testing.assert_allclose(resumed.value_filter.state, expected)

    def test_checkpoint_preserves_training_window(self):
        sim = isolated_simulation(seed=42, backend="numpy", motor_scaffold=True)
        sim.window_metrics = Metrics(sim.body)
        sim.run_headless(.1)
        expected = sim.window_metrics.snapshot(sim.body)
        self.assertGreater(expected["path_length"], 0)
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / "checkpoint"
            sim.save_checkpoint(stem)
            resumed = isolated_simulation(seed=42, backend="numpy", motor_scaffold=True)
            resumed.window_metrics = Metrics(resumed.body)
            resumed.load_checkpoint(stem)
            self.assertEqual(resumed.window_metrics.snapshot(resumed.body), expected)
            row = checkpoint_metrics(resumed, 1.0, "learning")
            self.assertEqual(row["window_start_s"], 0)
            self.assertEqual(row["window_net_path"], expected["net_path"])

    def test_training_window_restarts_at_900_seconds_without_resetting_body(self):
        sim = isolated_simulation(seed=42, backend="numpy", motor_scaffold=True)
        sim.window_metrics = Metrics(sim.body)
        sim.window_metrics.path_length = 5
        sim.simulated_ms = 899980
        body = sim.body
        sim._tick()
        self.assertIs(sim.body, body)
        self.assertEqual(sim.simulated_ms, 900000)
        self.assertEqual(sim.window_metrics.path_length, 0)
        np.testing.assert_allclose(sim.window_metrics.origin, body.position)


if __name__ == "__main__":
    unittest.main()
