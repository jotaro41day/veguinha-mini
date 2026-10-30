import unittest

import numpy as np

from veguinha_mini.config import Config
from veguinha_mini.network import build_network, run_diagnostic
from veguinha_mini.regions import Pathway


class PhaseOneTests(unittest.TestCase):
    def test_anatomy_and_dale_law(self):
        model = build_network(Config(scale=0.1, seed=7))
        self.assertEqual(len(model.group), sum(len(v) for v in model.regions.values()))
        self.assertEqual(set(model.cell_types), {"RS", "IB", "CH", "FS", "LTS"})
        self.assertTrue(0.13 <= model.inhibitory.mean() <= 0.17)
        self.assertGreater(len(model.synapses), 0)
        pre = np.asarray(model.synapses.i[:], dtype=int)
        weights = np.asarray(model.synapses.w[:])
        self.assertTrue(np.all(weights[~model.inhibitory[pre]] > 0))
        self.assertTrue(np.all(weights[model.inhibitory[pre]] < 0))
        delays = np.asarray(model.synapses.delay[:] / model.ms)
        self.assertTrue(np.all((delays >= 1) & (delays <= 5)))
        self.assertIn("kc", model.regions)
        self.assertIn("cx", model.regions)
        self.assertEqual(sum(name.startswith("vnc_") for name in model.regions), 6)
        model.network.run(0 * model.ms)

    def test_isolated_activity_is_reported_without_body(self):
        result = run_diagnostic(Config(scale=0.1, seed=7), duration_ms=120)
        self.assertEqual(set(result.rates_hz), set(result.counts))
        self.assertTrue(all(np.isfinite(rate) for rate in result.rates_hz.values()))
        self.assertGreater(sum(result.counts.values()), 0)
        self.assertLess(max(result.rates_hz.values()), 100)

    def test_kenyon_code_is_sparse_at_default_scale(self):
        result = run_diagnostic(Config(seed=7), duration_ms=300)
        self.assertLess(result.kc_active_fraction_50ms, 0.15)
        self.assertLess(result.rates_hz["kc"], 0.1)
        self.assertTrue(all(np.isfinite(rate) for rate in result.late_rates_hz.values()))

    def test_config_controls_connections_and_delays(self):
        config = Config(scale=0.1, pathways=(Pathway("al_glom", "al_proj", 1, 0.2),),
                        delay_min_ms=2, delay_max_ms=3)
        model = build_network(config)
        self.assertEqual(len(model.synapses), len(model.regions["al_proj"]))
        delays = np.asarray(model.synapses.delay[:] / model.ms)
        self.assertTrue(np.all((delays >= 2) & (delays <= 3)))
        model.network.run(0 * model.ms)


if __name__ == "__main__":
    unittest.main()
