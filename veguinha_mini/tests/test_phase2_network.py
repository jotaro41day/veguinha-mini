import unittest

import numpy as np
from brian2 import ms

from veguinha_mini.config import Config
from veguinha_mini.network import build_network, set_odor_pulse


class DenseNetworkTests(unittest.TestCase):
    def test_odor_pulse_reaches_projection_and_kenyon(self):
        model = build_network(Config(scale=1, seed=42, backend="numpy"))
        model.network.run(1000 * ms)
        baseline = np.asarray(model.monitor.count[:], dtype=np.int64)
        set_odor_pulse(model, 12)
        model.network.run(1000 * ms)
        response = np.asarray(model.monitor.count[:], dtype=np.int64) - baseline
        for name, minimum in (("al_proj", 1.10), ("kc", 1.05)):
            indices = model.regions[name]
            self.assertGreater(response[indices].sum() / baseline[indices].sum(), minimum)

    def test_target_indegrees_and_kenyon_claws(self):
        model = build_network(Config(scale=1, seed=11, backend="numpy"))
        pre = np.asarray(model.synapses.i[:], dtype=int)
        post = np.asarray(model.synapses.j[:], dtype=int)
        degree = np.bincount(post, minlength=len(model.group))
        for name, indices in model.regions.items():
            if name != "kc":
                self.assertTrue(np.all(degree[indices] >= 150), name)
        self.assertTrue(np.all(degree[model.regions["kc"]] >= 7))
        self.assertTrue(np.all(degree[model.regions["cx"]] >= 150))
        self.assertTrue(np.all(degree[model.regions["vnc_lf"]] >= 150))
        kc_from_glom = np.isin(pre, model.regions["al_glom"]) & np.isin(post, model.regions["kc"])
        claws = np.bincount(post[kc_from_glom], minlength=len(model.group))[model.regions["kc"]]
        self.assertTrue(np.all(claws == 7))
        self.assertGreater(np.mean(degree[model.regions["mbon"]]), 400)
        self.assertGreater(np.mean(degree[model.regions["apl"]]), 400)
        model.network.run(0 * model.ms)


if __name__ == "__main__":
    unittest.main()
