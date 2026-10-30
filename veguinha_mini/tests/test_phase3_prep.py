import unittest

import numpy as np

from veguinha_mini.config import Config
from veguinha_mini.network import build_network, set_odor_pulse


class PrepTests(unittest.TestCase):
    def test_four_odors_use_separate_glomeruli_and_distinct_claws(self):
        model = build_network(Config(scale=1, backend="numpy"))
        glom = model.regions["al_glom"]
        groups = []
        for odor in range(4):
            set_odor_pulse(model, 12, odor)
            groups.append(set(glom[np.asarray(model.group.I_odor[glom]) > 0]))
        self.assertTrue(all(len(group) == len(glom) // 4 for group in groups))
        self.assertEqual(len(set.union(*groups)), len(glom))
        self.assertTrue(all(not groups[a] & groups[b]
                            for a in range(4) for b in range(a + 1, 4)))
        pre = np.asarray(model.synapses.i[:])
        post = np.asarray(model.synapses.j[:])
        mask = np.isin(pre, glom) & np.isin(post, model.regions["kc"])
        for kc in model.regions["kc"]:
            self.assertEqual(len(set(pre[mask & (post == kc)])), 7)

    def test_vnc_motor_pools_are_excitatory_and_reciprocally_inhibited(self):
        model = build_network(Config(scale=1, backend="numpy"))
        pre = np.asarray(model.synapses.i[:])
        post = np.asarray(model.synapses.j[:])
        for leg in ("lf", "lm", "lh", "rf", "rm", "rh"):
            region = model.regions[f"vnc_{leg}"]
            self.assertTrue(0.65 <= (~model.inhibitory[region]).mean() <= 0.70)
            pools = model.motor_pools[leg]
            self.assertEqual(set(pools), {"hip_flex", "hip_extend", "knee_flex", "knee_extend"})
            for pool in pools.values():
                self.assertTrue(np.all(~model.inhibitory[pool]))
            for a, b in (("hip_flex", "hip_extend"), ("knee_flex", "knee_extend")):
                for target in (a, b):
                    inhibitory_inputs = model.inhibitory[pre] & np.isin(post, pools[target])
                    self.assertTrue(inhibitory_inputs.any())


if __name__ == "__main__":
    unittest.main()
