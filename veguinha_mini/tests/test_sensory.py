import unittest

import numpy as np

from veguinha_mini.config import Config
from veguinha_mini.network import build_network
from veguinha_mini.sensory import inject_sensors


class SensoryTests(unittest.TestCase):
    def test_sensory_currents_target_the_documented_populations(self):
        model = build_network(Config(scale=0.1, backend="numpy"))
        sample = {"angles": [[0.5, -0.3]] + [[0, 0]] * 5,
                  "contacts": [True] * 6, "touch": True,
                  "energy": 0.4, "antenna_odor": [0.8, 0.2]}
        inject_sensors(model, sample, odor_type=1)
        sensory = np.asarray(model.group.I_sense[:])
        odor = np.asarray(model.group.I_odor[:])
        self.assertGreater(sensory[model.motor_pools["lf"]["hip_extend"]].sum(), 0)
        self.assertGreater(sensory[model.motor_pools["lf"]["knee_flex"]].sum(), 0)
        self.assertGreater(sensory[model.interneurons["lf"]["inhibitory"]].sum(), 0)
        self.assertGreater(sensory[model.regions["seg"]].sum(), 0)
        self.assertGreater(sensory[model.regions["value"]].sum(), 0)
        self.assertGreater(odor[model.regions["al_glom"]].sum(), 0)
        self.assertEqual(odor[model.regions["kc"]].sum(), 0)


if __name__ == "__main__":
    unittest.main()
