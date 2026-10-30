"""A razão oficial exclui teleportes e agrega episódios antes de dividir."""

import unittest

import numpy as np

from veguinha_mini.body import Body
from veguinha_mini.metrics import Metrics
from veguinha_mini.world import World


class OfficialMetricTests(unittest.TestCase):
    def test_net_path_aggregates_episode_endpoints_without_teleport(self):
        world = World()
        body = Body(world)
        measured = Metrics(body)
        torque = np.zeros((6, 2))
        body.position[0] += 10
        measured.update(body, torque)
        body.position[0] -= 10
        measured.update(body, torque)
        restarted = Body(world)
        restarted.position[0] += 100
        measured.episode_reset(body, restarted)
        restarted.position[0] += 5
        measured.update(restarted, torque)
        result = measured.snapshot(restarted)
        self.assertEqual(result["episodes"], 2)
        self.assertEqual(result["path_length"], 25)
        self.assertEqual(result["displacement"], 5)
        self.assertEqual(result["net_path"], .2)


if __name__ == "__main__":
    unittest.main()
