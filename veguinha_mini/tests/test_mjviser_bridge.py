import unittest
from unittest.mock import patch

import numpy as np

from veguinha_mini.config import Config
from veguinha_mini.mujoco_body import MujocoBody
from veguinha_mini.sim import Simulation
from veguinha_mini.world import World


class ViewerIntegrationTests(unittest.TestCase):
    def test_food_and_odor_remain_fixed_when_animal_moves(self):
        import mujoco
        from veguinha_mini.mjviser_bridge import MjviserBridge
        world=World();body=MujocoBody(world)
        class Observation:
            pass
        observation=Observation();observation.body=body
        bridge=MjviserBridge(observation,port=0)
        try:
            self.assertFalse(bridge.scene.camera_tracking_enabled)
            def frame():
                return dict(body=body.snapshot(world),world=world.snapshot(),epoch=0,seq=1,speed=1)
            bridge.render(frame())
            food=np.asarray(bridge.food.position).copy()
            odor=bridge.odor.points.copy();colors=bridge.odor.colors.copy()
            first_odor=np.array(body.snapshot(world)['antenna_odor'])
            first_distance=np.linalg.norm(food[:2]-body.position)
            body.data.qpos[0]+=.6;mujoco.mj_forward(body.model,body.data)
            bridge.render(frame())
            np.testing.assert_array_equal(bridge.food.position,food)
            np.testing.assert_array_equal(bridge.odor.points,odor)
            np.testing.assert_array_equal(bridge.odor.colors,colors)
            np.testing.assert_allclose(bridge.scene._scene_offset,0,atol=1e-12)
            np.testing.assert_allclose(bridge.scene.fixed_bodies_frame.position,0,atol=1e-12)
            self.assertLess(np.linalg.norm(food[:2]-body.position),first_distance)
            self.assertTrue((np.array(body.snapshot(world)['antenna_odor'])>first_odor).all())
            self.assertEqual(world.snapshot()['odor'],[450.,190.])
        finally:bridge.stop()

    def test_food_command_changes_sensory_world(self):
        sim = Simulation(Config(scale=.05, backend="numpy", manual_test_mode=True))
        sim.submit({"cmd": "food", "value": [90, 80]})
        sim._commands()
        self.assertEqual(sim.world.snapshot()["odor"], [90.0, 80.0])
        self.assertGreater(float(sim.world.odor_at([90, 80])), .99)
        sim.submit({"cmd": "food", "value": None})
        sim._commands()
        self.assertIsNone(sim.world.snapshot()["odor"])
        self.assertEqual(float(sim.world.odor_at([90, 80])), 0)
        with self.assertRaises(ValueError):
            sim.submit({"cmd": "food", "value": [float("nan"), 2]})

    def test_bridge_copies_physical_state_without_mujoco_step(self):
        try:
            from veguinha_mini.mjviser_bridge import MjviserBridge
        except ImportError as exc:
            self.skipTest(str(exc))
        world = World()
        body = MujocoBody(world)
        body.step(.02, np.zeros((6, 2)))
        frame = {"body": body.snapshot(world),
                 "world": world.snapshot(), "epoch": 0, "seq": 1, "speed": 1}

        class FakeSimulation:
            def latest(self):
                return frame

        bridge = MjviserBridge(FakeSimulation(), port=0)
        try:
            with patch("mujoco.mj_step", side_effect=AssertionError("mj_step usado")):
                bridge.render(frame)
            self.assertEqual(bridge.model.nmocap, 0)
            np.testing.assert_allclose(bridge.data.qpos, body.data.qpos, atol=1e-8)
            np.testing.assert_allclose(bridge.data.site_xpos[bridge.toe_ids],
                                       body.data.site_xpos[body.toe_ids], atol=1e-7)
            self.assertTrue(bridge.food.visible)
            frame["world"] = World(food_enabled=False).snapshot()
            bridge.render(frame)
            self.assertFalse(bridge.food.visible)
            self.assertFalse(bridge.odor.visible)
        finally:
            bridge.stop()


if __name__ == "__main__":
    unittest.main()
