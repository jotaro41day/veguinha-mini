import unittest

import numpy as np

from veguinha_mini.body import Body
from veguinha_mini.world import World
from veguinha_mini.config import Config
from veguinha_mini.sim import Simulation


class BodyTests(unittest.TestCase):
    def test_large_step_matches_stable_substeps_and_retains_touch(self):
        world = World()
        for wall_collision in (False, True):
            with self.subTest(wall_collision=wall_collision):
                body, reference = Body(world), Body(world)
                torques = np.zeros((6, 2))
                torques[0] = [.8, -.4]
                contacts = np.ones(6, dtype=bool)
                if wall_collision:
                    contacts[:] = False
                    for target in (body, reference):
                        target.position[0] = target.radius + .01
                        target.velocity[0] = -20
                body.step(.05, torques, contacts, world)
                touched = False
                for _ in range(3):
                    reference.step(.05 / 3, torques, contacts, world)
                    touched |= reference.touch
                for attribute in ("position", "velocity", "heading", "angular_velocity",
                                  "angles", "joint_velocity", "energy", "energy_cost_rate"):
                    np.testing.assert_allclose(getattr(body, attribute),
                                               getattr(reference, attribute), atol=1e-12, rtol=0)
                self.assertEqual(body.touch, touched)
                if wall_collision:
                    self.assertTrue(body.touch)
                    self.assertFalse(reference.touch)

    def test_joint_hard_stop_cannot_generate_frictional_propulsion(self):
        world = World()
        for joint, limit in enumerate((1.2, 1.5)):
            for sign in (-1, 1):
                with self.subTest(joint=joint, sign=sign):
                    body = Body(world)
                    body.angles[0, joint] = sign * limit
                    body.joint_velocity[0, joint] = sign
                    torques = np.zeros((6, 2))
                    torques[0, joint] = sign
                    contacts = np.zeros(6, dtype=bool)
                    contacts[0] = True
                    start = body.position.copy()
                    body.step(.02, torques, contacts, world)
                    self.assertEqual(body.angles[0, joint], sign * limit)
                    self.assertEqual(body.joint_velocity[0, joint], 0)
                    np.testing.assert_allclose(body.velocity, 0, atol=1e-12)
                    np.testing.assert_allclose(body.position, start, atol=1e-12)

    def test_same_local_joint_angles_mirror_left_and_right_legs(self):
        body = Body(World())
        body.angles[0] = [.3, -.2]
        body.angles[3] = [.3, -.2]
        hips, knees, feet, _, hip_jac, knee_jac = body.geometry()
        for points in (hips, knees, feet, hip_jac, knee_jac):
            np.testing.assert_allclose(points[0, 0], points[3, 0], atol=1e-10)
            if points is hips or points is knees or points is feet:
                np.testing.assert_allclose(points[0, 1] + points[3, 1],
                                           2 * body.position[1], atol=1e-10)
            else:
                np.testing.assert_allclose(points[0, 1], -points[3, 1], atol=1e-10)

    def test_rest_stays_still_and_manual_torque_produces_frictional_motion(self):
        world = World()
        body = Body(world)
        start = body.position.copy()
        torques = np.zeros((6, 2))
        contacts = np.ones(6, dtype=bool)
        for _ in range(100):
            body.step(0.01, torques, contacts, world)
        np.testing.assert_allclose(body.position, start, atol=1e-8)
        torques[0, 0] = 1
        for _ in range(100):
            body.step(0.01, torques, contacts, world)
        self.assertGreater(np.linalg.norm(body.position - start), 0.1)
        self.assertLess(body.energy, 1)
        snapshot = body.snapshot(world, contacts)
        self.assertEqual(len(snapshot["feet"]), 6)
        self.assertEqual(len(snapshot["antennae"]), 2)
        self.assertEqual(len(snapshot["angles"]), 6)

    def test_airborne_leg_exerts_no_ground_force_and_walls_constrain_body(self):
        world = World()
        body = Body(world)
        start = body.position.copy()
        torques = np.zeros((6, 2))
        torques[0, 0] = 1
        contacts = np.zeros(6, dtype=bool)
        for _ in range(100):
            body.step(0.01, torques, contacts, world)
        np.testing.assert_allclose(body.position, start, atol=1e-8)
        self.assertGreater(body.snapshot(world, contacts)["angles"][0][0], 0)
        body.position[:] = [world.width - 1, world.height + 1]
        body.step(0.01, np.zeros((6, 2)), contacts, world)
        self.assertLessEqual(body.position[0], world.width - body.radius)
        self.assertLessEqual(body.position[1], world.height - body.radius)
        self.assertTrue(body.touch)

    def test_ground_load_reacts_on_the_joint(self):
        world = World()
        loaded, airborne = Body(world), Body(world)
        torques = np.zeros((6, 2))
        torques[0, 0] = 0.8
        ground = np.zeros(6, dtype=bool)
        ground[0] = True
        air = np.zeros(6, dtype=bool)
        for _ in range(100):
            loaded.step(.01, torques, ground, world)
            airborne.step(.01, torques, air, world)
        self.assertGreater(abs(loaded.angles[0, 0] - airborne.angles[0, 0]), .001)

    def test_manual_commands_change_body_without_motor_network_output(self):
        sim = Simulation(Config(scale=0.05, backend="numpy", manual_test_mode=True))
        sim.submit({"cmd": "motor", "leg": "lf", "joint": 0, "value": 0.8})
        sim.submit({"cmd": "contact", "leg": "lf", "value": False})
        sim.submit({"cmd": "jaw", "value": 0.7})
        sim._commands()
        frame = sim.latest()
        self.assertEqual(float(sim.manual_torques[0, 0]), 0.8)
        self.assertFalse(sim.contacts[0])
        self.assertEqual(sim.body.jaw_open, 0.7)
        self.assertEqual(len(frame["body"]["feet"]), 6)


if __name__ == "__main__":
    unittest.main()
