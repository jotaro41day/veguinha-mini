import math
import unittest

import mujoco
import numpy as np

from veguinha_mini.mujoco_body import MujocoBody, JOINT_NAMES


class MujocoPhysicsTests(unittest.TestCase):
    def test_head_on_floor_counts_as_fall_before_upright_threshold(self):
        body = MujocoBody()
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, np.array([0., 1., 0.]),
                                 math.radians(35))
        body.data.qpos[3:7] = quat
        body.data.qpos[2] = .16
        mujoco.mj_forward(body.model, body.data)
        body._update_contacts()
        self.assertGreater(body.upright, .5)
        self.assertTrue(body.fell)

    def test_all_motor_channels_map_to_matching_joints(self):
        body = MujocoBody()
        self.assertEqual(body.model.nu, 12)
        for index, joint in enumerate(JOINT_NAMES):
            actuator = body.model.actuator(f"{joint}_motor")
            self.assertEqual(int(body.model.actuator_trnid[actuator.id, 0]),
                             body.model.joint(joint).id)
        body.step(.02, np.zeros((6, 2)))
        self.assertEqual(body.data.qpos.size, 19)

    def test_passive_settling_and_free_body_fall(self):
        body = MujocoBody()
        zero = np.zeros((6, 2))
        for _ in range(250):
            body.step(.02, zero)
        self.assertFalse(body.fell)
        self.assertEqual(sum(body.contacts), 6)
        self.assertLess(np.linalg.norm(body.data.qvel), .005)
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, np.array([1., 0., 0.]),
                                 math.radians(35))
        body.data.qpos[3:7] = quat
        body.data.qpos[2] = .23
        body.data.qvel[:] = 0
        mujoco.mj_forward(body.model, body.data)
        for _ in range(150):
            body.step(.02, zero)
        self.assertTrue(body.fell)
        self.assertLess(body.upright, .5)


if __name__ == "__main__":
    unittest.main()
