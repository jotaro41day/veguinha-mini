"""Short mechanical and signal-path gates; no gait or learning is imposed."""
import unittest
import numpy as np
import mujoco
from pathlib import Path
import xml.etree.ElementTree as ET
from veguinha_mini.mujoco_body import MujocoBody
from veguinha_mini.config import Config
from veguinha_mini.network import build_network
from veguinha_mini.motor import MotorDecoder
from veguinha_mini.sensory import inject_sensors


class Mk1Tests(unittest.TestCase):
    def test_femur_motor_has_effort_to_unload_foot(self):
        b = MujocoBody(variant='mk1')
        np.testing.assert_allclose(b.model.actuator_gear[1::3, 0], .2)

    def test_upgrade_only_changes_femur_transmission(self):
        assets = Path(__file__).resolve().parents[1]/'assets'
        old = ET.parse(assets/'hexapod_mk1_v1.xml').getroot()
        new = ET.parse(assets/'hexapod_mk1.xml').getroot()
        for motor in old.find('actuator'):
            if motor.get('joint').endswith('_femur'):
                motor.set('gear', '0.2')
        self.assertEqual(ET.tostring(old), ET.tostring(new))

    def test_mass_dofs_and_passive_support(self):
        b = MujocoBody(variant="mk1")
        self.assertEqual(b.model.nu, 18)
        self.assertAlmostEqual(b.model.body_mass.sum(), .316, places=12)
        visual = b.model.geom_group == 1
        self.assertEqual(int(visual.sum()), 23)
        self.assertTrue(np.all(b.model.geom_contype[visual] == 0))
        self.assertTrue(np.all(b.model.geom_conaffinity[visual] == 0))
        for _ in range(150):
            b.step(.02, np.zeros((6, 3)))
        self.assertTrue(np.all(b.contacts))
        self.assertGreater(b.upright, .999)
        np.testing.assert_allclose(b.normal_load, .316 * 9.81 / 6, atol=.003)

    def test_joint_positive_direction_all_sides(self):
        b = MujocoBody(variant="mk1")
        for leg_i, leg in enumerate(("lf", "lm", "lh", "rf", "rm", "rh")):
            for joint_i, joint in enumerate(b.joint_names):
                mujoco.mj_resetData(b.model, b.data)
                mujoco.mj_forward(b.model, b.data)
                before = b.data.site_xpos[b.toe_ids[leg_i]].copy()
                origin = b.data.xpos[b.model.body(f"{leg}_{joint}").id].copy()
                qpos = b.model.joint(f"{leg}_{joint}").qposadr
                b.data.qpos[qpos] = .1
                mujoco.mj_forward(b.model, b.data)
                after = b.data.site_xpos[b.toe_ids[leg_i]]
                if joint_i == 0:
                    self.assertGreater(after[0] - before[0], 0, leg)
                else:
                    radial = before[:2] - origin[:2]
                    self.assertGreater(np.dot(after[:2] - before[:2], radial), 0, leg)

    def test_three_joint_pools_sensors_decoder(self):
        c = Config(body_model="mk1", backend="numpy", scale=1)
        m = build_network(c, record_spikes=False)
        d = MotorDecoder(m, c)
        for leg, pools in m.motor_pools.items():
            self.assertEqual(len(m.regions[f"vnc_{leg}"]), 144)
            self.assertEqual(len(m.interneurons[leg]["excitatory"]), 50)
            self.assertEqual(len(m.interneurons[leg]["inhibitory"]), 46)
            self.assertEqual(len(pools), 6)
            self.assertTrue(all(len(p) == 8 for p in pools.values()))
            self.assertFalse(m.inhibitory[np.concatenate(list(pools.values()))].any())
        count = np.zeros(len(m.group), dtype=int)
        count[m.motor_pools["lf"]["coxa_flex"]] = 1
        torque = d.update(count, .02)
        self.assertEqual(torque.shape, (6, 3))
        self.assertGreater(torque[0, 0], 0)
        self.assertEqual(np.count_nonzero(torque), 1)
        b = MujocoBody(variant="mk1")
        sample = b.snapshot()
        sample["angles"][0][2] = .5
        inject_sensors(m, sample, config=c, efference=torque)
        sensed = np.asarray(m.group.I_sense[:])
        self.assertTrue(np.all(sensed[m.motor_pools["lf"]["tibia_extend"]] > 0))
        self.assertTrue(np.all(sensed[m.motor_pools["rf"]["tibia_extend"]] == 0))


if __name__ == "__main__":
    unittest.main()
