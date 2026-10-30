"""The optional skin must not change the hexapod's dynamics."""

import json
import unittest
from pathlib import Path

import numpy as np

from veguinha_mini.body import LEG_NAMES
from veguinha_mini.mujoco_body import MujocoBody


FIXTURES = Path(__file__).with_name("fixtures")


class VisualSkinTests(unittest.TestCase):
    def test_each_leg_has_tapered_visuals_on_both_joint_links(self):
        model = MujocoBody().model
        for leg in LEG_NAMES:
            for link in ("femur", "tibia"):
                body_id = model.body(f"{leg}_{link}").id
                visuals = [geom for geom in range(model.ngeom)
                           if model.geom_bodyid[geom] == body_id
                           and model.geom_group[geom] == 1]
                self.assertGreaterEqual(len(visuals), 2, f"{leg}_{link}")
                radii = [model.geom_size[geom, 0] for geom in visuals]
                self.assertGreater(max(radii), min(radii), f"{leg}_{link}")

    def test_skin_has_no_collision_or_inertia_and_physics_matches_reference(self):
        before = json.loads((FIXTURES / "hexapod_physics_before_visual.json").read_text())
        model = MujocoBody().model
        self.assertGreater(model.ngeom, before["ngeom"])
        self.assertEqual(model.nbody, before["nbody"])
        self.assertEqual(model.nv, before["nv"])
        visual = np.flatnonzero(model.geom_group == 1)
        self.assertGreater(len(visual), 0)
        np.testing.assert_array_equal(model.geom_contype[visual], 0)
        np.testing.assert_array_equal(model.geom_conaffinity[visual], 0)
        physical = model.geom_group == 0
        for field, expected in before.items():
            if field.startswith("geom_"):
                actual = getattr(model, field)[physical]
            elif field in ("ngeom", "nbody", "nv"):
                continue
            else:
                actual = getattr(model, field)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12,
                                       err_msg=field)

    def test_same_torques_give_same_trajectory_and_contacts(self):
        expected = np.load(FIXTURES / "hexapod_trajectory_before_visual.npz")["state"]
        body = MujocoBody(dt=.002)
        observed = []
        for step in range(len(expected)):
            torque = .5 * np.sin(2 * np.pi * (step * .02 / .38)
                                 + np.arange(12) * .4).reshape(6, 2)
            body.step(.02, torque)
            observed.append(np.r_[body.data.qpos.copy(), body.data.qvel.copy(),
                                  body.contacts.astype(float), body.normal_load.copy()])
        np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
