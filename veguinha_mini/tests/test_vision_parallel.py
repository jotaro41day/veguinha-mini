import pickle,unittest
from dataclasses import replace
import numpy as np,mujoco
from veguinha_mini.config import Config
from veguinha_mini.mujoco_body import MujocoBody
from veguinha_mini.perception import ExternalSenses
from veguinha_mini.world import World

class ParallelVisionTests(unittest.TestCase):
    def check_workers(self,workers):
        c=Config(vision_enabled=True,binocular_enabled=True,optical_flow_enabled=True,foveated_enabled=True,vision_width=320,vision_height=180,vision_interval_ms=480,vision_periphery_ms=120,vision_workers=1)
        body=MujocoBody(variant='mk1',prey_count=2);world=World(shade_enabled=True)
        serial=ExternalSenses(c);parallel=ExternalSenses(replace(c,vision_workers=workers))
        pose=body.data.qpos.copy();velocity=body.data.qvel.copy()
        for s in (serial,parallel):s.bind_model(body.model)
        for dt in (None,.12,.12,.12,.12):
            for s in (serial,parallel):s.scan(body,world,dt)
            np.testing.assert_array_equal(serial.rgb,parallel.rgb)
            np.testing.assert_array_equal(serial.retina,parallel.retina)
            np.testing.assert_array_equal(serial.optical.previous,parallel.optical.previous)
            self.assertEqual(serial.optical.result,parallel.optical.result)
            self.assertEqual(serial.last_ray_count,parallel.last_ray_count)
            self.assertEqual(serial.detail_sequence,parallel.detail_sequence)
        np.testing.assert_array_equal(body.data.qpos,pose);np.testing.assert_array_equal(body.data.qvel,velocity)
        resumed=pickle.loads(pickle.dumps(parallel));resumed.scan(body,world,.48)
        serial.scan(body,world,.48);np.testing.assert_array_equal(serial.rgb,resumed.rgb)

    def test_two_workers_preserve_pixels_clocks_flow_and_body_state(self):self.check_workers(2)

    def test_four_workers_preserve_pixels_clocks_flow_and_body_state(self):self.check_workers(4)

    def test_worker_count_is_bounded(self):
        for n in (0,3,9,1.5):
            with self.assertRaises(ValueError):Config(vision_workers=n)

if __name__=='__main__':unittest.main()
