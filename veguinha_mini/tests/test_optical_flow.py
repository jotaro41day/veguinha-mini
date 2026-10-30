import unittest
from dataclasses import replace
import numpy as np
from veguinha_mini.config import Config


class OpticalFlowTests(unittest.TestCase):
    def texture(self):
        rng=np.random.default_rng(318)
        # Smooth texture supports subpixel matching without semantic/world data.
        a=rng.uniform(.1,.9,(2,90,160))
        return (a+np.roll(a,1,1)+np.roll(a,1,2))/3

    def test_pixel_translation_has_signed_two_axis_angular_flow(self):
        from veguinha_mini.optical_flow import OpticalFlow
        a=self.texture()
        for dy,dx in ((2,4),(-2,-4),(0,0)):
            f=OpticalFlow();f.update(a,None,80)
            result=f.update(np.roll(a,(dy,dx),axis=(1,2)),.2,80)
            confidence=np.asarray(result['confidence']);v=np.asarray(result['angular_rad_s']);valid=confidence>.1
            self.assertGreater(valid.mean(),.4)
            np.testing.assert_allclose(np.median(v[valid],axis=0),[dx/80/.2,dy/80/.2],atol=.025)
            self.assertEqual(result['units'],'rad/s (image axes)')

    def test_flat_monocular_brightness_and_no_time_are_safe(self):
        from veguinha_mini.optical_flow import OpticalFlow
        a=self.texture();f=OpticalFlow();f.update(a,None,80)
        r=f.update(a*.7+.1,.2,80);valid=np.asarray(r['confidence'])>.1
        self.assertGreater(valid.mean(),.4);self.assertLess(abs(np.asarray(r['angular_rad_s'])[valid]).max(),.015)
        f=OpticalFlow();f.update(np.zeros_like(a),None,80);r=f.update(np.ones_like(a),.2,80)
        self.assertEqual(r['valid_fraction'],0);self.assertFalse(np.asarray(r['angular_rad_s']).any())
        f.update(a,None,80);r=f.update(a,None,80)
        self.assertEqual(r['valid_fraction'],0)

    def test_vision_motion_channels_are_pixel_flow_and_sensor_only(self):
        import mujoco
        from brian2 import start_scope
        from veguinha_mini.perception import ExternalSenses
        from veguinha_mini.mujoco_body import MujocoBody
        from veguinha_mini.world import World
        from veguinha_mini.network import build_network
        from veguinha_mini.sensory import inject_sensors
        start_scope();c=Config(scale=.05,backend='numpy',vision_enabled=True,binocular_enabled=True,optical_flow_enabled=True,vision_width=160,vision_height=90)
        body=MujocoBody(variant='control');sense=ExternalSenses(c);sense.bind_model(body.model)
        world=World(food_enabled=False);sense.scan(body,world)
        body.data.qpos[1]+=.02;mujoco.mj_forward(body.model,body.data);sense.scan(body,world,dt=.1)
        r=sense.snapshot()['optical_flow'];self.assertGreater(r['valid_fraction'],0)
        self.assertGreater(abs(np.asarray(r['angular_rad_s'])).max(),0)
        m=build_network(c,record_spikes=False);sample=body.snapshot(world);sample.update(sense.snapshot());inject_sensors(m,sample,config=c)
        self.assertTrue(np.isfinite(m.group.I_sense[:]).all())
        np.testing.assert_array_equal(np.asarray(m.group.I_sense[:])[m.regions['dan']],0)
        before=np.asarray(m.group.I_sense[:]).copy();sample['optical_flow']['angular_rad_s']=(np.ones((2,3,8,2))*.3).tolist();sample['optical_flow']['confidence']=np.ones((2,3,8)).tolist();inject_sensors(m,sample,config=c)
        after=np.asarray(m.group.I_sense[:]);self.assertTrue(np.any(after[m.regions['cx']]!=before[m.regions['cx']]))
        motors=np.concatenate([np.concatenate(list(p.values())) for p in m.motor_pools.values()]);np.testing.assert_array_equal(after[motors],before[motors])

if __name__=='__main__':unittest.main()
