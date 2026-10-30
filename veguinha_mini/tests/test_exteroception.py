import unittest
from dataclasses import replace
import numpy as np
import mujoco
from veguinha_mini.config import Config
from veguinha_mini.world import World
from veguinha_mini.mujoco_body import MujocoBody


class ExternalSenseTests(unittest.TestCase):
    def test_retina_changes_with_body_but_food_stays_in_world(self):
        from veguinha_mini.perception import ExternalSenses
        body=MujocoBody(variant='mk1');sense=ExternalSenses(Config(vision_enabled=True))
        sense.bind_model(body.model)
        rotation=body.data.xmat[body.body_id].reshape(3,3)
        origin=body.data.xpos[body.body_id]+rotation@sense.eye_origins[0]
        direction=rotation@sense.directions[0,-8,7]
        distance=(.065-origin[2])/direction[2]
        point=origin+distance*direction
        world=World(odor_x=(point[0]+3)*100,odor_y=(2-point[1])*100)
        sense.scan(body,world);first=sense.retina.copy()
        self.assertGreater(first[:,:,1].max(),0)
        body.data.qpos[1]+=.6;mujoco.mj_forward(body.model,body.data);sense.scan(body,world)
        self.assertFalse(np.array_equal(first,sense.retina))
        np.testing.assert_allclose(world.food_position_m,point,atol=1e-8)

    def test_rgb_cameras_have_motion_and_a_separate_sample_clock(self):
        import base64
        from veguinha_mini.perception import ExternalSenses
        c=Config(vision_enabled=True);body=MujocoBody(variant='mk1');sense=ExternalSenses(c);sense.bind_model(body.model)
        sense.scan(body,World(food_enabled=False));first=sense.snapshot()['camera']
        self.assertEqual((first['width'],first['height']),(48,24))
        self.assertEqual(len(base64.b64decode(first['eyes'][0])),48*24*3)
        sense.scan(body,World(food_enabled=False),dt=.02)
        self.assertEqual(sense.snapshot()['camera']['seq'],first['seq'])
        body.data.qpos[0]+=.08;mujoco.mj_forward(body.model,body.data)
        sense.scan(body,World(food_enabled=False))
        self.assertNotEqual(sense.snapshot()['camera']['eyes'],first['eyes'])
        self.assertGreater(abs(sense.retina[:,:,2:]).max(),0)

    def test_impact_threshold_and_decay(self):
        from veguinha_mini.perception import ExternalSenses
        sense=ExternalSenses(Config(nociception_enabled=True))
        sense.transduce_force(np.array([.5,.5]),.002)
        np.testing.assert_array_equal(sense.pain,0)
        sense.transduce_force(np.array([10.,0]),.002)
        self.assertGreater(sense.pain[0],0);self.assertEqual(sense.pain[1],0)
        previous=sense.pain.copy();sense.transduce_force(np.zeros(2),.3)
        np.testing.assert_allclose(sense.pain,previous*np.exp(-1))

    def test_observing_senses_does_not_change_physics(self):
        from veguinha_mini.perception import ExternalSenses
        a=MujocoBody(variant='mk1');b=MujocoBody(variant='mk1')
        b.senses=ExternalSenses(Config(vision_enabled=True,nociception_enabled=True));b.senses.bind_model(b.model)
        for _ in range(15):
            a.step(.02,np.zeros((6,3)),world=World());b.step(.02,np.zeros((6,3)),world=World())
        np.testing.assert_array_equal(a.data.qpos,b.data.qpos)
        np.testing.assert_array_equal(a.data.qvel,b.data.qvel)
        np.testing.assert_array_equal(b.senses.pain,0)

    def test_real_wall_collision_generates_nociception(self):
        from veguinha_mini.perception import ExternalSenses
        body=MujocoBody(variant='mk1');body.senses=ExternalSenses(Config(nociception_enabled=True))
        body.data.qpos[0]=2.50;body.data.qvel[0]=3.;mujoco.mj_forward(body.model,body.data)
        peak=0.;pain=0.
        for _ in range(30):
            body.step(.02,np.zeros((6,3)),world=World())
            peak=max(peak,float(body.senses.wall_peak_n.sum()))
            pain=max(pain,float(body.senses.pain.sum()))
        self.assertGreater(peak,2.)
        self.assertGreater(pain,0.)

    def test_sensory_currents_reach_cx_value_and_seg_without_dan_injection(self):
        from brian2 import start_scope
        from veguinha_mini.network import build_network
        from veguinha_mini.sensory import inject_sensors
        start_scope()
        config=Config(scale=.1,backend='numpy',phase5_enabled=True,neural_value_channels=True,
                      vision_enabled=True,nociception_enabled=True,feeding_enabled=True)
        model=build_network(config,record_spikes=False)
        sample=dict(angles=np.zeros((6,2)),contacts=[False]*6,touch=False,energy=.5,
                    flow=[0,0,0],energy_cost_rate=.01,antenna_odor=[0,0],retina=np.zeros((2,8,2)),pain=[0,0],mouth_contact=0)
        inject_sensors(model,sample,config=config);before=np.array(model.group.I_sense[:])
        sample['retina'][0,0,0]=1;sample['pain']=[1,0];sample['mouth_contact']=1
        inject_sensors(model,sample,config=config);after=np.array(model.group.I_sense[:])
        for name in ('cx','value','seg'):
            self.assertTrue((after[model.regions[name]]>before[model.regions[name]]).any())
        np.testing.assert_array_equal(after[model.regions['dan']],0)


if __name__=='__main__':unittest.main()
