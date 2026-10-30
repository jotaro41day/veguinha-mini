import unittest
from dataclasses import replace
from types import SimpleNamespace
import numpy as np,mujoco
from brian2 import start_scope
from veguinha_mini.config import Config
from veguinha_mini.somatic import looming_from_flow,taste_sample
from veguinha_mini.mujoco_body import MujocoBody
from veguinha_mini.world import World
from veguinha_mini.perception import ExternalSenses,mouth_position

class SomaticTests(unittest.TestCase):
 def test_looming_uses_expansion_not_translation_or_missing_texture(self):
  y,x=np.meshgrid((np.arange(3)+.5)/3-.5,(np.arange(8)+.5)/8-.5,indexing='ij');radial=np.stack([x,y],axis=-1)
  flow=dict(angular_rad_s=np.stack([radial,radial]).tolist(),confidence=np.ones((2,3,8)).tolist())
  self.assertGreater(looming_from_flow(flow),.1)
  flow['angular_rad_s']=np.ones((2,3,8,2)).tolist();self.assertLess(looming_from_flow(flow),1e-10)
  flow['angular_rad_s']=np.stack([-radial,-radial]).tolist();self.assertLess(looming_from_flow(flow),1e-10)
  flow['confidence']=np.zeros((2,3,8)).tolist();self.assertEqual(looming_from_flow(flow),0)

 def test_sparse_translation_is_not_an_approaching_obstacle(self):
  velocity=np.zeros((2,3,8,2));confidence=np.zeros((2,3,8))
  velocity[:,0,2:6]=[.4,.2];confidence[:,0,2:6]=1
  self.assertLess(looming_from_flow(dict(angular_rad_s=velocity,confidence=confidence)),1e-9)
  velocity[:]=0;confidence[:]=0;velocity[:,1,5]=[.8,0];confidence[:,1,5]=1
  self.assertEqual(looming_from_flow(dict(angular_rad_s=velocity,confidence=confidence)),0)

 def test_taste_requires_oral_contact_and_distinguishes_composition(self):
  body=MujocoBody(variant='mk1',prey_count=2);world=World();c=Config()
  world=replace(world,odor_x=590,odor_y=390)
  np.testing.assert_array_equal(taste_sample(body,world,c),0)
  p=mouth_position(body);world=replace(world,odor_x=(p[0]+3)*100,odor_y=(2-p[1])*100)
  world.food_position_m # physical ball is on the floor, so lower the mouth to it for the controlled test
  body.data.qpos[2]-=p[2]-world.food_position_m[2];mujoco.mj_forward(body.model,body.data)
  ball=taste_sample(body,world,c);self.assertGreater(ball[:,0].mean(),ball[:,1].mean());self.assertTrue(np.isfinite(ball).all())
  world=replace(world,food_enabled=False)
  for k in range(2):
   j=int(body.model.joint(f'prey{k}_root').qposadr[0]);body.data.qpos[j:j+3]=mouth_position(body)+[0,0,.005]
  mujoco.mj_forward(body.model,body.data)
  prey=taste_sample(body,world,c);self.assertGreater(prey[:,1].mean(),prey[:,0].mean());self.assertTrue(np.isfinite(prey).all());self.assertTrue(((prey>=0)&(prey<=1)).all())

 def test_tactile_contact_is_local_and_decays(self):
  b=MujocoBody(variant='mk1');s=ExternalSenses(Config(somatic_enabled=True));s.bind_model(b.model)
  for _ in range(50):mujoco.mj_step(b.model,b.data);s.observe(b,.002)
  self.assertGreater(s.tactile[:,0].sum(),0);self.assertLess(s.tactile[6:,0].sum(),1e-9)
  b.data.qpos[2]+=1;mujoco.mj_forward(b.model,b.data)
  for _ in range(100):s.observe(b,.002)
  self.assertLess(s.tactile[:,0].max(),1e-3)

 def test_sensor_afferents_do_not_command_motor_or_dopamine(self):
  from veguinha_mini.network import build_network
  from veguinha_mini.sensory import inject_sensors
  start_scope();c=Config(scale=.1,backend='numpy',phase5_enabled=True,neural_value_channels=True,somatic_enabled=True)
  m=build_network(c,record_spikes=False);dn=m.regions['dn'];go,hold=np.array_split(dn[~m.inhibitory[dn]],2);m.intent=SimpleNamespace(go=go,hold=hold)
  sample=dict(angles=np.zeros((6,2)),contacts=[False]*6,touch=False,energy=.9,flow=[0,0,0],energy_cost_rate=0,antenna_odor=[0,0])
  inject_sensors(m,sample,config=c);base=np.array(m.group.I_sense[:]);weights=np.array(m.synapses.w[:])
  sample.update(looming=.8,taste=[[.3,.9,.2]]*2,tactile=np.ones((9,3)).tolist())
  inject_sensors(m,sample,config=c);after=np.array(m.group.I_sense[:])
  self.assertGreater(after[hold].mean(),base[hold].mean());self.assertGreater(after[m.regions['seg']].max(),base[m.regions['seg']].max())
  motors=np.concatenate([np.concatenate(list(p.values())) for p in m.motor_pools.values()]);np.testing.assert_array_equal(after[motors],base[motors]);np.testing.assert_array_equal(after[m.regions['dan']],0);np.testing.assert_array_equal(m.synapses.w[:],weights)

if __name__=='__main__':unittest.main()
