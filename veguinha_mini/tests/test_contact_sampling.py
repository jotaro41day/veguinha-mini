import unittest,numpy as np
from veguinha_mini.mujoco_body import MujocoBody
class ContactSamplingTests(unittest.TestCase):
 def test_interval_sensors_do_not_change_physics(self):
  a=MujocoBody(variant='mk1');b=MujocoBody(variant='mk1',sample_contacts=True)
  for _ in range(100):
   a.step(.02,np.zeros((6,3)));b.step(.02,np.zeros((6,3)))
  observed=[];command=np.random.default_rng(42).uniform(-1,1,(6,3))
  a.step(.02,command);b.step(.02,command,observe=lambda x:observed.append(x._instant_foot_loads()))
  np.testing.assert_array_equal(a.data.qpos,b.data.qpos)
  np.testing.assert_array_equal(a.data.qvel,b.data.qvel)
  loads=np.asarray(observed);self.assertEqual(loads.shape,(10,6))
  np.testing.assert_allclose(b.mean_normal_load,loads.mean(0))
  np.testing.assert_allclose(b.load_signal,np.clip(loads/b.foot_weight,0,1).mean(0))
  np.testing.assert_allclose(b.contact_fraction,(loads>.02).mean(0))
  self.assertTrue(np.any(loads>.02))
 def test_contact_metrics_integrate_pair_support_before_averaging(self):
  from types import SimpleNamespace
  from veguinha_mini.metrics import Metrics
  body=SimpleNamespace(position=np.zeros(2),energy=1.,angles=np.zeros((6,3)),joint_velocity=np.zeros((6,3)),velocity=np.zeros(2),heading=0.,angular_velocity=0.,touch=False)
  m=Metrics(body);samples=np.zeros((10,6));samples[:5,:2]=1
  m.update(body,np.zeros((6,3)),contact_samples=samples)
  r=m.snapshot(body)
  self.assertEqual(r['contact_fraction'][0],.5)
  self.assertEqual(r['contact_correlation'][0],1.)
if __name__=='__main__':unittest.main()

