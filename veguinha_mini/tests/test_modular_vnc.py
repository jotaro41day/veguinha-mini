import unittest,numpy as np
from brian2 import start_scope
from veguinha_mini.config import Config
from veguinha_mini.network import build_network
from veguinha_mini.sensory import inject_sensors
class ModularVNCTests(unittest.TestCase):
 def test_existing_brain_uses_corrected_motor_regions_and_sensory_path(self):
  start_scope();c=Config(body_model='mk1',motor_circuit='modular',dt_ms=.5,backend='numpy')
  p=build_network(c,record_spikes=False)
  self.assertEqual(len(p.group),4024)
  self.assertIsNotNone(p.cord)
  pre=np.asarray(p.synapses.i[:]);post=np.asarray(p.synapses.j[:]);w=np.asarray(p.synapses.w[:])
  dn=np.isin(pre,p.regions['dn']) & np.isin(post,np.concatenate([p.regions['vnc_'+l] for l in ('lf','lm','lh','rf','rm','rh')]))
  degree=np.bincount(post,minlength=len(p.group))
  normalized=abs(w[dn])*np.sqrt(degree[post[dn]])/.45
  self.assertTrue((normalized>=.9-1e-6).all() and (normalized<=1.1+1e-6).all())
  self.assertEqual(len(p.regions['vnc_lf']),196)
  for k,l in enumerate(('lf','lm','lh','rf','rm','rh')):
   np.testing.assert_array_equal(p.inhibitory[p.cord.inhibitory[k].ravel()],True)
  p.group.noise=2.5
  self.assertTrue((np.asarray(p.group.noise_gain[:])[p.cord.motors.ravel()]==0).all())
  p.motor_neutral=np.zeros((6,3))
  sample=dict(angles=np.full((6,3),.1),contacts=[True]*6,contact_fraction=[.5]*6,load_signal=[.5]*6,normal_load=[2]*6,touch=False,energy=1,antenna_odor=[0,0])
  inject_sensors(p,sample,config=c)
  current=np.asarray(p.group.I_sense[:])
  self.assertTrue((current[p.cord.premotor.ravel()]==0).all())
  self.assertTrue((current[p.cord.positions.ravel()]>0).any())
  np.testing.assert_allclose(current[p.cord.loads.ravel()],6)
if __name__=='__main__':unittest.main()
