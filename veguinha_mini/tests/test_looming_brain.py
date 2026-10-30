import unittest
import numpy as np
from veguinha_mini.config import Config
from veguinha_mini.somatic import looming_from_flow
from veguinha_mini.tests import test_motor_intent


class LoomingBrainTests(unittest.TestCase):
 def test_expansion_drives_spiking_competition_to_suppress_motors(self):
  y,x=np.meshgrid((np.arange(3)+.5)/3-.5,(np.arange(8)+.5)/8-.5,indexing='ij')
  field=np.stack([x,y],axis=-1)
  risk=looming_from_flow(dict(angular_rad_s=np.stack([field,field]),confidence=np.ones((2,3,8))))
  for seed in (42,43,44):
   probe=test_motor_intent.MotorIntentTests();baseline=probe.probe(14,1.6,seed=seed,measure_ms=300,threat_hold=1.6+Config().looming_gain*risk,fast_brake=True)
   late=float(np.mean([b['motor_hz'] for b in probe.reflex_blocks[-3:]]))
   print(dict(seed=seed,looming=risk,motors_baseline_hz=baseline[0],late_motor_hz=late,blocks=probe.reflex_blocks),flush=True)
   self.assertGreater(baseline[0],1);self.assertLess(late,baseline[0]*.25)
   self.assertLess(probe.reflex_blocks[1]['inhibition'],-40)
   self.assertTrue(all(b['motor_hz']==0 for b in probe.reflex_blocks[1:]))
   self.assertLess(max(b['gate_hz'] for b in probe.reflex_blocks),150)

if __name__=='__main__':unittest.main()
