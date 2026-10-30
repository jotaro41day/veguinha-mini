import os,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np

SCRIPT='''
import sys,pickle,numpy as np
from pathlib import Path
from brian2 import start_scope
from veguinha_mini.config import Config
from veguinha_mini.sim import Simulation
start_scope();root=Path(sys.argv[1]);mode=sys.argv[2]
s=Simulation(Config(scale=.05,backend='cython'),headless=True)
if mode=='save':
 s.run_headless(.07);s.save_checkpoint(root/'checkpoint')
else:s.load_checkpoint(root/'checkpoint')
s.run_headless(.03)
np.savez(root/(mode+'.npz'),v=s.model.group.v[:],u=s.model.group.u[:],counts=s.model.monitor.count[:],angles=s.body.angles,position=s.body.position)
'''
class CheckpointProcessTests(unittest.TestCase):
 def test_cython_continuation_is_identical_in_a_fresh_process(self):
  with tempfile.TemporaryDirectory() as directory:
   env={**os.environ,'PYTHONHASHSEED':'0'}
   for mode in ('save','load'):
    result=subprocess.run([sys.executable,'-X','faulthandler','-c',SCRIPT,directory,mode],capture_output=True,text=True,timeout=70,env=env)
    self.assertEqual(result.returncode,0,result.stderr[-5000:])
   a=np.load(Path(directory)/'save.npz');b=np.load(Path(directory)/'load.npz')
   for key in a.files:np.testing.assert_array_equal(a[key],b[key],err_msg=key)

 def test_binocular_cython_resume_preserves_held_images_and_neural_continuation(self):
  script=SCRIPT.replace("Config(scale=.05,backend='cython')","Config(scale=.05,backend='cython',somatic_enabled=True,vision_enabled=True,binocular_enabled=True,optical_flow_enabled=True,foveated_enabled=True,vision_width=64,vision_height=32,vision_interval_ms=40,vision_periphery_ms=20)").replace("headless=True)","headless=True,physics='mujoco',body_substep_ms=2)")
  script=script.replace("position=s.body.position)","position=s.body.position,tactile=s.body.senses.tactile,tactile_previous=s.body.senses.tactile_previous,optic_v=s.binocular.group.v[:],optic_counts=s.binocular.monitor.count[:],rgb=s.body.senses.rgb,retina=s.body.senses.retina,optic_rates=s.binocular.rates,flow_previous=s.body.senses.optical.previous,flow_velocity=s.body.senses.optical.result['angular_rad_s'],flow_confidence=s.body.senses.optical.result['confidence'])")
  with tempfile.TemporaryDirectory() as directory:
   env={**os.environ,'PYTHONHASHSEED':'0'}
   for mode in ('save','load'):
    result=subprocess.run([sys.executable,'-X','faulthandler','-c',script,directory,mode],capture_output=True,text=True,timeout=90,env=env)
    self.assertEqual(result.returncode,0,result.stderr[-5000:])
   a=np.load(Path(directory)/'save.npz');b=np.load(Path(directory)/'load.npz')
   for key in a.files:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
if __name__=='__main__':unittest.main()
