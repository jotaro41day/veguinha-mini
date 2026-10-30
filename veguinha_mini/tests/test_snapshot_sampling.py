import unittest
from types import SimpleNamespace
import numpy as np
from brian2 import start_scope,ms
from veguinha_mini.config import Config
from veguinha_mini.sim import Simulation

class SnapshotSamplingTests(unittest.TestCase):
 def test_frame_reads_only_the_eight_spikes_it_transmits(self):
  start_scope();s=Simulation(Config(scale=.05,backend='numpy'));s.speed='turbo';s.model.network.run(100*ms)
  mon=s.model.monitor;n=int(mon.num_events);self.assertGreater(n,8)
  times=mon.t[:];ids=np.asarray(mon.i[:]);indices=np.linspace(0,n-1,8,dtype=np.int32)
  expected=[[round(float(t),1),int(r)] for t,r in zip(np.asarray(times[indices]/ms,dtype=np.float32),s.region_index[ids[indices]])]
  class BoundedRead:
   def __init__(self,values):self.values=values
   def __getitem__(self,selection):
    selected=self.values[selection]
    if len(selected)>8:raise AssertionError('leitura desnecessária do histórico inteiro')
    return selected
  s.model.monitor=SimpleNamespace(t=BoundedRead(times),i=BoundedRead(ids),count=np.asarray(mon.count[:]),num_events=n)
  s.spike_cursor=0;s._publish(0)
  self.assertEqual(s.latest()['spikes'],expected)

if __name__=='__main__':unittest.main()
