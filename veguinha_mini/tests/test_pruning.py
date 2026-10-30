import unittest
from types import SimpleNamespace
import numpy as np
from veguinha_mini.config import Config
from veguinha_mini.learning import MotorPlasticity

class PruningTests(unittest.TestCase):
 def make(self):
  pre=np.repeat(np.arange(24),10);post=np.tile(np.arange(30,40),24);inhibitory=np.zeros(44,bool);inhibitory[20:24]=True
  syn=SimpleNamespace(i=pre,j=post,w=np.where(inhibitory[pre],-1.,1.))
  regions={'vnc_'+leg:np.arange(30+2*k,32+2*k) for k,leg in enumerate(('lf','lm','lh','rf','rm','rh'))};regions.update(dan=np.array([42]),oct=np.array([43]))
  model=SimpleNamespace(group=np.arange(44),synapses=syn,plastic_edges=np.arange(len(pre)),regions=regions,inhibitory=inhibitory)
  p=MotorPlasticity(model,Config(pruning_enabled=True,prune_after_s=.05,learning_rate=0))
  p.weight[:100]=.21;p.initial[:]=1;p.weight[0]=.21
  return p

 def test_disabled_plasticity_prevents_pruning(self):
  p=self.make();before=p.model.synapses.w.copy()
  for _ in range(20):p.update(np.zeros(44,int),.02,enabled=False)
  np.testing.assert_array_equal(p.model.synapses.w,before);self.assertTrue(p.active.all())

 def test_weak_unused_synapses_prune_with_degree_and_importance_protected(self):
  p=self.make();p.importance[0]=1
  for _ in range(60):p.update(np.zeros(44,int),.02)
  self.assertGreater((~p.active).sum(),0);self.assertTrue(p.active[0]);self.assertTrue(p.active[p.sign<0].all())
  for target in np.unique(p.post):self.assertGreaterEqual(np.sum(p.active&(p.post==target)&(p.sign>0)),18)
  self.assertTrue((p.weight[~p.active]==0).all());self.assertTrue((p.model.synapses.w[~p.active]==0).all())
  self.assertTrue((p.sign[p.active]*p.model.synapses.w[p.active]>=0).all())
  p._normalize();self.assertTrue((p.weight[~p.active]==0).all())

 def test_legacy_learning_state_migrates_without_changing_weights(self):
  p=self.make();del p.active,p.importance,p.weak_time,p.usage
  before=p.model.synapses.w.copy();p.update(np.zeros(44,int),.02,enabled=False)
  self.assertTrue(p.active.all());np.testing.assert_array_equal(before,p.model.synapses.w)

 def test_normalization_preserves_original_e_i_budgets_without_regrowing(self):
  p=self.make();p.weight[:]=1;removed=np.array([0,10]);p.active[removed]=False;p.weight[removed]=0
  for _ in range(4):p._normalize()
  np.testing.assert_allclose(np.bincount(p.post[p.sign>0],weights=p.weight[p.sign>0],minlength=44)[30:40],20,rtol=1e-6)
  np.testing.assert_allclose(np.bincount(p.post[p.sign<0],weights=p.weight[p.sign<0],minlength=44)[30:40],4,rtol=1e-6)
  np.testing.assert_array_equal(p.weight[removed],0)

if __name__=='__main__':unittest.main()
