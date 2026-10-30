import unittest,numpy as np
from brian2 import start_scope
class CordTests(unittest.TestCase):
 def test_fused_local_modules_preserve_topology_and_dale(self):
  from veguinha_mini.cord import build_cord
  start_scope();p=build_cord(42,cross_gain=2,backend='numpy',warm=False)
  self.assertEqual(len(p.group),1176)
  self.assertEqual(p.motors.shape,(6,3,2,8))
  i=np.asarray(p.synapses.i[:]);w=np.asarray(p.synapses.w[:]);j=np.asarray(p.synapses.j[:])
  self.assertTrue((w[np.isin(i,p.inhibitory)]<0).all())
  self.assertTrue((w[~np.isin(i,p.inhibitory)]>0).all())
  cross=p.edge_kind==2
  self.assertTrue(cross.any());self.assertTrue(np.isin(i[cross],p.inhibitory).all())
  self.assertTrue(np.isin(j[cross],p.premotor).all())
  incoming=np.bincount(j[cross],weights=w[cross],minlength=len(p.group))[p.premotor.ravel()]
  np.testing.assert_allclose(incoming,incoming[0])
  self.assertTrue((np.asarray(p.group.noise[:])[p.motors.ravel()]==0).all())
 def test_zero_strength_disables_edges_without_changing_anatomy(self):
  from veguinha_mini.cord import build_cord
  start_scope();p=build_cord(42,cross_gain=0,backend='numpy',warm=False)
  self.assertTrue((p.edge_kind==2).any())
  self.assertTrue((np.asarray(p.synapses.w[:])[p.edge_kind==2]==0).all())
  q=build_cord(42,cross_gain=2,backend='numpy',warm=False)
  np.testing.assert_array_equal(p.synapses.i[:],q.synapses.i[:])
  np.testing.assert_array_equal(p.synapses.j[:],q.synapses.j[:])
 def test_cross_current_changes_the_membrane_voltage(self):
  from veguinha_mini.cord import build_cord
  from brian2 import ms
  start_scope();p=build_cord(42,cross_gain=2,backend='numpy',warm=False)
  g=p.group;g.noise=0;g.I_base=0;g.I_noise=0;g.I_syn=0;g.I_slow=0;g.I_cross=0;g.v=-65;g.u=-13;g.spike_gate=0
  g.I_cross[0]=-10;p.network.run(.5*ms)
  self.assertAlmostEqual(float(g.v[0]-g.v[1]),-5,places=8)
if __name__=='__main__':unittest.main()

