import unittest
import numpy as np
from veguinha_mini.leg_circuit import build_leg

class SensoryOutputTests(unittest.TestCase):
    def test_reroute_preserves_cells_weights_and_non_sensory_edges(self):
        a=build_leg(42,coupling=2,motor_strength=12,backend='numpy')
        b=build_leg(42,coupling=2,motor_strength=12,backend='numpy',sensory_output=True)
        np.testing.assert_array_equal(a.group.v[:],b.group.v[:])
        mask=np.isin(a.kinds,['stretch','end_coxa','loaded','unloaded'])
        other=np.isin(b.kinds,['stretch','end_coxa','loaded','unloaded'])
        for attr in ('i','j','w','slow'):
            np.testing.assert_array_equal(np.asarray(getattr(a.synapses,attr)[:])[~mask],np.asarray(getattr(b.synapses,attr)[:])[~other])
        for kind in ('stretch','end_coxa','loaded','unloaded'):
            np.testing.assert_array_equal(np.unique(a.synapses.w[a.kinds==kind]),np.unique(b.synapses.w[b.kinds==kind]))
        self.assertTrue(np.isin(np.asarray(b.synapses.j[:])[other],b.motors).all())
        self.assertFalse(np.isin(np.asarray(b.synapses.j[:])[other],b.premotor).any())
        self.assertTrue((np.asarray(b.synapses.w[:])[other]>0).all())
        ia=np.bincount(np.asarray(a.synapses.j[:])[mask],weights=np.asarray(a.synapses.w[:])[mask],minlength=len(a.group))
        ib=np.bincount(np.asarray(b.synapses.j[:])[other],weights=np.asarray(b.synapses.w[:])[other],minlength=len(b.group))
        for j in range(3):
            for action in range(2):
                np.testing.assert_allclose(ib[b.motors[j,action]],ia[a.premotor[j,action,0]])

    def test_excitability_diversity_preserves_core_and_subthreshold_tonic_drive(self):
        a=build_leg(42,coupling=2,motor_strength=12,backend='numpy',sensory_output=True)
        b=build_leg(42,coupling=2,motor_strength=12,backend='numpy',sensory_output=True,motor_excitability=True)
        np.testing.assert_array_equal(a.group.v[:],b.group.v[:])
        np.testing.assert_array_equal(a.synapses.w[:],b.synapses.w[:])
        np.testing.assert_array_equal(a.group.b[a.premotor.ravel()],b.group.b[b.premotor.ravel()])
        params=np.asarray(b.group.b[:])[b.motors]
        np.testing.assert_allclose(params.mean(2),.2)
        self.assertTrue((params.std(2)>0).all())
        fold_current=(5-params)**2/.16-140
        self.assertTrue((fold_current>2).all())
        rest=(-(5-params)-np.sqrt((5-params)**2-.16*(140+2)))/.08
        self.assertTrue((.08*rest+5-.02<0).all())

    def test_diverse_motors_do_not_self_oscillate_without_upstream_events(self):
        from brian2 import start_scope,ms
        from veguinha_mini.tools.motor_pool_bench import configure_leg
        start_scope()
        p=build_leg(42,coupling=2,motor_strength=12,backend='numpy',sensory_output=True,motor_excitability=True)
        configure_leg(p,'tonic_threshold')
        p.group.spike_gate=0;p.group.spike_gate[p.motors.ravel()]=1
        from veguinha_mini.network import EmittedSpikeMonitor
        events=EmittedSpikeMonitor(p.group,'emit',record=True,name='quiet_test')
        p.network.add(events);p.network.run(2000*ms)
        # Ten motor adaptation time constants; initial 2-mV voltage scatter may spike.
        self.assertFalse(np.any(np.asarray(events.t/ms)>=500))

    def test_full_ablation_stops_all_motor_torque(self):
        from brian2 import start_scope,ms
        from veguinha_mini.tools.motor_pool_bench import configure_leg
        from veguinha_mini.motor import MotorDecoder
        start_scope()
        p=build_leg(42,coupling=2,motor_strength=12,backend='numpy',sensory_output=True,motor_excitability=True)
        configure_leg(p,'tonic_threshold');p.group.spike_gate=0
        p.network.run(200*ms)
        decoder=MotorDecoder(p,p.config,legs=('lf',))
        np.testing.assert_array_equal(decoder.update(np.asarray(p.monitor.count[:]),.02),np.zeros((1,3)))

if __name__=='__main__':unittest.main()



