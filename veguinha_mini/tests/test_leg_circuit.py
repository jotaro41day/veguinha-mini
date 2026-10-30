import unittest
import numpy as np
from types import SimpleNamespace
from veguinha_mini.config import Config
from veguinha_mini.motor import MotorDecoder
from veguinha_mini.leg_circuit import build_leg, FixedTrunk, sense
from veguinha_mini.tools.leg_reflex_check import joint_metrics


class LegCircuitTests(unittest.TestCase):
    def test_motor_pool_growth_preserves_upstream_and_input_degree(self):
        a=build_leg(seed=42,backend='numpy')
        for size in (16,32):
            b=build_leg(seed=42,backend='numpy',motor_size=size)
            self.assertEqual(b.motors.shape,(3,2,size))
            np.testing.assert_array_equal(a.premotor,b.premotor)
            np.testing.assert_array_equal(a.group.v[a.premotor.ravel()],b.group.v[b.premotor.ravel()])
            degree=np.bincount(np.asarray(b.synapses.j[:]),minlength=len(b.group))
            np.testing.assert_array_equal(degree[b.motors.ravel()],22)

    def test_motor_adaptation_diversity_preserves_mean_and_upstream_network(self):
        a=build_leg(seed=42,backend='numpy')
        b=build_leg(seed=42,backend='numpy',motor_heterogeneity=True)
        d=np.asarray(b.group.d[:])[b.motors]
        np.testing.assert_allclose(d.mean(2),8)
        self.assertTrue((d.std(2)>0).all())
        np.testing.assert_array_equal(a.group.v[:],b.group.v[:])
        np.testing.assert_array_equal(a.group.d[a.premotor.ravel()],b.group.d[b.premotor.ravel()])
        np.testing.assert_array_equal(a.synapses.w[:],b.synapses.w[:])

    def test_excess_load_brake_is_negative_feedback_on_pressing_motors(self):
        p=build_leg(seed=42,backend='numpy',load_brake=True)
        sense(p,np.zeros(3),.316*9.81/6,position_gain=4,load_gain=12)
        self.assertTrue((np.asarray(p.group.I_sense[:])[p.load_brake]==0).all())
        sense(p,np.zeros(3),1.5,position_gain=4,load_gain=12)
        self.assertTrue((np.asarray(p.group.I_sense[:])[p.load_brake]>0).all())
        src=np.asarray(p.synapses.i[:]);dst=np.asarray(p.synapses.j[:]);w=np.asarray(p.synapses.w[:])
        mask=np.isin(src,p.load_brake)
        self.assertTrue(mask.any());self.assertTrue((w[mask]<0).all())
        self.assertTrue(np.isin(dst[mask],p.motors[1:,1]).all())
        self.assertFalse(np.any(np.isin(src,p.load.ravel())))
        self.assertTrue((np.asarray(p.group.I_sense[:])[p.motors.ravel()]==0).all())

    def test_compliant_guide_passively_replaces_other_five_legs(self):
        b=FixedTrunk(horizontal_free=True,walls=False,compliant=True)
        self.assertGreater(b.support_constants['stiffness'],0)
        self.assertGreater(b.support_constants['damping'],0)
        self.assertAlmostEqual(b.support_constants['preload_n'],.316*9.81*5/6,places=3)
        start=b.data.xpos[b.root].copy()
        for _ in range(50):b.step(np.zeros(3))
        self.assertAlmostEqual(b.load,.316*9.81/6,delta=.015)
        self.assertLess(np.linalg.norm(b.data.xpos[b.root]-start),.001)

    def test_diagnostic_observer_sees_each_physics_substep(self):
        b=FixedTrunk(horizontal_free=True,walls=False)
        times=[]
        b.step(np.zeros(3),observe=lambda body,load: times.append((body.data.time,load)))
        self.assertEqual(len(times),10)
        np.testing.assert_allclose(np.diff([r[0] for r in times]),.002,atol=1e-12)
        self.assertTrue(all(load>=0 for _,load in times))

    def test_end_receptor_recruits_the_reciprocal_inhibitory_reset(self):
        p=build_leg(seed=42,backend='numpy',position_gate=True,gate_memory=True,
                    coherent_reset=True,reset_inhibition=True)
        src=np.asarray(p.synapses.i[:]);dst=np.asarray(p.synapses.j[:]);w=np.asarray(p.synapses.w[:])
        mask=p.kinds=='sensory_reset_inhibition'
        self.assertTrue(mask.any());self.assertTrue((w[mask]>0).all())
        self.assertTrue(np.isin(src[mask],p.position[0]).all())
        self.assertTrue(np.isin(dst[mask],p.inhibitory).all())

    def test_shared_inhibition_is_dale_consistent_and_neighbor_normalized(self):
        p=build_leg(seed=42,backend='numpy',shared_inhibition=True)
        mask=p.kinds=='interjoint_inhibition'
        src=np.asarray(p.synapses.i[:])[mask];dst=np.asarray(p.synapses.j[:])[mask]
        w=np.asarray(p.synapses.w[:])[mask]
        self.assertGreater(len(w),0);self.assertTrue((w<0).all())
        self.assertTrue(np.isin(src,p.inhibitory).all())
        totals=np.bincount(dst,weights=abs(w),minlength=len(p.group))[p.premotor.ravel()]
        np.testing.assert_allclose(totals,totals[0])

    def test_open_guide_does_not_contaminate_leg_test_with_arena_walls(self):
        b=FixedTrunk(horizontal_free=True,walls=False)
        self.assertTrue(all(b.model.geom_contype[i]==0 and b.model.geom_conaffinity[i]==0 for i in b.wall_ids))
        b.data.qpos[b.model.joint('bench_x').qposadr]=10
        b.step(np.zeros(3))
        self.assertFalse(any(int(c.geom1) in b.wall_ids or int(c.geom2) in b.wall_ids for c in b.data.contact))

    def test_end_stop_reset_has_equal_proximal_and_distal_gain(self):
        p=build_leg(seed=42,backend='numpy',position_gate=True,gate_memory=True,coherent_reset=True)
        src=np.asarray(p.synapses.i[:]);dst=np.asarray(p.synapses.j[:]);w=np.asarray(p.synapses.w[:])
        for action in range(2):
            source=p.position[0,action]
            proximal=w[np.isin(src,source)&np.isin(dst,p.premotor[0,action])]
            distal=w[np.isin(src,source)&np.isin(dst,p.premotor[1,action])]
            self.assertAlmostEqual(proximal.mean(),distal.mean())

    def test_motor_copy_memory_does_not_fake_a_position_event(self):
        p=build_leg(seed=42,backend='numpy',position_gate=True,gate_memory=True)
        p.network.run(1000*p.ms)
        self.assertEqual(int(np.asarray(p.monitor.count[:])[p.position[0].ravel()].sum()),0)

    def test_position_comparator_receives_efference_and_only_senses_currents(self):
        p=build_leg(seed=42,backend='numpy',position_gate=True)
        sense(p,[-.1,0,0],.516,position_gain=4,load_gain=12)
        self.assertTrue((np.asarray(p.group.I_base[:])[p.position[0].ravel()]==-2).all())
        self.assertTrue((np.asarray(p.group.I_sense[:])[p.premotor.ravel()]==0).all())
        self.assertTrue((np.asarray(p.group.I_sense[:])[p.motors.ravel()]==0).all())
        self.assertTrue(np.any(p.kinds=='efference_gate'))

    def test_silencing_suppresses_all_motor_output_spikes(self):
        p=build_leg(seed=42,backend='numpy');p.group.spike_gate=0
        p.network.run(200*p.ms)
        self.assertEqual(int(np.sum(p.monitor.count[:])),0)

    def test_metrics_do_not_turn_silence_into_an_oscillator(self):
        for r in joint_metrics(np.zeros((1000,3)),np.zeros((1000,3))):
            self.assertFalse(r['rhythmic']);self.assertIsNone(r['period_s'])
            self.assertAlmostEqual(r['fractions']['stopped'],1)
        t=np.arange(1000)*.02
        q=np.sin(2*np.pi*t[:,None]/.4+np.array([0,np.pi/2,np.pi]))
        r=joint_metrics(q,.2*q)
        self.assertTrue(all(a['rhythmic'] for a in r))
        self.assertAlmostEqual(r[1]['relative_phase_deg'],90,places=5)

    def test_clipped_command_can_be_periodic_and_saturated(self):
        t=np.arange(1000)*.02
        q=np.sin(2*np.pi*t[:,None]/.4+np.array([0,.4,.8]))
        for r in joint_metrics(q,np.sign(q)):
            self.assertTrue(r['rhythmic'])
            self.assertAlmostEqual(r['fractions']['oscillating'],1)
            self.assertGreater(r['fractions']['saturated'],.9)

    def test_periodic_harmonics_are_not_misclassified_as_noise(self):
        phi=2*np.pi*np.arange(1000)[:,None]*.02/.4+np.array([0,.4,.8])
        q=.5*np.sin(phi)+np.sin(2*phi)+.7*np.sin(3*phi)
        self.assertTrue(all(r['rhythmic'] for r in joint_metrics(q,.2*np.sin(phi))))
        noise=np.random.default_rng(42).normal(0,.1,(1000,3))
        self.assertFalse(any(r['rhythmic'] for r in joint_metrics(noise,np.zeros_like(noise))))

    def test_one_leg_decoder_uses_original_motor_readout(self):
        names = [f'{j}_{a}' for j in ('coxa','femur','tibia') for a in ('flex','extend')]
        pools = dict(zip(names,np.arange(48).reshape(6,8)))
        model=SimpleNamespace(group=np.zeros(192),motor_pools={'lf':pools})
        d=MotorDecoder(model,Config(body_model='mk1'),legs=('lf',))
        count=np.zeros(192,dtype=int);count[pools['coxa_flex']]=1
        torque=d.update(count,.02)
        self.assertEqual(torque.shape,(1,3));self.assertGreater(torque[0,0],0)
        self.assertEqual(np.count_nonzero(torque),1)

    def test_three_joint_motif_preserves_dale_and_local_inhibition(self):
        p=build_leg(seed=42,joints=3,backend='numpy')
        self.assertEqual(len(p.group),196) # Three 64-cell motifs + four load receptors.
        self.assertEqual(p.motors.shape,(3,2,8))
        weights=np.asarray(p.synapses.w[:]);sources=np.asarray(p.synapses.i[:])
        self.assertTrue((weights[np.isin(sources,p.inhibitory.ravel())]<0).all())
        self.assertTrue((weights[~np.isin(sources,p.inhibitory.ravel())]>0).all())
        self.assertTrue(np.allclose(p.synapses.delay[:]/p.ms,2))

    def test_fixed_root_and_same_transmission(self):
        b=FixedTrunk()
        root=b.data.xpos[b.root].copy()
        for _ in range(10):b.step(np.array([.2,-.2,.2]))
        np.testing.assert_allclose(b.data.xpos[b.root],root,atol=1e-12)
        self.assertAlmostEqual(b.model.body_mass.sum(),.316)
        np.testing.assert_allclose(b.model.actuator_gear[b.actuators,0],[.08,.2,.2])


if __name__=='__main__':unittest.main()
