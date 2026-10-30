import unittest,tempfile
from pathlib import Path
from dataclasses import replace
import numpy as np
import mujoco
from veguinha_mini.config import Config
from veguinha_mini.world import World
from veguinha_mini.mujoco_body import MujocoBody


class EcologyPhysicsTests(unittest.TestCase):
    def test_appended_prey_preserve_hexapod_inertia_and_motor_mapping(self):
        a=MujocoBody(variant='mk1');b=MujocoBody(variant='mk1',prey_count=2)
        self.assertEqual(b.model.nu,a.model.nu+6);self.assertEqual(b.model.nq,a.model.nq+20)
        for index in range(a.model.nbody):
            name=a.model.body(index).name;j=b.model.body(name).id
            np.testing.assert_array_equal(a.model.body_mass[index],b.model.body_mass[j]);np.testing.assert_array_equal(a.model.body_inertia[index],b.model.body_inertia[j])
        torque=np.zeros((6,3));torque[3,0]=.3;b.step(.02,torque)
        self.assertEqual(b.data.ctrl[b.motor_ids[9]],.3)
        np.testing.assert_array_equal(b.data.ctrl[b.prey_motor_ids.ravel()],0)

    def test_dust_requires_rubbing_and_rest_recovers_fatigue(self):
        from veguinha_mini.ecology import EcologyState
        b=MujocoBody(variant='mk1',prey_count=1);c=Config(ecology_enabled=True,motor_circuit='modular',body_model='mk1',dt_ms=.5,intent_enabled=True)
        e=EcologyState(c,b);e.dust[:]=.5
        e.rub(7,0,1,.1);self.assertEqual(e.dust[7],.5)
        e.rub(7,1,0,.1);self.assertEqual(e.dust[7],.5)
        e.rub(7,1,1,.1);self.assertLess(e.dust[7],.5)
        before=e.dust.copy();e.shed(np.zeros(8),.1);np.testing.assert_array_equal(e.dust,before)
        e.shed(np.r_[np.full(6,60.),0,0],.1);self.assertTrue(np.all(e.dust[:6]<before[:6]));self.assertGreater(e.shaken.sum(),0)
        e.fatigue=.7;e.recover(.1,0,1,0);self.assertLess(e.fatigue,.7)
        old=e.fatigue;e.recover(.1,1,0,1);self.assertGreater(e.fatigue,old)

    def test_prey_intake_is_finite_requires_neural_bite_and_overlap(self):
        from veguinha_mini.ecology import EcologyState
        from veguinha_mini.perception import mouth_position
        b=MujocoBody(variant='mk1',prey_count=1);e=EcologyState(Config(),b);b.ecology_state=e;b.energy=.2
        joint=b.model.joint('prey0_root');p=mouth_position(b);b.data.qpos[int(joint.qposadr[0]):int(joint.qposadr[0])+3]=p;mujoco.mj_forward(b.model,b.data)
        self.assertEqual(e.feed(b,0,.1),0)
        before=e.prey_energy.sum();old=b.energy;taken=e.feed(b,1,.1)
        self.assertGreater(taken,0);self.assertLess(e.prey_health[0],1);self.assertAlmostEqual(b.energy-old,taken)
        self.assertAlmostEqual(before-e.prey_energy.sum(),taken)
        joint=b.model.joint('prey0_root');b.data.qpos[int(joint.qposadr[0])]+=2;mujoco.mj_forward(b.model,b.data)
        self.assertEqual(e.feed(b,1,.1),0)

    def test_shadow_is_fixed_and_seen_in_rgb(self):
        from veguinha_mini.perception import ExternalSenses
        w=World(shade_enabled=True);self.assertLess(w.illumination_at(np.array([[-2,1.1]]))[0],.2)
        b=MujocoBody(variant='mk1');c=Config(vision_enabled=True,binocular_enabled=True,vision_width=80,vision_height=45);sense=ExternalSenses(c);sense.bind_model(b.model)
        sense.scan(b,w);bright=sense.rgb.mean()
        b.data.qpos[0]=-2.35;b.data.qpos[1]=1.1;mujoco.mj_forward(b.model,b.data);sense.scan(b,w)
        self.assertLess(sense.rgb.mean(),bright*.85)
        self.assertEqual(w.snapshot()['shade']['center_m'],[-2.,1.1])

    def test_floor_prey_is_reachable_by_the_settled_mouth(self):
        from veguinha_mini.ecology import EcologyState
        from veguinha_mini.perception import mouth_position
        b=MujocoBody(variant='mk1',prey_count=1)
        for _ in range(100):b.step(.02,np.zeros((6,3)))
        state=EcologyState(Config(),b);point=mouth_position(b)
        adr=int(b.model.joint('prey0_root').qposadr[0]);b.data.qpos[adr:adr+2]=point[:2]-np.array([.025,0])
        b.data.qpos[adr+2]=b.model.geom_size[b.model.geom('prey0_collision0').id,0]+.0001
        mujoco.mj_forward(b.model,b.data)
        self.assertGreater(state.mouth_contact(b)[0],.1)

class PreyNeuralTests(unittest.TestCase):
    def test_400_neurons_sparse_dale_and_silence(self):
        from brian2 import start_scope,Network,Clock,ms,prefs
        from veguinha_mini.ecology import PreyCircuit
        start_scope();prefs.codegen.target='numpy';net=Network();p=PreyCircuit(Config(),Clock(dt=1*ms,name='prey_test_clock'),net,1)
        self.assertEqual(len(p.group),400);self.assertGreater(len(p.synapses),2000);self.assertLess(len(p.synapses),8000)
        pre=np.asarray(p.synapses.i[:]);w=np.asarray(p.synapses.w[:]);self.assertTrue(np.all(w[p.inhibitory[pre]]<=0));self.assertTrue(np.all(w[~p.inhibitory[pre]]>=0))
        net.run(300*ms,namespace={});p.decode(.3);self.assertGreater(p.monitor.num_events,0)
        p.group.spike_gate=0;p.previous=np.asarray(p.monitor.count[:]).copy();p.rates[:]=0
        net.run(100*ms,namespace={});np.testing.assert_array_equal(p.decode(.1),0)


class EcologyIntegrationTests(unittest.TestCase):
    def config(self):
        return Config(body_model='mk1',motor_circuit='modular',dt_ms=.5,block_ms=20,backend='numpy',ecology_enabled=True,prey_count=1,intent_enabled=True,feeding_enabled=True)

    def test_optional_circuits_reach_drives_and_preserve_checkpoint(self):
        from brian2 import start_scope
        from veguinha_mini.sim import Simulation
        start_scope();c=self.config();s=Simulation(c,headless=True,physics='mujoco',body_substep_ms=2)
        self.assertEqual(len(s.prey.group),400);self.assertEqual(len(s.maintenance.group),100)
        e=s.body.ecology_state;e.dust[:]=1;e.fatigue=1;e.light[:]=.1;s._ecology_input()
        current=np.asarray(s.maintenance.group.I_sense[:]).copy();self.assertGreater(current[s.maintenance.exc[0]].mean(),10)
        self.assertGreater(current[s.maintenance.exc[2]].mean(),10)
        
        mask=np.zeros(100,bool);mask[s.maintenance.inh.ravel()]=True
        for syn in s.maintenance.synapses:
            pre=np.asarray(syn.i[:]);w=np.asarray(syn.w[:]);self.assertTrue(np.all(w[mask[pre]]<=0));self.assertTrue(np.all(w[~mask[pre]]>=0))
        s.prey.enabled=False;s._ecology_input();self.assertFalse(np.any(s.prey.group.spike_gate[:]))
        weights=np.asarray(s.model.synapses.w[:]).copy();s.paused=True
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'eco';s.save_checkpoint(p);q=Simulation(c,headless=True,physics='mujoco',body_substep_ms=2);q.load_checkpoint(p)
            np.testing.assert_array_equal(q.body.data.qpos,s.body.data.qpos);np.testing.assert_array_equal(q.body.ecology_state.dust,e.dust)
            np.testing.assert_array_equal(q.model.synapses.w[:],weights)
            q.paused=False;q.run_headless(.04);self.assertTrue(np.isfinite(q.body.data.qpos).all())

    def test_migration_appends_prey_without_changing_saved_root(self):
        from veguinha_mini.ecology import migrate_body
        a=MujocoBody(variant='mk1');a.data.qpos[0]=-.7;a.data.qvel[0]=.1;mujoco.mj_forward(a.model,a.data)
        b=MujocoBody(variant='mk1',prey_count=1);state=migrate_body(a.__dict__,b,self.config())
        np.testing.assert_array_equal(state['data'].qpos[:a.model.nq],a.data.qpos)
        np.testing.assert_array_equal(state['data'].qvel[:a.model.nv],a.data.qvel)
        self.assertEqual(state['prey_count'],1)

    def test_prey_appetitive_bias_is_in_sensory_neurons_not_torque_or_dan(self):
        from brian2 import start_scope
        from veguinha_mini.sim import Simulation
        from veguinha_mini.sensory import inject_sensors
        start_scope();c=self.config();s=Simulation(c,headless=True,physics='mujoco',body_substep_ms=2)
        sample=s.body.snapshot(s.world);sample['energy']=.95;sample['antenna_odor']=[.5,.5];sample['prey_odor']=[0,0]
        inject_sensors(s.model,sample,config=c);ball=np.asarray(s.model.group.I_sense[:]).copy()
        sample['antenna_odor']=[0,0];sample['prey_odor']=[.5,.5]
        inject_sensors(s.model,sample,config=c);prey=np.asarray(s.model.group.I_sense[:]).copy()
        self.assertGreater(prey[s.intent.go].mean(),ball[s.intent.go].mean())
        np.testing.assert_array_equal(prey[s.intent.motors],ball[s.intent.motors])
        np.testing.assert_array_equal(prey[s.model.regions['dan']],0)

    def test_radius_migration_preserves_root_and_somatic_state(self):
        from veguinha_mini.ecology import EcologyState,migrate_body
        a=MujocoBody(variant='mk1',prey_count=1,prey_radius_m=.013)
        a.ecology_state=EcologyState(Config(prey_radius_m=.013),a);a.ecology_state.dust[:]=.3
        b=MujocoBody(variant='mk1',prey_count=1,prey_radius_m=.025)
        state=migrate_body(a.__dict__,b,self.config())
        np.testing.assert_array_equal(state['data'].qpos[:25],a.data.qpos[:25])
        np.testing.assert_array_equal(state['ecology_state'].dust,.3)
        self.assertEqual(float(state['model'].geom_size[state['model'].geom('prey0_collision0').id,0]),.025)
        self.assertEqual(float(a.model.geom_size[a.model.geom('prey0_collision0').id,0]),.013)

    def test_left_antenna_matches_left_eye_convention(self):
        b=MujocoBody(variant='mk1');rotation=b.data.xmat[b.body_id].reshape(3,3)
        point=b.data.xpos[b.body_id]+rotation@np.array([.6,.4,0])
        world=World(odor_x=(point[0]+3)*100,odor_y=(2-point[1])*100)
        odor=b.snapshot(world)['antenna_odor'];self.assertGreater(odor[0],odor[1])

if __name__=='__main__':unittest.main()
