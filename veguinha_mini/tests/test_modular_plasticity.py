import unittest
import numpy as np
from brian2 import start_scope
from veguinha_mini.config import Config
from veguinha_mini.network import build_network
from veguinha_mini.learning import MotorPlasticity
from veguinha_mini.sensory import inject_sensors


class ModularPlasticityTests(unittest.TestCase):
    def test_local_afferents_descending_and_interleg_only(self):
        start_scope()
        config=Config(body_model='mk1',motor_circuit='modular',dt_ms=.5,backend='numpy',
                      phase5_enabled=True,neural_value_channels=True,
                      motor_plastic_scope='afferent_dn_interleg')
        model=build_network(config,record_spikes=False)
        self.assertNotIn('proprio',model.regions)
        self.assertIn('flow',model.regions);self.assertIn('oct',model.regions)
        pre=np.asarray(model.synapses.i[:],int);post=np.asarray(model.synapses.j[:],int)
        leg=np.full(len(model.group),-1)
        for i,name in enumerate(('lf','lm','lh','rf','rm','rh')):leg[model.regions['vnc_'+name]]=i
        positions=model.cord.positions.ravel()
        allowed=np.isin(pre,positions)&(leg[post]>=0)
        allowed|=np.isin(pre,model.regions['dn'])&(leg[post]>=0)
        allowed|=(leg[pre]>=0)&(leg[post]>=0)&(leg[pre]!=leg[post])
        actual=np.zeros(len(pre),bool);actual[model.plastic_edges]=True
        np.testing.assert_array_equal(actual,allowed&(np.asarray(model.synapses.w[:])!=0))
        self.assertTrue((actual & np.isin(pre,positions)).any())
        sample=dict(angles=np.zeros((6,3)),contacts=[True]*6,normal_load=[.5]*6,load_signal=[.5]*6,
                    touch=False,energy=.8,antenna_odor=[0,0],flow=[.1,0,0],energy_cost_rate=.02,upright=1)
        inject_sensors(model,sample,config=config)
        current=np.asarray(model.group.I_sense[:])
        np.testing.assert_array_equal(current[model.regions['dan']],0)
        self.assertTrue((current[model.regions['value']]>0).any())
        plasticity=MotorPlasticity(model,config)
        before=np.asarray(model.synapses.w[:]).copy()
        count=np.zeros(len(model.group),dtype=np.int64)
        for step in range(60):
            count[plasticity.pre[0]]+=1
            plasticity.update(count,.02,enabled=step>=20)
            if step<20:np.testing.assert_array_equal(model.synapses.w[:],before)
            count[plasticity.post[0]]+=1;count[model.regions['dan']]+=1;count[model.regions['oct']]+=1
            plasticity.update(count,.02,enabled=step>=20)
        after=np.asarray(model.synapses.w[:]).copy()
        self.assertTrue((after[actual]!=before[actual]).any())
        np.testing.assert_array_equal(after[~actual],before[~actual])
        for _ in range(20):
            count+=1;plasticity.update(count,.02,enabled=False)
        np.testing.assert_array_equal(model.synapses.w[:],after)

    def test_disabled_time_does_not_create_update_backlog(self):
        start_scope();config=Config(scale=.05,backend='numpy',phase5_enabled=True)
        model=build_network(config,record_spikes=False);learning=MotorPlasticity(model,config)
        count=np.zeros(len(model.group),dtype=np.int64)
        for _ in range(250):learning.update(count,.02,enabled=False)
        self.assertGreater(learning.next_weight_update,learning.elapsed)
        self.assertGreater(learning.next_normalization,learning.elapsed)


if __name__=='__main__':unittest.main()
