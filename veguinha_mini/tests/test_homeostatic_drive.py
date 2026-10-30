"""Switch the same spiking network: no reset between physiological conditions."""
import unittest
from types import SimpleNamespace
import numpy as np
from brian2 import Clock,Network,NeuronGroup,ms,prefs,start_scope
from veguinha_mini.config import Config
from veguinha_mini.network import EQUATIONS,EmittedSpikeMonitor
from veguinha_mini.intent import build_intent


def intent_fixture(seed):
    start_scope();prefs.codegen.target='cython';rng=np.random.default_rng(seed)
    g=NeuronGroup(140,EQUATIONS.replace(' + I_syn + I_noise',' + I_syn + I_cross + I_noise')+'\ndI_cross/dt=-I_cross/(40*ms):1',threshold='v>=30',reset='v=c;u+=d',events={'emit':'v>=30 and spike_gate>0'},method='euler',clock=Clock(dt=.5*ms,name='clock'),name='drive_cells')
    g.a=.02;g.b=.2;g.c=-65;g.d=8;g.v=-65+rng.normal(0,1,140);g.u='b*v';g.spike_gate=1;g.tau_syn=5*ms
    inhibitory=np.zeros(140,bool);inhibitory[:24]=True;g.a[:24]=.1;g.d[:24]=2
    pools={leg:{'flex':np.array([128+2*k]),'extend':np.array([129+2*k])} for k,leg in enumerate(('lf','lm','lh','rf','rm','rh'))}
    monitor=EmittedSpikeMonitor(g,'emit',record=False,name='eventmonitor')
    model=SimpleNamespace(group=g,regions={'dn':np.arange(96),'oct':np.arange(96,128)},inhibitory=inhibitory,motor_pools=pools,network=Network(g,monitor),monitor=monitor)
    intent=build_intent(model,Config(seed=seed));g.I_base[128:]=6
    return model,intent


class HomeostaticDriveTests(unittest.TestCase):
    def test_same_network_can_leave_rest_eat_and_resume_without_reset(self):
        self.rows=[]
        for seed in (42,43,44):
            model,intent=intent_fixture(seed);rows=[]
            for name,go,hold in [('rest',0,8),('need',7.2,4.8),('eat',7.2,34.8),('need_again',7.2,4.8)]:
                model.group.I_sense[intent.go]=go;model.group.I_sense[intent.hold]=hold
                model.network.run(500*ms);before=np.asarray(model.monitor.count[:]).copy();model.network.run(300*ms)
                hz=(np.asarray(model.monitor.count[:])-before)/.3
                row=dict(seed=seed,condition=name,motor_hz=float(hz[128:].mean()),go_hz=float(hz[intent.go].mean()),hold_hz=float(hz[intent.hold].mean()),go_inhibition=float(np.asarray(model.group.I_cross[intent.go]).mean()))
                rows.append(row);print(row,flush=True)
            self.rows.extend(rows)
            self.assertGreater(rows[1]['motor_hz'],rows[0]['motor_hz']+1)
            self.assertLess(rows[2]['motor_hz'],rows[1]['motor_hz']*.25)
            self.assertGreater(rows[3]['motor_hz'],rows[2]['motor_hz']+1)

    def test_need_is_continuous_reaches_value_and_does_not_write_motors_or_dans(self):
        from veguinha_mini.network import build_network
        from veguinha_mini.sensory import inject_sensors
        start_scope();c=Config(scale=.1,backend='numpy',phase5_enabled=True,neural_value_channels=True)
        m=build_network(c,record_spikes=False);dn=m.regions['dn'];go,hold=np.array_split(dn[~m.inhibitory[dn]],2);m.intent=SimpleNamespace(go=go,hold=hold)
        sample=dict(angles=np.zeros((6,2)),contacts=[False]*6,touch=False,energy=1.,flow=[0,0,0],energy_cost_rate=0,antenna_odor=[0,0])
        inject_sensors(m,sample,config=c);full=np.asarray(m.group.I_sense[:]).copy()
        sample['energy']=.96;inject_sensors(m,sample,config=c);need=np.asarray(m.group.I_sense[:]).copy()
        self.assertGreater(need[go].mean(),12)
        self.assertLess(need[hold].mean(),2)
        motors=np.concatenate([np.concatenate(list(p.values())) for p in m.motor_pools.values()])
        np.testing.assert_array_equal(need[motors],full[motors]);np.testing.assert_array_equal(need[m.regions['dan']],0)
        self.assertGreater(need[m.regions['value']].mean(),full[m.regions['value']].mean())
        sample['antenna_odor']=[1,1];inject_sensors(m,sample,config=c);appetite=np.asarray(m.group.I_sense[:]).copy()
        self.assertGreater(appetite[go].mean(),need[go].mean())
        np.testing.assert_array_equal(appetite[motors],need[motors]);np.testing.assert_array_equal(appetite[m.regions['dan']],0)

if __name__=='__main__':unittest.main()
