import unittest
from types import SimpleNamespace
import numpy as np
from brian2 import NeuronGroup,SpikeGeneratorGroup,Synapses,Network,Clock,ms,start_scope,prefs
from veguinha_mini.config import Config
from veguinha_mini.network import EQUATIONS,EmittedSpikeMonitor


class MotorIntentTests(unittest.TestCase):
    def probe(self,go,hold,silent=False,seed=42,settle_ms=300,measure_ms=600,threat_hold=None,fast_brake=False):
        from veguinha_mini.intent import build_intent
        start_scope();prefs.codegen.target='numpy';clock=Clock(dt=.5*ms,name='intent_test_clock')
        g=NeuronGroup(140,EQUATIONS.replace(' + I_syn + I_noise',' + I_syn + I_cross + I_noise')+'\ndI_cross/dt=-I_cross/(40*ms):1',threshold='v>=30',reset='v=c;u+=d',events={'emit':'v>=30 and spike_gate>0'},method='euler',clock=clock,name='intent_test_cells')
        g.a=.02;g.b=.2;g.c=-65;g.d=8;g.v=-65;g.u=-13;g.I_base=0;g.spike_gate=1;g.tau_syn=5*ms
        inhibitory=np.zeros(140,bool);inhibitory[:24]=True;g.a[:24]=.1;g.d[:24]=2
        pools={leg:{'flex':np.array([128+2*k]),'extend':np.array([129+2*k])} for k,leg in enumerate(('lf','lm','lh','rf','rm','rh'))}
        monitor=EmittedSpikeMonitor(g,'emit',record=False,name='intent_test_monitor')
        model=SimpleNamespace(group=g,regions={'dn':np.arange(96),'oct':np.arange(96,128)},inhibitory=inhibitory,motor_pools=pools,network=Network(g,monitor))
        intent=build_intent(model,Config(seed=seed,somatic_enabled=fast_brake));g.I_sense[intent.go]=go;g.I_sense[intent.hold]=hold
        if silent:g.spike_gate[:96]=0
        times=np.array([.05,.25,.45,.65,.85,1.05,1.25,1.45,1.65,1.85])
        source=SpikeGeneratorGroup(1,np.zeros(len(times),int),times*1000*ms,clock=clock,name='intent_test_input')
        stimulus=Synapses(source,g,on_pre='I_syn_post+=32',clock=clock,name='intent_test_stim');stimulus.connect(i=0,j=np.arange(128,140))
        model.network.add(source,stimulus);model.network.run(settle_ms*ms);before=np.array(monitor.count[:]);model.network.run(measure_ms*ms)
        rates=(np.asarray(monitor.count[:])-before)/(measure_ms/1000)
        self.assertTrue(np.isfinite(np.asarray(g.v[:])).all())
        self.diagnostic=dict(go_i_hz=float(rates[intent.go_i].mean()),gate_hz=float(np.asarray(intent.monitor.count[:]).mean()/.9),gate_input=float(np.asarray(intent.gates.I_syn[:]).mean()),motor_inhibition=float(np.asarray(g.I_cross[128:]).mean()))
        result=(float(rates[128:].mean()),float(rates[intent.go].mean()),float(rates[intent.hold].mean()))
        print(dict(inputs=(go,hold,silent),rates=result,**self.diagnostic),flush=True)
        self.assertLess(self.diagnostic['gate_hz'],200)
        if threat_hold is not None:
            g.I_sense[intent.hold]=threat_hold;self.reflex_blocks=[]
            for block in range(6):
                before=np.asarray(monitor.count[:]).copy();gate_before=np.asarray(intent.monitor.count[:]).copy();model.network.run(100*ms)
                response=(np.asarray(monitor.count[:])-before)/.1
                self.reflex_blocks.append(dict(after_ms=100*(block+1),motor_hz=float(response[128:].mean()),inhibition=float(np.asarray(g.I_cross[128:]).mean()),gate_hz=float((np.asarray(intent.monitor.count[:])-gate_before).mean()/.1)))
        return result

    def test_motor_output_depends_on_descending_neural_competition(self):
        rest=self.probe(0,8);go=self.probe(14,1.6);eat=self.probe(14,30);silent=self.probe(14,1.6,True)
        print(dict(rest=rest,go=go,eat=eat,silent=silent),flush=True)
        self.assertGreater(go[0],rest[0]+1)
        self.assertGreater(go[1],go[2])
        self.assertGreater(eat[2],eat[1])
        self.assertLess(eat[0],go[0]*.25)
        self.assertEqual(silent[0],0)

    def test_sensory_context_changes_descending_currents_without_motor_commands(self):
        from veguinha_mini.network import build_network
        from veguinha_mini.sensory import inject_sensors
        start_scope();config=Config(scale=.05,backend='numpy',phase5_enabled=True,neural_value_channels=True)
        model=build_network(config,record_spikes=False)
        dn=model.regions['dn'];go,hold=np.array_split(dn[~model.inhibitory[dn]],2)
        model.intent=SimpleNamespace(go=go,hold=hold)
        sample=dict(angles=np.zeros((6,2)),contacts=[False]*6,touch=False,energy=1.,flow=[0,0,0],energy_cost_rate=0,antenna_odor=[0,0],mouth_contact=0)
        inject_sensors(model,sample,config=config);rest=np.array(model.group.I_sense[:])
        sample['energy']=.2;inject_sensors(model,sample,config=config);hungry=np.array(model.group.I_sense[:])
        sample['mouth_contact']=1;inject_sensors(model,sample,config=config);mouth=np.array(model.group.I_sense[:])
        self.assertGreater(hungry[go].mean(),rest[go].mean())
        self.assertGreater(mouth[hold].mean(),hungry[hold].mean())
        motors=np.concatenate([np.concatenate(list(p.values())) for p in model.motor_pools.values()])
        np.testing.assert_array_equal(mouth[motors],hungry[motors])
        np.testing.assert_array_equal(mouth[model.regions['dan']],0)


if __name__=='__main__':unittest.main()
