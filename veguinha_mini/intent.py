"""Spiking descending competition and tonic VNC inhibition; no action state machine."""
from types import SimpleNamespace
import numpy as np
from brian2 import NeuronGroup,Synapses,ms
from .network import EQUATIONS,EmittedSpikeMonitor


def prepare_intent(model,intent):
    g=model.group;dn=model.regions['dn']
    g.I_base[dn]=-1.5;g.I_base[np.r_[intent.go_i,intent.hold_i]]=-2
    g.noise_gain[dn]=.25;g.v[dn]=-65;g.u[dn]=-65*np.asarray(g.b[dn]);g.I_syn[dn]=0;g.I_cross[dn]=0
    g.I_cross[intent.motors]=-60
    # Explicit initialization must not inherit the previous disinhibition transient.
    intent.gates.I_syn=0;intent.gates.v=-65;intent.gates.u='b*v'


def build_intent(model,config):
    core=model.group;dn=model.regions['dn']
    go,hold=np.array_split(dn[~model.inhibitory[dn]],2)
    go_i,hold_i=np.array_split(dn[model.inhibitory[dn]],2)
    gates=NeuronGroup(24,EQUATIONS,threshold='v>=30',reset='v=c;u+=d',
        events={'emit':'v>=30 and spike_gate>0'},method='euler',clock=core.clock,name='permission_cells')
    rng=np.random.default_rng(config.seed+9831)
    gates.a=.1*rng.uniform(.95,1.05,24);gates.b=.2;gates.c=-65;gates.d=2
    # Slow disinhibition bridges burst gaps; no phase or torque is prescribed.
    gates.v=rng.uniform(-65,-55,24);gates.u='b*v';gates.I_base=8;gates.spike_gate=1;gates.tau_syn=100*ms
    pre=[];post=[];weights=[];slow=[]
    def connect(src,dst,strength,delay_current=False):
        pre.extend(np.repeat(src,len(dst)));post.extend(np.tile(dst,len(src)))
        weights.extend([strength/np.sqrt(len(src))]*(len(src)*len(dst)));slow.extend([int(delay_current)]*(len(src)*len(dst)))
    for exc,inh,other in ((go,go_i,hold),(hold,hold_i,go)):
        connect(exc,exc,7);connect(exc,inh,14);connect(inh,exc,-8);connect(inh,other,-3,True)
    if 'oct' in model.regions:connect(model.regions['oct'],go,3)
    recurrent=Synapses(core,core,'w:1\nslow:integer (constant)',
        on_pre='I_syn_post+=w*(1-slow); I_cross_post=clip(I_cross_post+w*slow,-80,80)',
        on_event='emit',clock=core.clock,name='intent_recurrence')
    recurrent.connect(i=np.array(pre,int),j=np.array(post,int));recurrent.w=weights;recurrent.slow=slow;recurrent.delay=2*ms
    descending=Synapses(core,gates,'w:1',on_pre='I_syn_post=clip(I_syn_post+w,-60,40)',
        on_event='emit',clock=core.clock,name='intent_to_gate')
    descending.connect(i=np.repeat(go_i,24),j=np.tile(np.arange(24),len(go_i)))
    descending.w=-45/np.sqrt(len(go_i));descending.delay=2*ms
    inhibition=Synapses(gates,core,'w:1',on_pre='I_cross_post=clip(I_cross_post+w,-80,0)',
        on_event='emit',clock=core.clock,name='gate_to_motor')
    src=[];dst=[]
    for k,pools in enumerate(model.motor_pools.values()):
        motors=np.concatenate(list(pools.values()));src.extend(np.repeat(np.arange(4*k,4*k+4),len(motors)));dst.extend(np.tile(motors,4))
    inhibition.connect(i=np.array(src,int),j=np.array(dst,int));inhibition.w=-config.motor_gate_strength/2;inhibition.delay=2*ms
    monitor=EmittedSpikeMonitor(gates,'emit',record=False,name='permission_spikes')
    objects=(gates,recurrent,descending,inhibition,monitor)
    brake=None
    if config.somatic_enabled:
        # Suppression can recruit inhibition before old GO disinhibition decays.
        brake=Synapses(core,gates,'w:1',on_pre='I_syn_post+=w*clip(-I_syn_post,0,60)/60',
            on_event='emit',clock=core.clock,name='hold_to_permission')
        brake.connect(i=np.repeat(hold,24),j=np.tile(np.arange(24),len(hold)))
        brake.w=8/np.sqrt(len(hold));brake.delay=1*ms
        objects+=(brake,)
    model.network.add(*objects)
    intent=SimpleNamespace(go=go,hold=hold,go_i=go_i,hold_i=hold_i,gates=gates,monitor=monitor,
        motors=np.unique(dst),objects=objects,synapses=(recurrent,descending,inhibition)+((brake,) if brake is not None else ()),somatic_brake=brake,
        last_core=np.zeros(len(core),np.int64),last_gate=np.zeros(24,np.int64),rates=np.zeros(3))
    prepare_intent(model,intent);model.intent=intent
    return intent


def update_rates(intent,count,dt):
    delta=np.asarray(count)-intent.last_core;gate=np.asarray(intent.monitor.count[:])
    observed=[delta[intent.go].mean()/dt,delta[intent.hold].mean()/dt,(gate-intent.last_gate).mean()/dt]
    intent.rates+=(1-np.exp(-dt/.2))*(np.asarray(observed)-intent.rates)
    intent.last_core=np.asarray(count).copy();intent.last_gate=gate.copy()


def status(intent,energy,config):
    if intent is None:return None
    from .sensory import homeostatic_need
    return dict(zip(('go_hz','hold_hz','gate_hz'),intent.rates.tolist()),
                need=homeostatic_need(energy,config.intent_hunger_span))
