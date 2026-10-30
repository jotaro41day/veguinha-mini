"""Six existing leg modules fused into one Brian2 group; no gait or velocity controller."""
from types import SimpleNamespace
import numpy as np
from brian2 import NeuronGroup,Synapses,Network,Clock,Equations,ms,prefs,seed as brian_seed
from .leg_circuit import build_leg,LEG_EQUATIONS
from .network import EmittedSpikeMonitor
from .body import LEG_NAMES
from .isolated import COUPLINGS


def build_cord(seed=42,cross_gain=0,backend='cython',warm=True):
 if cross_gain<0:raise ValueError('negative coupling')
 fields=('a','b','c','d','v','u','I_base','I_syn','I_slow','I_noise','I_sense','I_odor','noise','noise_gain','spike_gate','v_threshold')
 rng=np.random.default_rng(seed+8100);states=[];pres=[];posts=[];weights=[];slow=[]
 durations=rng.uniform(.15,1.2,6) if warm else np.zeros(6)
 for k in range(6):
  p=build_leg(seed+1009*k,coupling=2,motor_strength=12,backend='numpy',sensory_output=True,motor_excitability=True)
  # Validated motor parameters; no diagnostic stimulus or burst source is used.
  p.group.noise[p.motors.ravel()]=0
  p.group.I_base[p.motors.ravel()]=2
  p.group.v_threshold[p.motors.ravel()]=np.tile(30+5*np.linspace(-1,1,8),6)
  if warm:p.network.run(round(durations[k]*2000)*.5*ms,namespace={})
  states.append({f:np.asarray(getattr(p.group,f)[:]).copy() for f in fields})
  pres.append(np.asarray(p.synapses.i[:],int)+196*k);posts.append(np.asarray(p.synapses.j[:],int)+196*k)
  weights.append(np.asarray(p.synapses.w[:]).copy());slow.append(np.asarray(p.synapses.slow[:],int).copy())
  template=p
 prefs.codegen.target=backend;brian_seed(seed)
 equations=LEG_EQUATIONS.replace(' + I_slow + I_noise',' + I_slow + I_cross + I_noise')+'\ndI_cross/dt=-I_cross/(40*ms):1'
 g=NeuronGroup(1176,equations,threshold='v>=v_threshold',reset='v=c;u+=d',events={'emit':'v>=v_threshold and spike_gate>0'},method='euler',clock=Clock(dt=.5*ms,name='cord_clock'),name='cord_cells')
 for f in fields:setattr(g,f,np.concatenate([s[f] for s in states]))
 g.tau_syn=5*ms;g.run_regularly('I_noise=noise*noise_gain*randn()',dt=1*ms,name='cord_noise')
 motors=np.array([template.motors+196*k for k in range(6)]);premotor=np.array([template.premotor+196*k for k in range(6)])
 inhibitory=np.array([template.inhibitory+196*k for k in range(6)])
 cross=[np.zeros(len(w),int) for w in weights]
 degree={leg:sum(leg in pair for pair in COUPLINGS) for leg in LEG_NAMES}
 for l,r in COUPLINGS:
  for source,target in ((l,r),(r,l)):
   a=LEG_NAMES.index(source);b=LEG_NAMES.index(target)
   for j in range(3):
    for action in range(2):
     pres.append(np.repeat(inhibitory[a,j,action],12));posts.append(np.tile(premotor[b,j,action],10))
     weights.append(np.full(120,-cross_gain/(np.sqrt(10)*degree[target])));slow.append(np.zeros(120,int));cross.append(np.ones(120,int))
 syn=Synapses(g,g,'w:1\nslow:integer (constant)\ncross:integer (constant)',on_pre='I_syn_post+=w*(1-slow)*(1-cross); I_slow_post+=w*slow; I_cross_post+=w*cross',on_event='emit',clock=g.clock,name='cord_synapses')
 syn.connect(i=np.concatenate(pres),j=np.concatenate(posts));syn.w=np.concatenate(weights);syn.slow=np.concatenate(slow);syn.cross=np.concatenate(cross);syn.delay=2*ms
 monitor=EmittedSpikeMonitor(g,'emit',record=False,name='cord_spikes')
 pools={leg:{name:motors[k,j,a] for j,joint in enumerate(template.config.joint_names) for a,action in enumerate(('flex','extend')) for name in (joint+'_'+action,)} for k,leg in enumerate(LEG_NAMES)}
 return SimpleNamespace(group=g,synapses=syn,monitor=monitor,network=Network(g,syn,monitor),motor_pools=pools,config=template.config,motors=motors,premotor=premotor,inhibitory=inhibitory,positions=np.array([template.position+196*k for k in range(6)]),loads=np.array([template.load+196*k for k in range(6)]),edge_kind=np.where(np.concatenate(cross),2,np.concatenate(slow)),warm_durations_s=durations.tolist())


def cord_currents(p,angles,load,position_gain=4,load_gain=12):
 currents=np.zeros(len(p.group));q=np.asarray(angles)
 currents[p.positions[:,:,0,:].ravel()]=np.repeat(position_gain*np.clip(-q/.1,0,2).ravel(),2)
 currents[p.positions[:,:,1,:].ravel()]=np.repeat(position_gain*np.clip(q/.1,0,2).ravel(),2)
 f=np.clip(np.asarray(load)/(.316*9.81/6),0,1)
 currents[p.loads[:,0,:].ravel()]=np.repeat(load_gain*f,2)
 currents[p.loads[:,1,:].ravel()]=np.repeat(load_gain*(1-f),2)
 return currents

def sense_cord(p,angles,load,position_gain=4,load_gain=12):
 p.group.I_sense=cord_currents(p,angles,load,position_gain,load_gain)
