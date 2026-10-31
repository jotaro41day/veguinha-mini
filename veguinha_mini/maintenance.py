"""Competing spiking drives for somatic maintenance, quiet vigilance and shelter."""
import numpy as np
from brian2 import NeuronGroup,Synapses,ms
from .network import EQUATIONS,EmittedSpikeMonitor
from .sensory import homeostatic_need


class MaintenanceCircuit:
    def __init__(self,config,model):
        self.config=config;self.model=model;clock=model.group.clock
        rng=np.random.default_rng(config.seed+81140)
        g=self.group=NeuronGroup(100,EQUATIONS,threshold='v>=30',reset='v=c;u+=d',
            events={'emit':'v>=30 and spike_gate>0'},method='euler',clock=clock,name='ecology_cells')
        self.exc=np.array([np.arange(20*k,20*k+16) for k in range(5)])
        self.inh=np.array([np.arange(20*k+16,20*k+20) for k in range(5)])
        g.a=.02;g.b=.2;g.c=-65;g.d=8;g.v=rng.uniform(-67,-63,100);g.u='b*v'
        g.I_base=-1;g.a[self.inh.ravel()]=.1;g.d[self.inh.ravel()]=2;g.spike_gate=1;g.tau_syn=20*ms
        pre=[];post=[];weights=[]
        def connect(src,dst,gain):
            if not len(src) or not len(dst):return
            pre.extend(np.repeat(src,len(dst)));post.extend(np.tile(dst,len(src)))
            weights.extend([gain/np.sqrt(len(src))]*(len(src)*len(dst)))
        for k in range(5):
            connect(self.exc[k],self.exc[k],2);connect(self.exc[k],self.inh[k],10)
            connect(self.inh[k],self.exc[k],-3)
            for j in range(5):
                if j!=k:connect(self.inh[k,2:],self.exc[j],-2)
        self.recurrent=Synapses(g,g,'w:1',on_pre='I_syn_post+=w',on_event='emit',clock=clock,name='ecology_recurrent')
        self.recurrent.connect(i=np.array(pre),j=np.array(post));self.recurrent.w=weights;self.recurrent.delay=2*ms
        pre.clear();post.clear();weights.clear()
        intent=model.intent
        for k in range(4):connect(self.inh[k,2:],intent.go,-8 if k!=2 else -12)
        connect(self.exc[2],intent.hold,3);connect(self.exc[3],model.regions['cx'][~model.inhibitory[model.regions['cx']]],2)
        for k,leg in enumerate(('lf','rf')):
            for joint in ('coxa','femur','tibia'):
                action='extend' if joint=='tibia' else 'flex'
                connect(self.exc[k],model.motor_pools[leg][joint+'_'+action],3 if joint=='coxa' else 1.5)
        for side,indices in enumerate((self.exc[4,:8],self.exc[4,8:])):
            connect(indices,intent.go,48)
            ipsi=[leg for leg,value in model.motor_side.items() if value==(1 if side==0 else -1)]
            contra=[leg for leg,value in model.motor_side.items() if value==(-1 if side==0 else 1)]
            for leg in ipsi:
                connect(indices,model.motor_pools[leg]['coxa_extend'],.8)
            for leg in contra:
                connect(indices,model.motor_pools[leg]['coxa_flex'],.8)
        self.output=Synapses(g,model.group,'w:1',on_pre='I_syn_post+=w',on_event='emit',clock=clock,name='ecology_to_core')
        self.output.connect(i=np.array(pre),j=np.array(post));self.output.w=weights;self.output.delay=2*ms
        src=np.r_[np.repeat(self.inh[0,2:],4),np.repeat(self.inh[1,2:],4)]
        dst=np.r_[np.tile(np.arange(4),2),np.tile(np.arange(12,16),2)]
        self.release=Synapses(g,intent.gates,'w:1',on_pre='I_syn_post=clip(I_syn_post+w,-60,40)',on_event='emit',clock=clock,name='groom_to_permission')
        self.release.connect(i=src,j=dst);self.release.w=-35/np.sqrt(2);self.release.delay=2*ms
        self.monitor=EmittedSpikeMonitor(g,'emit',record=False,name='ecology_spikes')
        self.objects=(g,self.recurrent,self.output,self.release,self.monitor)
        self.synapses=(self.recurrent,self.output,self.release);model.network.add(*self.objects)
        self.previous=np.zeros(100,np.int64);self.rates=np.zeros(5)

    def sense(self,body,state,sample):
        currents=np.zeros(100);need=homeostatic_need(body.energy,self.config.intent_hunger_span)
        if self.config.homeostasis_enabled:need=1-float(np.clip(body.energy,0,1))
        pain=float(np.clip(np.sum(sample.get('pain',[0,0])),0,1))
        dust=state.dust
        for side in range(2):
            local=float(np.mean(dust[[side*3,side*3+1,side*3+2,6,7]]))
            currents[self.exc[side]]=18*local
            currents[self.inh[side,:2]]+=8*need+15*pain
        currents[self.exc[2]]=12*state.fatigue+4*(1-float(state.light.mean()))
        comfort=1-state.fatigue**2 if self.config.homeostasis_enabled else 1.
        currents[self.inh[2,:2]]+=8*float(state.light.mean())*comfort+8*need+20*pain
        prey=np.asarray(sample.get('prey_odor',[0,0]),float)
        currents[self.exc[3]]=8*np.clip(prey.mean(),0,1)+2*(1-need)
        optical=sample.get('optical_flow')
        motion=0.
        if optical is not None:
            velocity=np.asarray(optical['angular_rad_s']);confidence=np.asarray(optical['confidence'])
            motion=float(np.max(np.linalg.norm(velocity,axis=-1)*confidence))
        currents[self.inh[3,:2]]+=12*np.clip(motion,0,2)+20*pain
        odor=np.asarray(sample['antenna_odor'],float)
        # Contralateral light and ipsilateral odor are receptor inputs, not yaw commands.
        for side in range(2):
            currents[self.exc[4,side*8:(side+1)*8]]=5*need*(1.8*prey[side]+.4*odor[side])+8*state.fatigue*state.light[1-side]
        self.group.I_sense=currents

    def read(self,dt):
        count=np.asarray(self.monitor.count[:]);delta=count-self.previous;self.previous=count.copy()
        observed=delta[self.exc].mean(axis=1)/dt
        self.rates+=(1-np.exp(-dt/.2))*(observed-self.rates)
        return self.rates.copy()
