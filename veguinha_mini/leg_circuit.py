"""Local candidate: reference half-centers, proprioceptors and foot-load afferents.

No phase, trajectory, gait state or velocity command is generated here.
This candidate is not enabled in the main application's network.
"""
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from brian2 import Clock, Network, NeuronGroup, Synapses, ms, prefs, seed as brian_seed
from .network import EQUATIONS, EmittedSpikeMonitor
from .config import Config
from .mujoco_body import MujocoBody


LEG_EQUATIONS=EQUATIONS.replace(' + I_syn + I_noise',' + I_syn + I_slow + I_noise')+'\ndI_slow/dt = -I_slow/(150*ms) : 1\nv_threshold:1\n'

def build_leg(seed=42,joints=3,coupling=0,dt_ms=.5,backend='cython',motor_strength=4,position_gate=False,gate_memory=False,coherent_reset=False,shared_inhibition=False,reset_inhibition=False,load_brake=False,motor_heterogeneity=False,motor_size=8,sensory_output=False,motor_excitability=False):
    if joints not in (2,3) or dt_ms not in (.25,.5,1) or coupling<0 or (load_brake and joints!=3) or motor_size not in (8,16,32):
        raise ValueError('invalid local diagnostic configuration')
    prefs.codegen.target=backend;brian_seed(seed);rng=np.random.default_rng(seed)
    names=('coxa','femur','tibia') if joints==3 else ('femur','tibia')
    motors=np.arange(16*joints).reshape(joints,2,8)
    premotor=np.arange(16*joints,40*joints).reshape(joints,2,12)
    inhibitory=np.arange(40*joints,60*joints).reshape(joints,2,10)
    position=np.arange(60*joints,64*joints).reshape(joints,2,2)
    load=np.arange(64*joints,64*joints+4).reshape(2,2)
    brake=np.arange(64*joints+4,64*joints+12) if load_brake else np.array([],dtype=int)
    extra=np.arange(64*joints+4+len(brake),64*joints+4+len(brake)+2*joints*(motor_size-8))
    if motor_size>8:motors=np.concatenate((motors,extra.reshape(joints,2,motor_size-8)),axis=2)
    equations=LEG_EQUATIONS
    group=NeuronGroup(64*joints+4+len(brake)+len(extra),equations,threshold='v>=v_threshold',reset='v=c; u+=d',
                      events={'emit':'v>=v_threshold and spike_gate>0'},name='leg_cells',
                      method='euler',clock=Clock(dt=dt_ms*ms,name='leg_clock'))
    group.a=.02;group.b=.2;group.c=-65;group.d=8;group.v_threshold=30
    if motor_excitability:group.b[motors.ravel()]=np.tile(np.linspace(.18,.22,motor_size),2*joints)
    if motor_heterogeneity:group.d[motors.ravel()]=np.tile(np.linspace(6,10,motor_size),2*joints)
    group.a[premotor.ravel()]=.005423742;group.d[premotor.ravel()]=11.211611
    group.a[np.r_[inhibitory.ravel(),brake]]=.1;group.d[np.r_[inhibitory.ravel(),brake]]=2
    group.v=-65+rng.normal(0,2,len(group));group.u='b*v'
    group.I_base=0;group.I_base[premotor.ravel()]=5.346912
    group.I_base[np.r_[inhibitory.ravel(),brake]]=.371140
    group.I_base[np.r_[position.ravel(),load.ravel()]]=1
    if position_gate:group.I_base[position[0].ravel()]=-2
    group.noise=2.5;group.noise_gain=1
    group.noise_gain[np.r_[position.ravel(),load.ravel(),brake]]=0
    group.spike_gate=1;group.tau_syn=5*ms
    # Keep the physical noise bandwidth fixed when comparing integration dt.
    group.run_regularly('I_noise=noise*noise_gain*randn()',dt=1*ms,name='leg_noise')
    pre=[];post=[];weight=[];slow=[];kinds=[]

    def connect(src,dst,gain,degree,sign=1,kind='local',is_slow=False,no_self=False):
        if gain==0:return
        for i in src:
            for j in dst:
                if no_self and i==j:continue
                pre.append(i);post.append(j);weight.append(sign*gain/np.sqrt(degree))
                slow.append(int(is_slow));kinds.append(kind)

    for joint in range(joints):
        for action in range(2):
            e=premotor[joint,action];i=inhibitory[joint,action];other=1-action
            connect(e,e,8.431984,23,no_self=True)
            connect(e,i,9.949371,12)
            connect(e,motors[joint,action],motor_strength,22)
            connect(i,premotor[joint,other],1.455507,23,-1,is_slow=True)
            connect(i,motors[joint,other],2,22,-1)
            connect(position[joint,action],motors[joint,action] if sensory_output else e,8 if coherent_reset and joint==0 else .795704,
                    23,kind='stretch')
    for first in range(joints-1):
        for a,b in ((first,first+1),(first+1,first)):
            for action in range(2):
                connect(premotor[a,action],premotor[b,action],coupling,23,kind='intraleg')
                if shared_inhibition:
                    neighbors=1 if b in (0,joints-1) else 2
                    connect(inhibitory[a,action],premotor[b,1-action],1.455507/neighbors,
                            23,-1,kind='interjoint_inhibition',is_slow=True)
    # Graded back/front stretch afferents can recruit lift/lower synergy.
    if joints==3:
        if position_gate:
            # A stretch comparator also receives the ongoing motor copy.
            for action in range(2):
                connect(premotor[0,action],position[0,1-action],.6 if gate_memory else 2,
                        23,kind='efference_gate',is_slow=gate_memory)
                connect(inhibitory[0,action],position[0,action],2,23,-1,kind='efference_gate')
        for action in range(2):
            for distal in (1,2):
                connect(position[0,action],motors[distal,action] if sensory_output else premotor[distal,action],8,23,kind='end_coxa')
        if reset_inhibition:
            # End-stretch recruits the next half-center's inhibitory relay too.
            for action in range(2):
                for joint in range(joints):
                    connect(position[0,action],inhibitory[joint,action],9.949371,12,
                            kind='sensory_reset_inhibition')
        if load_brake:
            # Positive distal torque lifts the foot; inhibit its opposing motor pool on overload.
            for j in (1,2):
                connect(brake,motors[j,1],motor_strength,22,-1,kind='load_brake')
        else:
            for j in range(joints):
                connect(load[0],motors[j,1] if sensory_output else premotor[j,1],8 if j==0 else 4,23,kind='loaded')
                connect(load[1],motors[j,0] if sensory_output else premotor[j,0],8 if j==0 else 4,23,kind='unloaded')
    syn=Synapses(group,group,model='w:1\nslow:integer (constant)',
                 on_pre='I_syn_post+=w*(1-slow); I_slow_post+=w*slow',
                 on_event='emit',clock=group.clock,name='leg_synapses')
    syn.connect(i=pre,j=post);syn.w=weight;syn.slow=slow;syn.delay=2*ms
    spikes=EmittedSpikeMonitor(group,'emit',record=False,name='leg_spikes')
    c=Config(body_model='mk1')
    config=SimpleNamespace(joint_names=names,motor_filter_ms=c.motor_filter_ms,
        motor_baseline_ms=c.motor_baseline_ms,motor_gain=c.motor_gain,motor_limit=c.motor_limit)
    p=SimpleNamespace(group=group,network=Network(group,syn,spikes),monitor=spikes,
        synapses=syn,motors=motors,premotor=premotor,inhibitory=inhibitory,position=position,
        load=load,load_brake=brake,config=config,ms=ms,kinds=np.array(kinds),
        motor_pools={'lf':{f'{name}_{action}':motors[j,a]
                        for j,name in enumerate(names) for a,action in enumerate(('flex','extend'))}})
    return p


def sense(p,angles,normal_load,position_gain=0,load_gain=0):
    """Physical positions/load -> receptor currents only; spikes drive muscles."""
    q=np.asarray(angles)[-len(p.config.joint_names):]
    currents=np.zeros(len(p.group))
    currents[p.position[:,0].ravel()]=np.repeat(position_gain*np.clip(-q/.1,0,2),2)
    currents[p.position[:,1].ravel()]=np.repeat(position_gain*np.clip(q/.1,0,2),2)
    force=np.clip(normal_load/(.316*9.81/6),0,1)
    currents[p.load[0]]=load_gain*force
    currents[p.load[1]]=load_gain*(1-force)
    currents[p.load_brake]=load_gain*np.clip(normal_load/(.316*9.81/6)-1,0,5)
    p.group.I_sense=currents


class FixedTrunk:
    """Settled mk-1, fixed height/orientation; optional XY guide for LF stance."""
    def __init__(self,dt=.002,horizontal_free=False,walls=True,compliant=False):
        if compliant and not horizontal_free:raise ValueError('Compliant support requires the horizontal guide')
        passive=MujocoBody(variant='mk1',dt=dt)
        for _ in range(100):passive.step(.02,np.zeros((6,3)))
        self.support_constants=None
        if compliant:
            stiffness=0.;damping=0.
            for leg in ('lm','lh','rf','rm','rh'):
                jac=np.zeros((3,passive.model.nv))
                mujoco.mj_jacSite(passive.model,passive.data,jac,None,passive.model.site(leg+'_toe').id)
                ids=np.array([passive.model.joint(leg+'_'+j).id for j in ('coxa','femur','tibia')])
                dofs=passive.model.jnt_dofadr[ids]
                leverage=jac[2,dofs];spring=passive.model.jnt_stiffness[ids]
                k=1/np.sum(leverage**2/spring)
                stiffness+=k
                damping+=k*k*np.sum(leverage**2*passive.model.dof_damping[dofs]/spring**2)
            preload=float(passive.normal_load[1:].sum())
            self.support_constants=dict(stiffness=stiffness,damping=damping,preload_n=preload)
        xml=ET.parse(passive.mjcf_path).getroot();thorax=xml.find(".//body[@name='thorax']")
        thorax.remove(thorax.find('freejoint'))
        thorax.set('pos',' '.join(map(str,passive.data.qpos[:3])))
        thorax.set('quat',' '.join(map(str,passive.data.qpos[3:7])))
        if not walls:
            for geom in xml.findall('./worldbody/geom'):
                if geom.get('name','').startswith('wall_'):
                    geom.set('contype','0');geom.set('conaffinity','0')
        if horizontal_free:
            for name,axis in (('bench_x','1 0 0'),('bench_y','0 1 0')):
                ET.SubElement(thorax,'joint',name=name,type='slide',axis=axis,
                              limited='false',damping='0',armature='0')
            if compliant:
                # Conservative spring approximates the five omitted legs; no gait input.
                ET.SubElement(thorax,'joint',name='bench_z',type='slide',axis='0 0 1',limited='false',
                    stiffness=str(stiffness),damping=str(damping),springref=str(preload/stiffness))
            # The other five passive feet must not anchor the one-leg fixture.
            for child in thorax.findall('body'):
                if child.get('name')!='lf_coxa':
                    for geom in child.iter('geom'):
                        geom.set('contype','0');geom.set('conaffinity','0')
        for asset in xml.findall('./asset/*'):
            if asset.get('file'):asset.set('file',str(passive.mjcf_path.parent/asset.get('file')))
        self.model=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'))
        self.model.opt.timestep=dt;self.data=mujoco.MjData(self.model)
        for j in range(self.model.njnt):
            name=self.model.joint(j).name
            if name in ('bench_x','bench_y','bench_z'):continue
            self.data.qpos[self.model.jnt_qposadr[j]]=passive.data.qpos[passive.model.joint(name).qposadr][0]
        self.root=self.model.body('thorax').id
        self.wall_ids={self.model.geom('wall_'+s).id for s in ('front','back','left','right')}
        self.qpos=np.array([self.model.joint('lf_'+j).qposadr[0] for j in ('coxa','femur','tibia')])
        self.actuators=np.array([self.model.actuator('lf_'+j+'_motor').id for j in ('coxa','femur','tibia')])
        self.foot=self.model.geom('lf_foot').id;self.floor=self.model.geom('floor').id
        self.toe=self.model.site('lf_toe').id
        self.neutral=self.data.qpos[self.qpos].copy()
        self.load=0.;mujoco.mj_forward(self.model,self.data)
        self.step(np.zeros(3))

    @property
    def angles(self):return self.data.qpos[self.qpos].copy()

    def step(self,torque,observe=None):
        self.data.ctrl[:]=0;self.data.ctrl[self.actuators]=np.clip(torque,-1,1)
        support=[];loads=[];force=np.zeros(6)
        for _ in range(round(.02/self.model.opt.timestep)):
            mujoco.mj_step(self.model,self.data);load=0.
            for k,c in enumerate(self.data.contact):
                if {int(c.geom1),int(c.geom2)}=={self.foot,self.floor}:
                    mujoco.mj_contactForce(self.model,self.data,k,force);load+=max(0.,force[0])
            loads.append(load);support.append(load>.02)
            if observe is not None:observe(self,load)
        mujoco.mj_forward(self.model,self.data)
        self.load=float(np.mean(loads))
        clearance=float(self.data.site_xpos[self.toe,2]-self.model.geom_size[self.foot,0])
        return float(np.mean(support)),clearance
