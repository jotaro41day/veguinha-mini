"""Physical resources and optional spiking animals; no behavioral scheduler."""
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from brian2 import NeuronGroup, Synapses, ms
from .network import EQUATIONS, EmittedSpikeMonitor


def load_model(path, prey_count=0,prey_radius_m=.025):
    if not prey_count:
        return mujoco.MjModel.from_xml_path(str(path))
    root=ET.parse(path).getroot()
    for element in root.findall('.//*[@file]'):
        element.set('file',str((path.parent/element.get('file')).resolve()))
    world=root.find('worldbody');actuators=root.find('actuator')
    for k in range(prey_count):
        parent=ET.SubElement(world,'body',name=f'prey{k}_segment0',pos=f'{-.25+.7*k} {.55-1.05*k} {prey_radius_m+.007}')
        ET.SubElement(parent,'freejoint',name=f'prey{k}_root')
        for segment in range(4):
            if segment:
                parent=ET.SubElement(parent,'body',name=f'prey{k}_segment{segment}',pos='.05 0 0')
                ET.SubElement(parent,'joint',name=f'prey{k}_joint{segment-1}',type='hinge',axis='0 0 1',
                    range='-.9 .9',damping='.00004',armature='.000001',stiffness='.0002',springref='0')
                ET.SubElement(actuators,'motor',name=f'prey{k}_motor{segment-1}',joint=f'prey{k}_joint{segment-1}',
                    gear='.002',ctrllimited='true',ctrlrange='-1 1')
            ET.SubElement(parent,'geom',name=f'prey{k}_collision{segment}',type='capsule',fromto='0 0 0 .05 0 0',
                size=str(prey_radius_m),mass='.003',friction='.65 .002 .0001',rgba='.4 .18 .14 1',group='3')
            ET.SubElement(parent,'geom',name=f'prey{k}_visual{segment}',type='capsule',fromto='0 0 0 .05 0 0',
                size=str(prey_radius_m+.0002),mass='0',contype='0',conaffinity='0',rgba=f'{.66-.04*segment} {.28-.02*segment} .22 1',group='2')
    return mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))


class EcologyState:
    """Continuous dust, fatigue and finite tissue. Only physics removes dust."""
    def __init__(self, config, body):
        self.config=config
        self.dust=np.zeros(8);self.cleaned=np.zeros(8);self.fatigue=0.
        self.shaken=np.zeros(8)
        self.dust_geoms=np.array([body.model.geom(f'{leg}_tibia_shaft').id for leg in ('lf','lm','lh','rf','rm','rh')]+[body.model.geom('shell').id,body.model.geom('head').id])
        self.previous_position=None;self.previous_velocity=None
        self.prey_energy=np.full(body.prey_count,config.prey_energy)
        self.prey_health=np.ones(body.prey_count)
        self.prey_consumed=0.;self.light=np.ones(2)
        self.previous_toes=self.relative_toes(body)
        self.prey_rates=np.zeros(body.prey_count)
        self.maintenance_rates=np.zeros(5)
        self.geom_leg={i:next((k for k,leg in enumerate(('lf','lm','lh','rf','rm','rh'))
                              if body.model.geom(i).name.startswith(leg+'_')),-1) for i in range(body.model.ngeom)}

    @staticmethod
    def relative_toes(body):
        return body.data.site_xpos[body.toe_ids]-body.data.xpos[body.body_id]

    def rub(self, site, speed, normal_force, dt):
        removal=min(float(self.dust[site]),self.config.dust_rub_gain*max(0.,speed)*max(0.,normal_force)*dt)
        self.dust[site]-=removal;self.cleaned[site]+=removal

    def shed(self,acceleration,dt):
        # Prototype adhesion model; no neural rate or grooming label enters this law.
        rate=self.config.dust_shed_gain*np.clip((np.asarray(acceleration)-self.config.dust_adhesion_accel)/self.config.dust_adhesion_accel,0,4)
        detached=self.dust*(-np.expm1(-rate*dt))
        self.dust-=detached;self.shaken+=detached

    def observe(self,body,dt):
        self.dust[:]=np.clip(self.dust+self.config.dust_deposition_rate*dt,0,1)
        if not hasattr(self,'shaken'):
            self.shaken=np.zeros(8)
            self.dust_geoms=np.array([body.model.geom(f'{leg}_tibia_shaft').id for leg in ('lf','lm','lh','rf','rm','rh')]+[body.model.geom('shell').id,body.model.geom('head').id])
        points=body.data.geom_xpos[self.dust_geoms].copy()
        previous=getattr(self,'previous_position',None)
        if previous is not None:
            velocity_now=(points-previous)/dt
            if self.previous_velocity is not None:
                self.shed(np.linalg.norm(velocity_now-self.previous_velocity,axis=1)/dt,dt)
            self.previous_velocity=velocity_now
        self.previous_position=points
        force=np.zeros(6);velocity=np.zeros((2,6))
        for index,c in enumerate(body.data.contact):
            for foot_geom,other in ((int(c.geom1),int(c.geom2)),(int(c.geom2),int(c.geom1))):
                leg=self.geom_leg.get(foot_geom,-1)
                if leg<0:continue
                target=self.geom_leg.get(other,-1)
                if target==leg:continue
                if target<0:
                    if other not in body.body_contact_ids:continue
                    target=7 if body.model.geom(other).name=='head' else 6
                mujoco.mj_contactForce(body.model,body.data,index,force)
                slips=[]
                for row,geom in enumerate((foot_geom,other)):
                    mujoco.mj_objectVelocity(body.model,body.data,mujoco.mjtObj.mjOBJ_GEOM,geom,velocity[row],0)
                    slips.append(velocity[row,3:]+np.cross(velocity[row,:3],c.pos-body.data.geom_xpos[geom]))
                slip=slips[0]-slips[1];normal=c.frame[:3]
                speed=float(np.linalg.norm(slip-normal*np.dot(slip,normal)))
                self.rub(target,speed,force[0],dt)
                self.rub(leg,speed,force[0],dt)

    def recover(self,dt,effort,quiet,light):
        self.fatigue=float(np.clip(self.fatigue+dt*(self.config.fatigue_gain*effort-
                           self.config.fatigue_recovery*quiet*(.2+.8*(1-light))),0,1))

    def update(self,body,dt):
        sense=getattr(body,'senses',None)
        if sense is not None and sense.sequence:
            self.light=np.clip(sense.rgb.mean((1,2,3))/255/.18,0,1)
        effort=float(np.mean(body.data.ctrl[body.motor_ids]**2))
        quiet=float(np.exp(-np.linalg.norm(body.velocity)/.03-np.mean(body.joint_velocity**2)))
        self.recover(dt,effort,quiet,float(self.light.mean()))
        self.prey_energy[:]=np.maximum(0,self.prey_energy-dt*(self.prey_health>0)*(.00002+.0001*np.mean(body.data.ctrl[body.prey_motor_ids]**2,axis=1)))

    def positions(self,body):
        return np.array([body.data.xpos[body.model.body(f'prey{k}_segment0').id] for k in range(body.prey_count)])

    def odor_at(self,body,points):
        delta=np.asarray(points)[...,None,:]-self.positions(body)[None,:,:2]
        return np.sum(np.exp(-np.sum(delta*delta,axis=-1)/(2*.65**2))*np.clip(self.prey_energy/self.config.prey_energy,0,1),axis=-1)

    def mouth_contact(self,body):
        from .perception import mouth_position
        rotation=body.data.xmat[body.body_id].reshape(3,3)
        centers=np.array([[body.data.geom_xpos[body.model.geom(f'prey{k}_collision{j}').id] for j in range(4)] for k in range(body.prey_count)])
        local=(centers-mouth_position(body))@rotation
        radius=np.array([body.model.geom_size[body.model.geom(f'prey{k}_collision0').id,0] for k in range(body.prey_count)])
        oral=np.array([.075,.095,.060])[None,None,:]+radius[:,None,None]
        contact=np.clip((1-np.min(np.linalg.norm(local/oral,axis=2),axis=1))/.3,0,1)
        return contact*(self.prey_energy>0)

    def feed(self,body,activation,dt):
        pressure=np.clip(activation,0,1)*dt*self.mouth_contact(body)
        self.prey_health[:]=np.maximum(0,self.prey_health-self.config.prey_bite_damage*pressure)
        demand=self.config.prey_bite_rate*pressure
        taken=np.minimum(self.prey_energy,demand)
        remaining=max(0.,1-body.energy)
        taken*=min(1.,remaining/max(float(taken.sum()),1e-12))
        self.prey_energy-=taken;total=float(taken.sum());body.energy+=total;self.prey_consumed+=total
        return total

    def feed_all(self,body,world,activation,dt):
        from .perception import mouth_contact
        pressure=float(np.clip(activation,0,1))*dt
        self.prey_health[:]=np.maximum(0,self.prey_health-self.config.prey_bite_damage*pressure*self.mouth_contact(body))
        ball=min(world.food_energy,self.config.feeding_rate*pressure*mouth_contact(body,world))
        tissue=np.minimum(self.prey_energy,self.config.prey_bite_rate*pressure*self.mouth_contact(body))
        scale=min(1.,max(0.,1-body.energy)/max(ball+float(tissue.sum()),1e-12))
        world,ball=world.consume(ball*scale);tissue*=scale
        self.prey_energy-=tissue;self.prey_consumed+=float(tissue.sum())
        total=ball+float(tissue.sum());body.energy+=total
        return world,total

    def snapshot(self,body):
        return dict(dust=self.dust.tolist(),cleaned=self.cleaned.tolist(),shaken=getattr(self,'shaken',np.zeros(8)).tolist(),fatigue=self.fatigue,
                    light=self.light.tolist(),maintenance_hz=self.maintenance_rates.tolist(),
                    prey=[dict(energy=float(self.prey_energy[k]),health=float(self.prey_health[k]),alive=bool(self.prey_health[k]>0 and self.prey_energy[k]>0),neural_hz=float(self.prey_rates[k]),neurons=400,
                        segments=[body.data.xpos[body.model.body(f'prey{k}_segment{j}').id].tolist() for j in range(4)])
                        for k in range(body.prey_count)],prey_consumed=self.prey_consumed)


class PreyCircuit:
    """400 Izhikevich cells/animal; sparse half-centers drive three muscles."""
    def __init__(self,config,clock,network,count):
        rng=np.random.default_rng(config.seed+72100);self.count=count
        eq=EQUATIONS.replace(' + I_syn + I_noise',' + I_syn + I_slow + I_noise')+'\ndI_slow/dt=-I_slow/(150*ms):1\n'
        self.group=g=NeuronGroup(400*count,eq,threshold='v>=30',reset='v=c;u+=d',
            events={'emit':'v>=30 and spike_gate>0'},method='euler',clock=clock,name='prey_cells')
        g.a=.02;g.b=.2;g.c=-65;g.d=8;g.v=rng.uniform(-67,-63,len(g));g.u='b*v'
        g.spike_gate=1;g.tau_syn=10*ms;g.noise=config.noise;g.noise_gain=.3
        self.inhibitory=np.zeros(len(g),bool);self.motors=np.zeros((count,3,2,12),int)
        pre=[];post=[];w=[];slow=[]
        def connect(src,dst,gain,degree=6,is_slow=False):
            for target in dst:
                source=np.asarray(src);source=source[source!=target];n=min(degree,len(source))
                chosen=rng.choice(source,n,replace=False)
                pre.extend(chosen);post.extend([target]*n);w.extend([gain/np.sqrt(n)]*n);slow.extend([int(is_slow)]*n)
        for animal in range(count):
            offset=400*animal
            for j in range(3):
                start=offset+96*j
                motor=np.arange(start,start+24).reshape(2,12);exc=np.arange(start+24,start+72).reshape(2,24);inh=np.arange(start+72,start+96).reshape(2,12)
                self.motors[animal,j]=motor;self.inhibitory[inh.ravel()]=True
                g.a[exc.ravel()]=rng.uniform(.0048,.0058,48);g.d[exc.ravel()]=10;g.I_base[exc.ravel()]=5.3
                g.I_base[motor.ravel()]=1.5;g.noise_gain[motor.ravel()]=0
                for action in range(2):
                    connect(exc[action],exc[action],5);connect(exc[action],inh[action],8)
                    connect(inh[action],exc[1-action],-4,is_slow=True)
                    connect(exc[action],motor[action],12);connect(inh[action],motor[1-action],-3)
                    sensory=np.arange(offset+288+(2*j+action)*8,offset+288+(2*j+action+1)*8)
                    connect(sensory,exc[action],2,2)
                    if j<2:connect(exc[action],np.arange(start+120+24*action,start+144+24*action),.8,2)
                connect(np.arange(offset+384,offset+392),exc.ravel(),1.,2)
            self.inhibitory[offset+392:offset+400]=True
            g.I_base[offset+384:offset+392]=4.5
            connect(np.arange(offset+384,offset+392),np.arange(offset+392,offset+400),3,4)
            connect(np.arange(offset+392,offset+400),np.arange(offset+384,offset+392),-2,4)
        g.a[self.inhibitory]=.1;g.d[self.inhibitory]=2
        g.run_regularly('I_noise=noise*noise_gain*randn()',clock=clock,name='prey_noise')
        self.synapses=Synapses(g,g,'w:1\nslow:integer (constant)',on_pre='I_syn_post+=w*(1-slow);I_slow_post+=w*slow',on_event='emit',clock=clock,name='prey_synapses')
        self.synapses.connect(i=np.array(pre),j=np.array(post));self.synapses.w=w;self.synapses.slow=slow;self.synapses.delay=2*ms
        self.monitor=EmittedSpikeMonitor(g,'emit',record=False,name='prey_spikes')
        self.objects=(g,self.synapses,self.monitor);network.add(*self.objects)
        self.previous=np.zeros(len(g),np.int64);self.rates=np.zeros((count,3,2));self.mean_hz=np.zeros(count)
        self.enabled=True

    def sense(self,body,state):
        currents=np.zeros(len(self.group));alive=(state.prey_energy>0)&(state.prey_health>0)
        for k in range(self.count):
            for j in range(3):
                angle=float(body.data.qpos[int(body.model.joint(f'prey{k}_joint{j}').qposadr[0])])
                for action,sign in enumerate((-1,1)):
                    start=k*400+288+(2*j+action)*8
                    currents[start:start+8]=6*np.clip(sign*angle/.5,0,2)
            currents[k*400+384:k*400+392]=2*np.clip(state.prey_energy[k]/state.config.prey_energy,0,1)
        self.group.I_sense=currents
        self.group.spike_gate=np.repeat(alive,400).astype(float)*float(self.enabled)

    def decode(self,dt):
        counts=np.asarray(self.monitor.count[:]);delta=counts-self.previous;self.previous=counts.copy()
        hz=delta[self.motors].mean(axis=-1)/dt
        self.rates+=(1-np.exp(-dt/.05))*(hz-self.rates)
        self.mean_hz=delta.reshape(self.count,400).mean(1)/dt
        return np.clip(.06*(self.rates[:,:,0]-self.rates[:,:,1]),-1,1)


def migrate_body(saved,current,config):
    """Append animals without resetting the saved hexapod's physical state."""
    old_radius=(float(saved['model'].geom_size[saved['model'].geom('prey0_collision0').id,0])
                if saved.get('prey_count',0) else None)
    if saved['model'].nq==current.model.nq and old_radius==config.prey_radius_m:
        return saved
    old=saved['data'];new=current.data
    if old.qpos.size>new.qpos.size:raise ValueError('não é possível remover corpos deste checkpoint')
    new.qpos[:old.qpos.size]=old.qpos;new.qvel[:old.qvel.size]=old.qvel
    new.ctrl[:old.ctrl.size]=old.ctrl;new.qacc_warmstart[:old.qacc_warmstart.size]=old.qacc_warmstart
    if old_radius is not None:
        for k in range(saved['prey_count']):
            adr=int(current.model.joint(f'prey{k}_root').qposadr[0]);new.qpos[adr+2]+=config.prey_radius_m-old_radius
    new.time=old.time;mujoco.mj_forward(current.model,new)
    result=dict(saved,model=current.model,data=new,prey_count=current.prey_count,
                motor_ids=current.motor_ids,prey_motor_ids=current.prey_motor_ids,
                prey_radius_m=config.prey_radius_m,
                prey_geom_ids=current.prey_geom_ids,
                ecology_state=saved.get('ecology_state') or EcologyState(config,current))
    if old_radius!=config.prey_radius_m:
        result['ecology_state'].previous_position=None;result['ecology_state'].previous_velocity=None
    return result
