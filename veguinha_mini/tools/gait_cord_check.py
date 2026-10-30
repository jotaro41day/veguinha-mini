"""Closed-loop locomotion diagnostics using the existing neural leg modules."""
import argparse,json,time,os
from pathlib import Path
import numpy as np,mujoco
from brian2 import start_scope,NetworkOperation,Clock,ms
from veguinha_mini.cord import build_cord,sense_cord
from veguinha_mini.motor import MotorDecoder
from veguinha_mini.mujoco_body import MujocoBody
from veguinha_mini.mk1_validation import atomic
from veguinha_mini.tools.leg_reflex_check import joint_metrics
from veguinha_mini.coordination_diagnostic import _phase_metrics,PAIRS
OUT=Path(__file__).resolve().parents[1]/'results/gait_loop_v1'
LIVE=OUT.parent/'leg_reflex_v1'

def run(seed=42,cross_gain=0,seconds=20,neural_only=False,feedback=True,shuffled=False,average_feedback=False):
 name=f'v2_cord_{seed}_k{cross_gain:g}_{"neural" if neural_only else "body"}_{"feedback" if feedback else "open"}_{"shuffled" if shuffled else "normal"}'
 if average_feedback:name+='_averaged'
 f=OUT/(name+'.json')
 if f.exists():return json.loads(f.read_text())
 start_scope();started=time.perf_counter()
 atomic(LIVE/'task.json',dict(title='Investigando coordenação e propulsão do corpo livre',detail='Seis módulos neurais existentes; torques por spikes; física2ms. Sem marcha prescrita nem aprendizado.'))
 def progress(state,t=0):atomic(LIVE/'progress.json',dict(pid=os.getpid(),state=state,profile=name,seed=seed,simulated_s=t,updated=time.time()))
 progress('preparando');p=build_cord(seed,cross_gain=cross_gain)
 body=MujocoBody(variant='mk1')
 for _ in range(100):body.step(.02,np.zeros((6,3)))
 neutral=body.angles.copy();body.energy=1;initial=body.position.copy()
 decoder=MotorDecoder(p,p.config);data=[];counts0=np.zeros(1176,int);force=np.zeros(6);subloads=[]
 perm=np.arange(6)
 if shuffled:
  rng=np.random.default_rng(seed+30000)
  while np.any(perm==np.arange(6)):perm=rng.permutation(6)
 def observe(b):
  loads=np.zeros(6)
  for n,c in enumerate(b.data.contact):
   a,d=int(c.geom1),int(c.geom2)
   foot=b.foot_index.get(d if a==b.floor_id else a if d==b.floor_id else -1)
   if foot is not None:
    mujoco.mj_contactForce(b.model,b.data,n,force);loads[foot]+=max(0.,force[0])
  subloads.append(loads)
 def tick():
  nonlocal counts0,subloads
  count=np.asarray(p.monitor.count[:],int);torque=decoder.update(count,.02)[perm].copy();subloads=[]
  if not neural_only:body.step(.02,torque,observe=observe)
  else:subloads=[np.zeros(6)]*10
  loads=np.array(subloads);support=(loads>.02).mean(0)
  if feedback and not neural_only:sense_cord(p,body.angles-neutral,np.clip(loads/(.316*9.81/6),0,1).mean(0)*(.316*9.81/6) if average_feedback else body.normal_load)
  r=body.data.xmat[body.body_id].reshape(3,3)
  data.append(dict(counts=count-counts0,torque=torque,rates=decoder.filtered_hz.copy(),angles=body.angles.copy(),support=support,subsupport=loads>.02,load=loads.mean(0),load_end=body.normal_load.copy(),contact_end=body.contacts.copy(),position=body.position.copy(),qpos=body.data.qpos.copy(),feet=body.data.site_xpos[body.toe_ids].copy(),forward=float(body.data.qvel[:3]@r[:,0]),side=float(body.data.qvel[:3]@r[:,1]),heading=body.heading,yaw_rate=body.angular_velocity,tilt=float(np.degrees(np.arccos(np.clip(body.upright,-1,1)))),fell=body.fell,touch=body.touch,energy=body.energy))
  counts0=count.copy()
  if len(data)%250==0:progress('executando',len(data)*.02)
 p.network.add(NetworkOperation(tick,clock=Clock(dt=20*ms,name='cord_body_clock'),when='end',name='cord_tick'))
 progress('compilando');p.network.run(0*ms,namespace={});built=time.perf_counter();print('compiled',name,'prepare_s',built-started,flush=True)
 progress('executando');p.network.run(seconds*1000*ms,namespace={});elapsed=time.perf_counter()-built
 if len(data)!=round(seconds/.02):raise RuntimeError('Incomplete diagnostic')
 a={k:np.asarray([d[k] for d in data]) for k in data[0]};late=slice(len(data)//2,None)
 e=a['counts'][:,p.premotor].mean(4).transpose(1,2,3,0)
 lock,drift,sd,phases,sds=_phase_metrics(e[:,0:1,...,late],.02,PAIRS[6])
 contacts=a['subsupport'][late].reshape(-1,6);A=contacts[:,[0,4,2]].mean(1);B=contacts[:,[3,1,5]].mean(1)
 alt=-float(np.corrcoef(A,B)[0,1]) if min(A.std(),B.std())>1e-10 else 0.
 joints=[joint_metrics(a['angles'][:,k],a['torque'][:,k]) for k in range(6)]
 end=a['contact_end'][late];miss=(a['support'][late]>0)&~end
 r=dict(circuit_revision=2,cross_channel_connected=True,seed=seed,seconds=seconds,cross_gain=cross_gain,neural_only=neural_only,feedback=feedback,shuffled=shuffled,average_feedback=average_feedback,permutation=perm.tolist(),cells=1176,synapses=len(p.synapses),phase_locked=lock,phase_drift_deg=drift,phase_sd_deg=sd,edge_phase_deg=phases,edge_sd_deg=sds,regular_joints=sum(v['rhythmic'] for leg in joints for v in leg),joints=joints,forward_m_s=float(a['forward'][late].mean()),side_rms_m_s=float(np.sqrt((a['side'][late]**2).mean())),yaw_rms_rad_s=float(np.sqrt((a['yaw_rate'][late]**2).mean())),net_m=float(np.linalg.norm(a['position'][-1]-a['position'][len(data)//2])),path_m=float(np.linalg.norm(np.diff(a['position'][late],axis=0),axis=1).sum()),support=contacts.mean(0).tolist(),missed_contact_block_fraction=miss.mean(0).tolist(),three_together=float((contacts[:,[0,4,2]].all(1)|contacts[:,[3,1,5]].all(1)).mean()),alternation=alt,tilt_deg=float(a['tilt'][late].mean()),fell=bool(a['fell'].any()),wall_contact=bool(a['touch'].any()),energy_spent=float(1-a['energy'][-1]),energy_spent_late=float(a['energy'][len(data)//2]-a['energy'][-1]),warm_durations_s=p.warm_durations_s,prepare_wall_s=built-started,simulation_wall_s=elapsed,device='25078PC3EG')
 np.savez_compressed(f.with_suffix('.npz'),**a);atomic(f,r);progress('concluído',seconds);print(json.dumps({k:v for k,v in r.items() if k not in ('joints',)}),flush=True);return r

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--seeds',nargs='+',type=int,default=[42]);a.add_argument('--cross',type=float,default=0);a.add_argument('--seconds',type=float,default=20);a.add_argument('--neural-only',action='store_true');a.add_argument('--open',action='store_true');a.add_argument('--shuffled',action='store_true');a.add_argument('--average-feedback',action='store_true');args=a.parse_args()
 for s in args.seeds:run(s,args.cross,args.seconds,args.neural_only,not args.open,args.shuffled,args.average_feedback)
