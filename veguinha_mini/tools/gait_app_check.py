"""Observe the real application controller; no learning or prescribed actions."""
import argparse,json,time,gc
from pathlib import Path
import numpy as np
from brian2 import start_scope,ms
from veguinha_mini.config import Config
from veguinha_mini.sim import Simulation
from veguinha_mini.mk1_validation import atomic
from veguinha_mini.tools.leg_reflex_check import joint_metrics
from veguinha_mini.coordination_diagnostic import _phase_metrics,PAIRS
OUT=Path(__file__).resolve().parents[1]/'results/gait_loop_v1'
LIVE=OUT.parent/'leg_reflex_v1'
class Observed(Simulation):
 def __init__(self,*a,**kw):
  self.samples=[];self.progress_hook=None;super().__init__(*a,**kw)
  self.body.sample_contacts=True  # Measurement only for legacy; its encoder still reads end contacts.
 def _tick(self):
  super()._tick();b=self.body;r=b.data.xmat[b.body_id].reshape(3,3)
  self.samples.append(dict(angles=b.angles.copy(),torque=self.applied_torque.copy(),rates=self.motor.filtered_hz.copy(),support=b.contact_fraction.copy(),subsupport=b.foot_load_samples>.02,load=b.mean_normal_load.copy(),contact_end=b.contacts.copy(),position=b.position.copy(),qpos=b.data.qpos.copy(),feet=b.data.site_xpos[b.toe_ids].copy(),forward=float(b.data.qvel[:3]@r[:,0]),yaw_rate=b.angular_velocity,tilt=float(np.degrees(np.arccos(np.clip(b.upright,-1,1)))),fell=b.fell,touch=b.touch,energy=b.energy,counts=np.asarray(self.model.monitor.count[:]).copy()))
  if self.progress_hook and len(self.samples)%50==0:self.progress_hook('executando',len(self.samples)*.02)
def run(seed=42,circuit='legacy',seconds=20,shuffled=False,cross_gain=2,silence_tail=0):
 name=f'v2_app_{circuit}_{seed}_s{seconds:g}_{"shuffled" if shuffled else "normal"}';name+=f'_k{cross_gain:g}' if cross_gain!=2 else '';name+=f'_silence{silence_tail:g}' if silence_tail else '';path=OUT/(name+'.json')
 if path.exists():return json.loads(path.read_text())
 gc.collect();start_scope();started=time.perf_counter()
 atomic(LIVE/'task.json',dict(title='Validando a malha completa do aplicativo',detail=f'Cérebro preservado, VNC {circuit};seed {seed};{seconds:g}s;sem plasticidade.'))
 def progress(state,t=0):atomic(LIVE/'progress.json',dict(pid=__import__('os').getpid(),state=state,profile=name,seed=seed,simulated_s=t,total_s=seconds+silence_tail,updated=time.time()))
 progress('preparando')
 s=Observed(Config(seed=seed,body_model='mk1',block_ms=20,motor_circuit=circuit,dt_ms=.5 if circuit=='modular' else 1,motor_cross_strength=cross_gain),headless=True,physics='mujoco',body_substep_ms=2)
 s.progress_hook=progress
 s.plasticity=False;perm=np.arange(6);rng=np.random.default_rng(seed+30000)
 if shuffled:
  while np.any(perm==np.arange(6)):perm=rng.permutation(6)
 decode=s.motor.update
 def decoded(count,dt):
  s.applied_torque=decode(count,dt)[perm].copy();return s.applied_torque.copy()
 s.motor.update=decoded
 progress('compilando');s.model.network.run(0*ms);built=time.perf_counter();print('compiled',name,built-started,flush=True)
 progress('executando');s.run_headless(seconds);wall=time.perf_counter()-built
 a={k:np.asarray([x[k] for x in s.samples]) for k in s.samples[0]};assert len(a['angles'])==round(seconds/.02)
 tail_stats=None;full=a
 if silence_tail:
  s.model.group.spike_gate=0;s.run_headless(silence_tail)
  full={k:np.asarray([x[k] for x in s.samples]) for k in s.samples[0]}
  tail_stats=dict(last_second_speed=float(np.mean(full['forward'][-50:])),last_second_world_speed_rms=float(np.sqrt(np.mean(np.sum((np.diff(full['position'][-51:],axis=0)/.02)**2,axis=1)))),last_second_angle_span_rad=np.ptp(full['angles'][-50:],axis=0).tolist(),last_second_torque_rms=float(np.sqrt((full['torque'][-50:]**2).mean())),emitted_spikes_after_first_block=int((full['counts'][-1]-full['counts'][len(a['angles'])+1]).sum()))
 late=slice(len(a['angles'])//2,None);m=np.asarray([[s.model.motor_pools[l][n] for n in s.motor.pool_names] for l in s.motor.legs]);rate=a['counts'][-1,m]/seconds
 contacts=a['subsupport'][late].reshape(-1,6);A=contacts[:,[0,4,2]].mean(1);B=contacts[:,[3,1,5]].mean(1)
 alt=-float(np.corrcoef(A,B)[0,1]) if min(A.std(),B.std())>1e-10 else 0.
 counts=np.diff(a['counts'],axis=0,prepend=np.zeros_like(a['counts'][:1]))
 e=counts[:,s.model.cord.premotor].mean(4).transpose(1,2,3,0) if s.model.cord else a['rates'].reshape(-1,6,3,2).transpose(1,2,3,0)
 phase=_phase_metrics(e[:,0:1,...,late],.02,PAIRS[6])
 joints=[joint_metrics(a['angles'][:,k],a['torque'][:,k]) for k in range(6)]
 regional_rates={n:float((a["counts"][-1,ids]-a["counts"][len(a["angles"])//2,ids]).sum()/len(ids)/((len(a["angles"])-1-len(a["angles"])//2)*.02)) for n,ids in s.model.regions.items()}
 r=dict(controller_revision=2,regional_rates_hz=regional_rates,seed=seed,seconds=seconds,silence_tail_s=silence_tail,silence_result=tail_stats,cross_gain=cross_gain,motor_circuit=circuit,cells=len(s.model.group),synapses=len(s.model.synapses),shuffled=shuffled,permutation=perm.tolist(),phase_locked=phase[0],phase_sd_deg=phase[2],edge_phases_deg=phase[3],forward_m_s=float(a['forward'][late].mean()),net_m=float(np.linalg.norm(a['position'][-1]-a['position'][len(a['angles'])//2])),path_m=float(np.linalg.norm(np.diff(a['position'][late],axis=0),axis=1).sum()),tilt_deg=float(a['tilt'][late].mean()),fell=bool(a['fell'].any()),wall_contact=bool(a['touch'].any()),support=contacts.mean(0).tolist(),three_together=float((contacts[:,[0,4,2]].all(1)|contacts[:,[3,1,5]].all(1)).mean()),alternation=alt,regular_joints=sum(v['rhythmic'] for leg in joints for v in leg),joints=joints,motor_rates_hz=rate.mean(2).tolist(),motor_torque_rms=np.sqrt((a['torque'][late]**2).mean(0)).tolist(),energy_spent_late=float(a['energy'][len(a['angles'])//2]-a['energy'][-1]),prepare_wall_s=built-started,wall_s=wall,device='25078PC3EG')
 np.savez_compressed(path.with_suffix('.npz'),**full);atomic(path,r);progress('concluído',seconds);print(json.dumps({k:v for k,v in r.items() if k!='joints'}),flush=True);return r
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--circuit',choices=('legacy','modular'),default='legacy');p.add_argument('--seeds',nargs='+',type=int,default=[42]);p.add_argument('--seconds',type=float,default=20);p.add_argument('--shuffled',action='store_true');p.add_argument('--cross',type=float,default=2);p.add_argument('--silence-tail',type=float,default=0);a=p.parse_args()
 for seed in a.seeds:run(seed,a.circuit,a.seconds,a.shuffled,a.cross,a.silence_tail)
