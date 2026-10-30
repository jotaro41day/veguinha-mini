"""Short newborn gates with the actual Brian2 network, no learning enabled."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import mujoco
from brian2 import ms, start_scope
from veguinha_mini.config import Config
from veguinha_mini.sim import Simulation
from veguinha_mini.mk1_validation import atomic, versions
from veguinha_mini.coordination_diagnostic import _phase_metrics, PAIRS, SAME_TRIPOD_PAIRS


def rhythm(angle, dt=.02):
    y=angle-angle.mean(axis=0,keepdims=True)
    spectrum=np.abs(np.fft.rfft(y,axis=0))**2
    freq=np.fft.rfftfreq(len(y),dt)
    band=(freq>=.5)&(freq<=8)
    selected=spectrum[band]; grid=freq[band]
    peak=np.argmax(selected,axis=0)
    concentration=np.take_along_axis(selected,peak[None],axis=0)[0]/np.maximum(1e-20,selected.sum(axis=0))
    valid=(np.ptp(angle,axis=0)>.02)&(concentration>.30)
    return valid,1/grid[peak],concentration


class Observed(Simulation):
    def __init__(self,*args,load_gain=1,**kwargs):
        self.observations=[];self.load_gain=load_gain
        super().__init__(*args,**kwargs)

    def _tick(self):
        super()._tick()
        if self.paused:return
        b=self.body;mat=b.data.xmat[b.body_id].reshape(3,3)
        tail=mujoco.mj_name2id(b.model,mujoco.mjtObj.mjOBJ_GEOM,'tail')
        tail_floor=tail>=0 and any((int(c.geom1)==tail and int(c.geom2)==b.floor_id)
                                  or (int(c.geom2)==tail and int(c.geom1)==b.floor_id)
                                  for c in b.data.contact)
        self.observations.append(dict(t=self.simulated_ms/1000,
            angles=b.angles.copy(),rates=self.motor.filtered_hz.copy(),
            torque=self.applied.copy(),support=b.contacts.copy(),
            qpos=b.data.qpos.copy(),velocity=b.data.qvel.copy(),
            forward=float(np.dot(b.data.qvel[:3],mat[:,0])),
            tilt=float(np.degrees(np.arccos(np.clip(b.upright,-1,1)))),fall=bool(b.fell or tail_floor)))
        joint="femur" if self.config.body_model=="mk1" else "hip"
        load=np.clip(b.normal_load/(.316*9.81/6),0,2)
        for index,leg in enumerate(("lf","lm","lh","rf","rm","rh")):
            target=self.model.half_centers[leg][f"{joint}_extend_e"]
            self.model.group.I_sense[target]+=self.load_gain*load[index]


def trial_config(seed, condition, gain=.18, contact_gain=1, motor_bias=None,
                 cross_tau=None, isolated=False):
    options=dict(body_model="control" if condition=="old" else "mk1",
                  block_ms=20,motor_scaffold=True,motor_gain=gain,
                  foot_contact_gain=contact_gain,
                  neural_value_channels=True,motor_plastic_scope='afferent_dn_interleg',
                  vnc_hebb_rate=0,learning_rate=.00001,motor_pool_bias=motor_bias,
                  motor_cross_tau_ms=cross_tau)
    if isolated:
        from veguinha_mini.isolated import isolated_config
        return isolated_config(seed=seed, **options)
    return Config(seed=seed, phase5_enabled=True, **options)


def run(seed,condition,output,seconds=20,gain=.18,load_gain=1,contact_gain=1,
        motor_bias=None,cross_tau=None,isolated=False):
    start_scope()
    config=trial_config(seed,condition,gain,contact_gain,motor_bias,cross_tau,isolated)
    started=time.perf_counter()
    s=Observed(config,headless=True,physics="mujoco",body_substep_ms=2,load_gain=load_gain)
    s.plasticity=False
    if condition=="uncoupled":
        source=np.asarray(s.model.synapses.i[:]);target=np.asarray(s.model.synapses.j[:])
        labels=np.full(len(s.model.group),-1)
        for i,leg in enumerate(("lf","lm","lh","rf","rm","rh")):
            labels[s.model.regions[f"vnc_{leg}"]]=i
        mask=(labels[source]>=0)&(labels[target]>=0)&(labels[source]!=labels[target])
        s.model.synapses.w[np.flatnonzero(mask)]=0
    permutation=np.arange(6)
    if condition=="shuffled":
        rng=np.random.default_rng(seed+19000)
        while np.any(permutation==np.arange(6)):permutation=rng.permutation(6)
    decode=s.motor.update
    def decoded(count,dt):
        torque=decode(count,dt)[permutation].copy();s.applied=torque.copy();return torque
    s.motor.update=decoded
    s.model.network.run(0*ms)  # Compile before starting the measured window.
    s.observations=[]
    warm=time.perf_counter()
    s.run_headless(seconds)
    elapsed=time.perf_counter()-warm
    obs=s.observations; arrays={k:np.asarray([o[k] for o in obs]) for k in obs[0]}
    late= {k:v[-500:] for k,v in arrays.items()}
    valid,period,concentration=rhythm(late['angles'])
    per_leg=valid.any(axis=1)
    a=late['support'][:,[0,4,2]];b=late['support'][:,[3,1,5]]
    aa=a.mean(axis=1);bb=b.mean(axis=1)
    alt=-float(np.corrcoef(aa,bb)[0,1]) if np.std(aa)>1e-10 and np.std(bb)>1e-10 else 0.
    three=float((a.all(axis=1)|b.all(axis=1)).mean())
    n=len(config.joint_names)
    rates=late['rates'].transpose(1,2,0).reshape(6,n,2,-1)
    intra_phases=[];intra_sd=[]
    for leg in range(6):
        joint_pairs=((0,1),(1,2),(0,2)) if n==3 else ((0,1),)
        _,_,_,p,jitter=_phase_metrics(rates[leg,:,None,:,:],.02,joint_pairs)
        intra_phases.append(p);intra_sd.append(jitter)
    selected=rates[:,[1 if n==3 else 0]]
    locked,drift,spread,phases,sds=_phase_metrics(selected,.02,PAIRS[6])
    _,_,_,same,same_sds=_phase_metrics(selected,.02,SAME_TRIPOD_PAIRS)
    same_ok=all(abs(x)<=45 and y<=35 for x,y in zip(same,same_sds))
    v=float(late['forward'].mean())
    row=dict(seed=seed,condition=condition,versions=versions(),seconds=seconds,
             circuit='isolated' if isolated else 'full', cells=len(s.model.group),
             actuator_gear=s.body.model.actuator_gear[:,0].tolist(),
             config=dict(motor_gain=gain,load_gain=load_gain,contact_gain=contact_gain,
                         motor_pool_bias=motor_bias,motor_cross_tau_ms=cross_tau),
             forward_m_s=v,tilt_deg=float(arrays['tilt'].mean()),late_tilt_deg=float(late['tilt'].mean()),
             max_tilt_deg=float(arrays['tilt'].max()),fall=bool(arrays['fall'].any()),
             support=late['support'].mean(axis=0).tolist(),oscillating_leg_fraction=float(per_leg.mean()),
             oscillating_joints=valid.tolist(),period_s=period.tolist(),
             spectral_fraction=concentration.tolist(),rates_hz=late['rates'].mean(axis=0).tolist(),
             torque_rms=np.sqrt((late['torque']**2).mean(axis=0)).tolist(),
             saturation=s.motor.saturation_fraction(),alternation=alt,three_feet=three,
             phase_window_s=min(seconds,10),
             phase_deg=phases,phase_sd_deg=sds,phase_drift_deg=drift,
             intra_joint_phase_deg=intra_phases,intra_joint_sd_deg=intra_sd,
             tripod=bool(locked and same_ok and v>0 and alt>.3 and three>.3),
             viable=bool(not arrays['fall'].any() and arrays['tilt'].mean()<15 and per_leg.all()),
             compilation_and_build_wall_s=warm-started,simulation_wall_s=elapsed,
             wall_s_per_sim_s=elapsed/seconds,permutation=permutation.tolist())
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output.with_suffix('.npz'),**arrays)
    atomic(output,row)
    print(json.dumps(row),flush=True)
    return row


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42)
    p.add_argument('--condition',choices=('coupled','uncoupled','shuffled','old'),default='coupled')
    p.add_argument('--seconds',type=float,default=20);p.add_argument('--gain',type=float,default=.18)
    p.add_argument('--load-gain',type=float,default=1);p.add_argument('--contact-gain',type=float,default=1)
    p.add_argument('--motor-bias',type=float)
    p.add_argument('--cross-tau',type=float)
    p.add_argument('--isolated',action='store_true')
    p.add_argument('--output',required=True)
    a=p.parse_args();run(a.seed,a.condition,a.output,a.seconds,a.gain,a.load_gain,a.contact_gain,a.motor_bias,a.cross_tau,a.isolated)
