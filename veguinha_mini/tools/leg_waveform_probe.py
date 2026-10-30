"""Mechanical probes only; these waveforms never control the neural animal."""
import os
import argparse
import time
import numpy as np
from veguinha_mini.leg_circuit import FixedTrunk
from veguinha_mini.tools.leg_reflex_check import OUT,joint_metrics
from veguinha_mini.mk1_validation import atomic


def run(seed,condition,rms=.35/np.sqrt(2)):
    if not 0<rms<=.5:raise ValueError('Diagnostic RMS outside bounds')
    a=np.load(OUT/f'link_{seed}_n0.5_p2_rail_open.npz');u=a['torque'].copy()
    if condition!='neural_matched':
        phase=np.random.default_rng(seed).uniform(0,2*np.pi)
        offset=np.array([0.,0.,np.pi/2 if condition=='sine_quadrature' else 0.])
        u=np.sin(2*np.pi*np.arange(1000)[:,None]*.02/.38+phase+offset)
    # Match RMS across probe conditions; these commands never enter the app.
    u*= rms/np.sqrt((u[-500:]**2).mean(0))
    if abs(u).max()>1:raise ValueError('RMS matching would clip the command')
    b=FixedTrunk(horizontal_free=True,walls=False);data=[];start=time.perf_counter()
    for k,torque in enumerate(u):
        stance,z=b.step(torque);data.append(np.r_[b.angles-b.neutral,stance,z,b.load])
        if k%250==0:atomic(OUT/'progress.json',dict(pid=os.getpid(),state='sonda física',profile='probe_'+condition,seed=seed,simulated_s=k*.02,updated=time.time()))
    x=np.array(data)
    row=dict(profile='probe_'+condition,mode='physics_replay',fixture='rail_open',cells=0,
        dt_ms=None,physics_dt_ms=2,seed=seed,seconds=20,command_rms=rms,device='controller 25078PC3EG',
        simulation_wall_s=time.perf_counter()-start,support=float(x[500:,3].mean()),
        max_clearance_mm=float(x[500:,4].max()*1000),normal_load_mean_n=float(x[500:,5].mean()),
        torque_rms_nm=np.sqrt((u[-500:]**2).mean(0))*np.array([.08,.2,.2]),
        joints=dict(zip(('coxa','femur','tibia'),joint_metrics(x[:,:3],u))))
    row['torque_rms_nm']=row['torque_rms_nm'].tolist()
    name=f'probe_{condition}_{seed}'+(f'_rms{rms:g}' if rms!=.35/np.sqrt(2) else '');np.savez_compressed(OUT/(name+'.npz'),samples=x,torque=u)
    atomic(OUT/(name+'.json'),row)
    print(seed,condition,'regular',[v['rhythmic'] for v in row['joints'].values()],
          'acf',[round(v['autocorrelation'],3) for v in row['joints'].values()],
          'support',round(row['support'],3),'lift',round(row['max_clearance_mm'],1),flush=True)
    return row


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--rms',type=float,default=.35/np.sqrt(2));args=parser.parse_args();rows=[]
    for seed in (42,43,44):
        for condition in ('neural_matched','sine_same','sine_quadrature'):rows.append(run(seed,condition,args.rms))
    atomic(OUT/('waveform_probe_summary.json' if args.rms==.35/np.sqrt(2) else f'waveform_probe_rms{args.rms:g}.json'),dict(trials=rows))
    atomic(OUT/'progress.json',dict(pid=os.getpid(),state='concluído',profile='probe_sine_quadrature',seed=44,simulated_s=20,updated=time.time()))
