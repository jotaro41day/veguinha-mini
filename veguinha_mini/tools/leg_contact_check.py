"""Substep force audit of the one-leg fixture; recorded torques, no learning."""
import os
import argparse
import time
from collections import Counter
import numpy as np
import mujoco
from veguinha_mini.leg_circuit import FixedTrunk
from veguinha_mini.tools.leg_reflex_check import OUT, joint_metrics
from veguinha_mini.mk1_validation import atomic


def run(seed,condition):
    source=OUT/f'link_{seed}_n0.5_p2_rail_open.npz'
    recorded=np.load(source);u=recorded['torque'];b=FixedTrunk(horizontal_free=True,walls=False,compliant=condition=='compliant')
    if condition=='frictionless':
        b.model.geom_friction[b.floor]=0;b.model.geom_friction[b.foot]=0
    elif condition not in ('original','compliant'):raise ValueError(condition)
    dof=np.array([b.model.joint('lf_'+j).dofadr[0] for j in ('coxa','femur','tibia')])
    fine=[];contacts=Counter();coarse=[];start=time.perf_counter()
    def observe(body,load):
        d=body.data
        fine.append(np.r_[d.time,load,d.qpos[body.qpos],d.qvel[dof],d.qfrc_actuator[dof],
                          d.qfrc_constraint[dof],d.qfrc_passive[dof],d.qfrc_bias[dof]])
        if len(fine)>5000:
            for c in d.contact:
                contacts[tuple(sorted((body.model.geom(int(c.geom1)).name,body.model.geom(int(c.geom2)).name)))]+=1
    for k,torque in enumerate(u):
        support,z=b.step(torque,observe=observe)
        coarse.append(np.r_[b.angles-b.neutral,support,z,b.data.xpos[b.root,2]])
        if k%250==0:atomic(OUT/'progress.json',dict(pid=os.getpid(),state='medindo contato',
                profile='contact_'+condition,seed=seed,simulated_s=k*.02,updated=time.time()))
    c=np.array(coarse);f=np.array(fine);late=f[5000:];vel=late[:,5:8];motor=late[:,8:11];reaction=late[:,11:14]
    in_contact=late[:,1]>.02
    row=dict(profile='contact_'+condition,mode='physics_replay',seed=seed,source=source.name,
        seconds=20,device='controller 25078PC3EG',fixture='rail_compliant' if condition=='compliant' else 'rail_open',cells=0,dt_ms=None,physics_dt_ms=2,
        support_constants=b.support_constants,normal_load_p95_n=float(np.quantile(late[:,1],.95)),
        root_vertical_range_m=float(np.ptp(c[500:,5])),
        simulation_wall_s=time.perf_counter()-start,support=float(in_contact.mean()),
        max_clearance_mm=float(c[500:,4].max()*1000),
        joints=dict(zip(('coxa','femur','tibia'),joint_metrics(c[:,:3],u))),
        torque_motor_rms_nm=np.sqrt((motor*motor).mean(0)).tolist(),
        torque_constraint_rms_nm=np.sqrt((reaction*reaction).mean(0)).tolist(),
        constraint_against_motor_stance=((motor[in_contact]*reaction[in_contact]<0).mean(0)).tolist(),
        motor_positive_work_j=(np.maximum(motor*vel,0).sum(0)*.002).tolist(),
        constraint_work_j=((reaction*vel).sum(0)*.002).tolist(),
        contacts_late=[dict(pair=list(k),samples=v) for k,v in contacts.items()],
        exact_replay_error_rad=float(abs(c[:,:3]-recorded['angles']).max()) if condition=='original' else None)
    name=f'contact_{condition}_{seed}'
    np.savez_compressed(OUT/(name+'.npz'),fine=f,coarse=c,torque=u)
    atomic(OUT/(name+'.json'),row)
    print(seed,condition,'regular',[v['rhythmic'] for v in row['joints'].values()],
          'constraint_rms',np.round(row['torque_constraint_rms_nm'],3),
          'contacts',row['contacts_late'],'wall',round(row['simulation_wall_s'],2),flush=True)
    return row


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--conditions',nargs='+',choices=('original','frictionless','compliant'),default=['original','frictionless'])
    args=parser.parse_args();rows=[]
    for seed in (42,43,44):
        for condition in args.conditions:rows.append(run(seed,condition))
    atomic(OUT/('contact_force_'+ '_'.join(args.conditions)+'.json'),dict(trials=rows))
    atomic(OUT/'progress.json',dict(pid=os.getpid(),state='concluído',profile='contact_'+args.conditions[-1],seed=44,simulated_s=20,updated=time.time()))
