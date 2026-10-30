"""Bounded one-leg validation of motor-pool variants; no learning or six-leg run."""
import argparse
import json
import os
from types import SimpleNamespace
from pathlib import Path
import time
import numpy as np
from veguinha_mini.tools.motor_pool_bench import CASES,OUT
from veguinha_mini.tools.leg_reflex_check import OUT as LEG,run
from veguinha_mini.mk1_validation import atomic


def path(case,seed):return LEG/f'poolleg_{case}_{seed}_n0.5_p2_rail_open.json'


def actual_variability(case,seed):
    row=json.loads(path(case,seed).read_text());a=np.load(path(case,seed).with_suffix('.npz'))
    c=a['counts'];e=c[:,48:120].reshape(-1,3,2,12).mean(3)
    m=c[:,np.asarray(row['motor_indices'])].mean(3)
    y=e[:,0,0]-e[:,0,1];f=np.fft.fftfreq(len(y),.02)
    phase=np.unwrap(np.angle(np.fft.ifft(np.fft.fft(y)*((f>=2)&(f<=3.5)))))
    starts=np.flatnonzero(np.diff(np.floor((phase-np.pi/2)/(2*np.pi)))>0)
    starts=starts[(starts>=500)&(starts<950)]
    def cv(x):
        v=np.array([x[i:j].sum(0) for i,j in zip(starts[:-1],starts[1:])])
        return v.std(0)/np.maximum(v.mean(0),1e-12)
    def rate_cv(x):
        values=np.array([x[i:j].sum(0)/((j-i)*.02) for i,j in zip(starts[:-1],starts[1:])])
        return (values.std(0)/np.maximum(values.mean(0),1e-12)).tolist()
    torque_cycles=np.array([np.sqrt((a['torque'][i:j]**2).mean(0)) for i,j in zip(starts[:-1],starts[1:])])
    return dict(premotor_cv=cv(e).tolist(),motor_cv=cv(m).tolist(),
        premotor_rate_cv=rate_cv(e),motor_rate_cv=rate_cv(m),
        torque_cv=(torque_cycles.std(0)/np.maximum(torque_cycles.mean(0),1e-12)).tolist())


def replay_filter(case,seed):
    from veguinha_mini.leg_circuit import FixedTrunk
    from veguinha_mini.motor import MotorDecoder
    from veguinha_mini.tools.leg_reflex_check import joint_metrics
    source=path('baseline',seed);row=json.loads(source.read_text());saved=np.load(source.with_suffix('.npz'))
    cfg=CASES[case];indices=np.asarray(row['motor_indices'])
    names=('coxa','femur','tibia')
    config=SimpleNamespace(joint_names=names,motor_filter_ms=cfg['filter_ms'],motor_baseline_ms=5000,motor_gain=.18,motor_limit=1.)
    model=SimpleNamespace(group=np.zeros(saved['counts'].shape[1]),motor_pools={'lf':{
        name+'_'+action:indices[j,a] for j,name in enumerate(names) for a,action in enumerate(('flex','extend'))}})
    decoder=MotorDecoder(model,config,legs=('lf',));body=FixedTrunk(horizontal_free=True,walls=False)
    counts=np.zeros(len(model.group),int);data=[];start=time.perf_counter()
    for k,increment in enumerate(saved['counts']):
        counts+=increment;torque=decoder.update(counts,.02)[0];stance,z=body.step(torque)
        data.append(dict(angles=body.angles-body.neutral,torque=torque,rates=decoder.filtered_hz[0].copy(),
            support=stance,clearance=z,load=body.load,root_position=body.data.xpos[body.root].copy(),wall_contact=False))
        if k%250==0:atomic(LEG/'progress.json',dict(pid=os.getpid(),profile='poolleg_'+case,seed=seed,state='reproduzindo spikes',simulated_s=k*.02,updated=time.time()))
    arrays={k:np.asarray([r[k] for r in data]) for k in data[0]};arrays['counts']=saved['counts']
    if case=='baseline':
        error=max(float(abs(arrays[k]-saved[k]).max()) for k in ('angles','torque','rates','support'))
        print('Exact replay error',seed,error,flush=True)
        if error!=0:raise AssertionError('Spike replay differs from neural reference')
        atomic(OUT/f'replay_check_{seed}.json',dict(seed=seed,max_error=error));return
    row.update(profile='poolleg_'+case,motor_case=case,motor_case_config=cfg,engine='exact baseline spike replay',
        replay_source=source.name,joints=dict(zip(names,joint_metrics(arrays['angles'],arrays['torque']))),
        support=float(arrays['support'][-500:].mean()),max_clearance_mm=float(arrays['clearance'][-500:].max()*1000),
        air_fraction=float((arrays['clearance'][-500:]>.002).mean()),
        motor_rates_hz=arrays['rates'][-500:].mean(0).reshape(-1,2).tolist(),
        simulation_wall_s=time.perf_counter()-start,build_wall_s=0.)
    np.savez_compressed(path(case,seed).with_suffix('.npz'),**arrays);atomic(path(case,seed),row)
    print(case,seed,'regular',sum(v['rhythmic'] for v in row['joints'].values()),'wall_s',row['simulation_wall_s'],flush=True)


def report(seeds):
    groups=[]
    for case in CASES:
        paths=[path(case,s) for s in seeds]
        if not all(p.exists() for p in paths):continue
        rows=[json.loads(p.read_text()) for p in paths]
        amplitude=np.array([[v['amplitude_deg'] for v in r['joints'].values()] for r in rows])
        baseline=[path('baseline',s) for s in seeds]
        ratio=None
        if all(p.exists() for p in baseline):
            base=np.array([[v['amplitude_deg'] for v in json.loads(p.read_text())['joints'].values()] for p in baseline])
            ratio=(amplitude/base).mean(0).tolist()
        variability=[actual_variability(case,s) for s in seeds]
        groups.append(dict(case=case,seeds=seeds,regular=sum(v['rhythmic'] for r in rows for v in r['joints'].values()),
            total=3*len(seeds),amplitude_ratio=ratio,
            motor_cv=float(np.mean([r['motor_cv'] for r in variability])),
            premotor_cv=float(np.mean([r['premotor_cv'] for r in variability])),
            motor_rate_cv=float(np.mean([r['motor_rate_cv'] for r in variability])),
            premotor_rate_cv=float(np.mean([r['premotor_rate_cv'] for r in variability])),
            torque_cv=float(np.mean([r['torque_cv'] for r in variability])),
            mean_simulation_wall_s=float(np.mean([r['simulation_wall_s'] for r in rows])),
            simulation_wall_s=float(sum(r['simulation_wall_s'] for r in rows))))
    atomic(OUT/('leg_summary_'+str(min(seeds))+'_'+str(max(seeds))+'.json'),dict(groups=groups))
    return groups


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cases',nargs='+',required=True,choices=[c for c in CASES if c!='private'])
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);p.add_argument('--replay-filters',action='store_true');args=p.parse_args();start=time.perf_counter()
    for seed in args.seeds:
        for case in args.cases:
            if path(case,seed).exists():continue
            if time.perf_counter()-start>200:
                print('Wall budget reached. Resume with the same command.',flush=True);print(report(args.seeds),flush=True);raise SystemExit(0)
            if args.replay_filters:
                if case not in ('baseline','filter20','filter50','filter100'):raise ValueError('Only decoder filters can reuse the same spikes')
                if not (OUT/f'replay_check_{seed}.json').exists():replay_filter('baseline',seed)
                replay_filter(case,seed)
            else:run(seed,'link',fixture='rail_open',motor_case=case)
    print(report(args.seeds),flush=True)
