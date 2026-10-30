"""Controlled premotor input is a diagnostic stimulus, never an animal controller."""
import argparse
import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
from brian2 import (Clock,Network,NeuronGroup,SpikeGeneratorGroup,Synapses,ms,prefs,
                    seed as brian_seed,start_scope)
from veguinha_mini.network import EQUATIONS,EmittedSpikeMonitor
from veguinha_mini.motor import MotorDecoder
from veguinha_mini.mk1_validation import atomic,versions

OUT=Path(__file__).resolve().parents[1]/'results/motor_pool_v1'
LIVE=OUT.parent/'leg_reflex_v1'
BASE=dict(pool=8,filter_ms=120,noise=2.5,tonic=0.,threshold_span=0.,shared=True,parent=None)
CASES={'baseline':BASE}
for name,parent,field,value in (
    ('n16','baseline','pool',16),('n32','baseline','pool',32),
    ('filter20','baseline','filter_ms',20),('filter50','baseline','filter_ms',50),
    ('filter100','baseline','filter_ms',100),('quiet','baseline','noise',0.),
    ('tonic','quiet','tonic',2.),('tonic_threshold','tonic','threshold_span',5.),
    ('private','baseline','shared',False)):
    CASES[name]={**CASES[parent],field:value,'parent':parent}


def stimulus(seed,shared=True):
    rng=np.random.default_rng(seed)
    groups=1 if shared else 32
    offsets=rng.integers(0,11,(groups,22))*.0005
    indices=[];times=[]
    for g in range(groups):
        for neuron in range(22):
            t=.1+np.arange(53)*.38+(0 if neuron<12 else .19)+offsets[g,neuron]
            t=t[t<20];indices.extend([22*g+neuron]*len(t));times.extend(t)
    order=np.argsort(times,kind='stable')
    return np.asarray(indices,dtype=int)[order],np.asarray(times)[order]


def decode_pool(counts,filter_ms):
    n=counts.shape[1]
    config=SimpleNamespace(joint_names=('tibia',),motor_filter_ms=filter_ms,
        motor_baseline_ms=5000,motor_gain=.18,motor_limit=1.)
    model=SimpleNamespace(group=np.zeros(2*n),motor_pools={'lf':{
        'tibia_flex':np.arange(n),'tibia_extend':np.arange(n,2*n)}})
    decoder=MotorDecoder(model,config,legs=('lf',));cumulative=np.zeros(2*n,int)
    torques=[];rates=[]
    for row in counts:
        cumulative[:n]+=row;torques.append(decoder.update(cumulative,.02)[0,0])
        rates.append(decoder.filtered_hz[0,0])
    return dict(torque=np.asarray(torques),filtered_rate=np.asarray(rates))


def cycle_stats(rate_counts,torque):
    # P=.38 s equals exactly 19 samples at the application's 20-ms block.
    rate=np.asarray(rate_counts)[:52*19].reshape(52,19)[27:]
    u=np.asarray(torque)[:52*19].reshape(52,19)[27:]
    rates=rate.sum(1)/.38;rms=np.sqrt((u*u).mean(1))
    def cv(x):return float(x.std(ddof=1)/max(abs(x.mean()),1e-12))
    return dict(rate_cv=cv(rates),torque_cv=cv(rms),mean_rate_hz=float(rates.mean()),
        torque_rms_nm=float(np.sqrt((u*u).mean())*.2),
        torque_amplitude_nm=float((np.quantile(u,.95)-np.quantile(u,.05))*.2),
        saturation=float((abs(u)>=.98).mean()))


def progress(case,seed,state,seconds=0):
    atomic(LIVE/'progress.json',dict(pid=os.getpid(),profile='pool_'+case,seed=seed,
           state=state,simulated_s=seconds,updated=time.time()))


def configure_leg(model,case):
    cfg=CASES[case]
    if not cfg['shared']:raise ValueError('Private stimulus is a synthetic bench control only')
    motor=model.motors.ravel()
    model.group.noise[motor]=cfg['noise']
    model.group.I_base[motor]=cfg['tonic']
    model.group.v_threshold[motor]=np.tile(30+cfg['threshold_span']*np.linspace(-1,1,cfg['pool']),6)
    model.config.motor_filter_ms=cfg['filter_ms']


def run(case,seed):
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/f'{case}_{seed}.json'
    if path.exists():return json.loads(path.read_text())
    gc.collect();start_scope();prefs.codegen.target='cython';brian_seed(seed)
    cfg=CASES[case];n=cfg['pool'];rng=np.random.default_rng(seed)
    clock=Clock(dt=.5*ms,name='pool_clock')
    eq=EQUATIONS+'\nv_threshold:1\n'
    g=NeuronGroup(64,eq,threshold='v>=v_threshold',reset='v=c;u+=d',
        events={'emit':'v>=v_threshold and spike_gate>0'},method='euler',clock=clock,name='pool_cells')
    g.a=.02;g.b=.2;g.c=-65;g.d=8;g.v=-65+rng.normal(0,2,64);g.u='b*v'
    g.I_base=0;g.I_base[:n]=cfg['tonic'];g.noise=cfg['noise'];g.noise_gain=0;g.noise_gain[:n]=1
    g.v_threshold=30;g.v_threshold[:n]=30+cfg['threshold_span']*np.linspace(-1,1,n)
    g.spike_gate=1;g.tau_syn=5*ms
    g.run_regularly('I_noise=noise*noise_gain*randn()',dt=1*ms,name='pool_noise')
    indices,times=stimulus(seed,cfg['shared'])
    source=SpikeGeneratorGroup(704,indices,times*1000*ms,clock=clock,name='pool_input')
    syn=Synapses(source,g,'w:1',on_pre='I_syn_post+=w',clock=clock,name='pool_synapses')
    pre=[];post=[];weights=[]
    for motor in range(n):
        pre.extend((np.arange(22)+(0 if cfg['shared'] else 22*motor)).tolist())
        post.extend([motor]*22);weights.extend(([12/np.sqrt(22)]*12)+([-2/np.sqrt(22)]*10))
    syn.connect(i=pre,j=post);syn.w=weights;syn.delay=2*ms
    monitor=EmittedSpikeMonitor(g,'emit',record=True,name='pool_spikes')
    net=Network(g,source,syn,monitor);progress(case,seed,'compilando')
    start=time.perf_counter();net.run(0*ms);build=time.perf_counter()-start
    progress(case,seed,'executando');start=time.perf_counter();net.run(20000*ms)
    elapsed=time.perf_counter()-start
    spike_t=np.asarray(monitor.t/ms)/1000;spike_i=np.asarray(monitor.i[:],int)
    bins=np.floor((spike_t+1e-10)/.02).astype(int);valid=(spike_i<n)&(bins<1000)
    counts=np.zeros((1000,n),int);np.add.at(counts,(bins[valid],spike_i[valid]),1)
    decoded=decode_pool(counts,cfg['filter_ms']);metrics=cycle_stats(counts.sum(1)/n,decoded['torque'])
    e=((indices%22)<12)&((indices<22*n) if not cfg['shared'] else True);input_bins=np.floor((times[e]+1e-10)/.02).astype(int)
    input_counts=np.bincount(input_bins,minlength=1000)[:1000]/(12*(1 if cfg['shared'] else n))
    input_metrics=cycle_stats(input_counts,np.ones(1000))
    row=dict(case=case,seed=seed,config=cfg,seconds=20,window_s=[10.26,19.76],
        device='controller 25078PC3EG',versions=versions(),input_rate_cv=input_metrics['rate_cv'],
        input_rate_hz=input_metrics['mean_rate_hz'],**metrics,
        simulation_wall_s=elapsed,build_wall_s=build)
    np.savez_compressed(OUT/f'{case}_{seed}.npz',counts=counts,spike_t=spike_t,spike_i=spike_i,
                        torque=decoded['torque'],filtered_rate=decoded['filtered_rate'],input_counts=input_counts)
    atomic(path,row);progress(case,seed,'concluído',20)
    print(case,seed,'CV motor/torque',round(100*metrics['rate_cv'],2),round(100*metrics['torque_cv'],2),
          'RMSNm',round(metrics['torque_rms_nm'],4),'real_s',round(elapsed,3),'build',round(build,2),flush=True)
    return row


def summary():
    result=[]
    for case in CASES:
        paths=[OUT/f'{case}_{s}.json' for s in range(42,52)]
        if not all(p.exists() for p in paths):continue
        rows=[json.loads(p.read_text()) for p in paths]
        result.append(dict(case=case,n=10,config=CASES[case],**{k:dict(mean=float(np.mean([r[k] for r in rows])),
            sd=float(np.std([r[k] for r in rows],ddof=1))) for k in
            ('input_rate_cv','rate_cv','torque_cv','torque_rms_nm','torque_amplitude_nm','simulation_wall_s')}))
    atomic(OUT/'pool_summary.json',dict(groups=result,seeds=list(range(42,52))))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cases',nargs='+',choices=list(CASES),default=list(CASES))
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(42,52)))
    args=p.parse_args();start=time.perf_counter()
    for case in args.cases:
        for seed in args.seeds:
            if time.perf_counter()-start>210:
                summary();print('Wall budget reached; rerun the same command to resume.',flush=True);raise SystemExit(0)
            run(case,seed)
    print(json.dumps(summary(),ensure_ascii=False),flush=True)
