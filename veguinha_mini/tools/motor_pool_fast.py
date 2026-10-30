"""Fast isolated-pool experiments. Use only after matching the Brian2 reference."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs':np.get_include()},language_level=3)
from veguinha_mini.motor_pool_kernel import simulate
from veguinha_mini.tools.motor_pool_bench import OUT,CASES,stimulus,decode_pool,cycle_stats,progress,summary
from veguinha_mini.mk1_validation import atomic,versions


def raw(case,seed):
    cfg=CASES[case];n=cfg['pool'];indices,times=stimulus(seed,cfg['shared'])
    pattern=np.zeros((760,64))
    for i,t in zip(indices[times<.38],times[times<.38]):
        channel=i%22;group=i//22;step=round(t/.0005)+4
        weight=(12 if channel<12 else -2)/np.sqrt(22)
        if cfg['shared']:pattern[step,:n]+=weight
        elif group<n:pattern[step,group]+=weight
    noise=np.random.RandomState(seed).standard_normal((20000,64))
    v=-65+np.random.default_rng(seed).normal(0,2,64)
    threshold=30+cfg['threshold_span']*np.linspace(-1,1,n)
    start=time.perf_counter()
    counts,t,i=simulate(v,noise,pattern,n,cfg['tonic'],cfg['noise'],threshold)
    return counts,t,i,time.perf_counter()-start


def compare(case,seed):
    metadata=json.loads((OUT/f'{case}_{seed}.json').read_text())
    if metadata.get('engine','brian2')!='brian2':raise ValueError('A fast-kernel result cannot be its own reference')
    reference=np.load(OUT/f'{case}_{seed}.npz');counts,t,i,elapsed=raw(case,seed)
    result=dict(case=case,seed=seed,count_max_error=int(abs(counts-reference['counts']).max()),
                total_spike_difference=int(counts.sum()-reference['counts'].sum()),
                spike_indices_identical=bool(np.array_equal(i,reference['spike_i'])),
                spike_time_max_error_s=float(np.max(abs(t-reference['spike_t']))) if len(t)==len(reference['spike_t']) else None,wall_s=elapsed)
    print(result,flush=True);return result


def run(case,seed):
    p=OUT/f'{case}_{seed}.json'
    if p.exists():return json.loads(p.read_text())
    progress(case,seed,'executando');start=time.perf_counter()
    counts,t,i,core=raw(case,seed);d=decode_pool(counts,CASES[case]['filter_ms'])
    metrics=cycle_stats(counts.sum(1)/CASES[case]['pool'],d['torque'])
    source_i,source_t=stimulus(seed,CASES[case]['shared']);mask=((source_i%22)<12)&((source_i<22*CASES[case]['pool']) if not CASES[case]['shared'] else True)
    bins=np.floor((source_t[mask]+1e-10)/.02).astype(int)
    input_counts=np.bincount(bins,minlength=1000)[:1000]/(12*(1 if CASES[case]['shared'] else CASES[case]['pool']))
    source=cycle_stats(input_counts,np.ones(1000))
    row=dict(case=case,seed=seed,config=CASES[case],seconds=20,window_s=[10.26,19.76],
        device='controller 25078PC3EG',engine='validated Cython Euler',versions=versions(),
        input_rate_cv=source['rate_cv'],input_rate_hz=source['mean_rate_hz'],**metrics,
        simulation_wall_s=time.perf_counter()-start,kernel_wall_s=core,build_wall_s=0)
    np.savez_compressed(OUT/f'{case}_{seed}.npz',counts=counts,spike_t=t,spike_i=i,
        torque=d['torque'],filtered_rate=d['filtered_rate'],input_counts=input_counts)
    atomic(p,row);progress(case,seed,'concluído',20)
    print(case,seed,round(100*row['rate_cv'],2),round(100*row['torque_cv'],2),round(row['simulation_wall_s'],3),flush=True)
    return row


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--compare',action='store_true');p.add_argument('--cases',nargs='+',default=list(CASES),choices=list(CASES));p.add_argument('--seeds',type=int,nargs='+',default=list(range(42,52)));args=p.parse_args()
    if args.compare:
        rows=[compare(c,s) for c in args.cases for s in args.seeds if (OUT/f'{c}_{s}.npz').exists() and json.loads((OUT/f'{c}_{s}.json').read_text()).get('engine','brian2')=='brian2']
        if not rows:raise RuntimeError('No independent Brian2 reference found')
        atomic(OUT/'kernel_comparison.json',dict(trials=rows));raise SystemExit(int(any(r['count_max_error'] or not r['spike_indices_identical'] or r['spike_time_max_error_s']>1e-10 for r in rows)))
    verification=json.loads((OUT/'kernel_comparison.json').read_text())['trials']
    if not verification or any(r['count_max_error'] or not r['spike_indices_identical'] or r['spike_time_max_error_s']>1e-10 for r in verification):raise RuntimeError('Kernel not validated')
    for case in args.cases:
        for seed in args.seeds:run(case,seed)
    print(json.dumps(summary()),flush=True)
