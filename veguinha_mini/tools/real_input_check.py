"""Replay true premotor events, then test the same motor candidate with sensory feedback."""
import argparse
import csv
import json
import os
from pathlib import Path
import time
import numpy as np
from brian2 import start_scope,SpikeGeneratorGroup,Synapses,ms
from veguinha_mini.leg_circuit import build_leg
from veguinha_mini.network import EmittedSpikeMonitor
from veguinha_mini.mk1_validation import atomic
from veguinha_mini.tools.leg_reflex_check import run,require_complete,OUT as LEG
from veguinha_mini.tools.motor_pool_bench import CASES,configure_leg,decode_pool

OUT=LEG.parent/'real_input_v1'
SEEDS=list(range(2001,2011))
CASE='tonic_threshold'


def passes(regular,total):return regular>.8*total


def bin_events(t,i,n):
    bins=np.floor((np.asarray(t)+1e-10)/.02).astype(int)
    valid=(bins>=0)&(bins<1000)
    a=np.zeros((1000,n),int)
    np.add.at(a,(bins[valid],np.asarray(i)[valid]),1)
    return a


def leg_path(seed,feedback=False):
    return LEG/f'{"closedloop" if feedback else "realinput"}_{CASE}_{seed}_n0.5_p2_rail_open.json'


def cv(x):
    x=np.asarray(x)
    return float(x.std()/max(abs(x.mean()),1e-12))


def replay(seed,case):
    path=OUT/f'replay_{case}_{seed}.json'
    if path.exists():return json.loads(path.read_text())
    saved=np.load(leg_path(seed).with_suffix('.npz'))
    start_scope();p=build_leg(seed,coupling=2,motor_strength=12)
    configure_leg(p,case);target=p.motors[2,0];n=len(target)
    oldpre=np.asarray(p.synapses.i[:]);oldpost=np.asarray(p.synapses.j[:]);w=np.asarray(p.synapses.w[:])
    mask=oldpost==target[0];sources=oldpre[mask];weights=w[mask]
    assert len(sources)==22 and len(np.unique(sources))==22
    for j in target:
        sel=oldpost==j
        np.testing.assert_array_equal(oldpre[sel],sources)
        np.testing.assert_array_equal(w[sel],weights)
        assert not np.asarray(p.synapses.slow[:])[sel].any()
    # Preserve the full RNG draw width and original motor initial conditions.
    # All padded cells have no output; only this one pool receives replayed events.
    p.synapses.w=0;p.group.I_base=0;p.group.noise_gain=0;p.group.spike_gate=0
    p.group.I_base[target]=CASES[case]['tonic'];p.group.noise_gain[target]=1;p.group.spike_gate[target]=1
    source_i=saved['event_i'];source_t=saved['event_t'];lookup=np.full(len(p.group),-1,int);lookup[sources]=np.arange(22)
    keep=lookup[source_i]>=0
    source=SpikeGeneratorGroup(22,lookup[source_i[keep]],source_t[keep]*1000*ms,clock=p.group.clock,name='real_premotor_input')
    syn=Synapses(source,p.group,'w:1',on_pre='I_syn_post+=w',clock=p.group.clock,name='real_motor_input')
    syn.connect(i=np.repeat(np.arange(22),n),j=np.tile(target,22));syn.w=np.repeat(weights,n);syn.delay=2*ms
    events=EmittedSpikeMonitor(p.group,'emit',record=True,name='isolated_motor_events')
    p.network.add(source,syn,events)
    started=time.perf_counter();p.network.run(0*ms,namespace={});build=time.perf_counter()-started
    started=time.perf_counter();p.network.run(20000*ms,namespace={});elapsed=time.perf_counter()-started
    require_complete(float(p.network.t/ms)/1000)
    times=np.asarray(events.t/ms)/1000;indices=np.asarray(events.i[:],int)
    lookup=np.full(len(p.group),-1,int);lookup[target]=np.arange(n)
    counts=bin_events(times,lookup[indices],n)
    allcounts=bin_events(source_t,source_i,len(p.group))
    reference=allcounts[:,target]
    identical=bool(np.array_equal(counts,reference)) if case==CASE else None
    if case==CASE:
        if not identical:raise AssertionError('Replayed pool does not match the actual motor output')
        expected=source_t[np.isin(source_i,target)]
        np.testing.assert_allclose(times,expected,atol=1e-10,rtol=0)
    # Data-derived cycle boundaries; no periodic source is generated in this test.
    c=allcounts[:,p.premotor].mean(3);signal=c[:,2,0]-c[:,2,1]
    f=np.fft.fftfreq(1000,.02)
    phase=np.unwrap(np.angle(np.fft.ifft(np.fft.fft(signal)*((f>=2)&(f<=3.5)))))
    starts=np.flatnonzero(np.diff(np.floor((phase-np.pi/2)/(2*np.pi)))>0)
    starts=starts[(starts>=500)&(starts<950)]
    spans=list(zip(starts[:-1],starts[1:]))
    def rates(x):return np.array([x[a:b].sum()/((b-a)*.02) for a,b in spans])
    decoded=decode_pool(counts,120);u=decoded['torque']
    rms=np.array([np.sqrt((u[a:b]**2).mean()) for a,b in spans])
    percell=np.array([counts[a:b].sum(0) for a,b in spans])
    row=dict(seed=seed,case=case,seconds=20,device='controller 25078PC3EG',pool='tibia_positive',
        source_file=str(leg_path(seed)),source_is_actual_events=True,cycles=len(spans),
        input_e_rate_cv=cv(rates(allcounts[:,p.premotor[2,0]].mean(1))),
        input_i_rate_cv=cv(rates(allcounts[:,p.inhibitory[2,1]].mean(1))),
        motor_rate_cv=cv(rates(counts.mean(1))),torque_cv=cv(rms),
        torque_rms_nm=float(np.sqrt((u[500:]**2).mean())*.2),
        torque_amplitude_nm=float(np.diff(np.percentile(u[500:],[5,95]))[0]*.2),
        mean_spikes_per_cell_cycle=float(percell.mean()),fraction_cell_cycles_one_spike=float((percell==1).mean()),
        fraction_cell_cycles_two_spikes=float((percell==2).mean()),
        reference_identical=identical,build_wall_s=build,simulation_wall_s=elapsed)
    np.savez_compressed(path.with_suffix('.npz'),counts=counts,torque=u,input_counts=allcounts[:,sources],cycle_boundaries=starts)
    atomic(path,row)
    print('replay',seed,case,'exact',identical,'E/I/motor CV',*[round(100*row[k],2) for k in ('input_e_rate_cv','input_i_rate_cv','motor_rate_cv')],flush=True)
    return row


def report():
    from veguinha_mini.tools.leg_reflex_report import signal_stats
    groups=[];joint_rows=[]
    for label in ('realinput','closedloop'):
        rows=[json.loads(leg_path(s,label=='closedloop').read_text()) for s in SEEDS if leg_path(s,label=='closedloop').exists()]
        if not rows:continue
        byjoint={j:sum(r['joints'][j]['rhythmic'] for r in rows) for j in ('coxa','femur','tibia')}
        neural=[]
        for r in rows:
            saved=np.load(leg_path(r['seed'],label=='closedloop').with_suffix('.npz'))
            e=saved['counts'][:,48:120].reshape(-1,3,2,12).mean(3)
            e=np.stack([np.convolve(v,np.ones(5)/5,'same') for v in e.reshape(len(e),6).T],1).reshape(-1,3,2)
            neural.append(signal_stats(e[:,:,0]-e[:,:,1]))
            for j,metrics in r['joints'].items():
                joint_rows.append(dict(condition=label,seed=r['seed'],joint=j,**{k:metrics[k] for k in ('rhythmic','measured_period_s','period_cv','amplitude_deg','autocorrelation','drift_fraction','saturation','phase_sd_deg')},support=r['support'],wall_s=r['simulation_wall_s']))
        groups.append(dict(condition=label,n=len(rows),regular=sum(byjoint.values()),total=3*len(rows),byjoint=byjoint,
            passes=bool(len(rows)==10 and passes(sum(byjoint.values()),30)),
            all_three_seeds=sum(all(v['rhythmic'] for v in r['joints'].values()) for r in rows),
            support_mean=float(np.mean([r['support'] for r in rows])),
            premotor_acf=np.mean([r['autocorrelation'] for r in neural],axis=0).tolist(),
            premotor_phase_sd_deg=np.mean([r['phase_sd_deg'] for r in neural],axis=0).tolist(),
            mean_build_wall_s=float(np.mean([r['build_wall_s'] for r in rows])),
            total_simulation_wall_s=float(sum(r['simulation_wall_s'] for r in rows)),
            joints={j:{k:float(np.mean([r['joints'][j][k] for r in rows])) for k in ('autocorrelation','amplitude_deg','saturation','phase_sd_deg')} for j in byjoint}))
    replay_groups=[]
    for case in ('baseline',CASE):
        rows=[json.loads((OUT/f'replay_{case}_{s}.json').read_text()) for s in SEEDS if (OUT/f'replay_{case}_{s}.json').exists()]
        if rows:replay_groups.append(dict(case=case,n=len(rows),all_exact=all(r['reference_identical'] for r in rows) if case==CASE else None,
            **{k:float(np.mean([r[k] for r in rows])) for k in ('input_e_rate_cv','input_i_rate_cv','motor_rate_cv','torque_cv','torque_rms_nm','mean_spikes_per_cell_cycle','fraction_cell_cycles_one_spike','fraction_cell_cycles_two_spikes','simulation_wall_s')}))
    result=dict(seeds=SEEDS,candidate=CASES[CASE],position_gain=4,load_gain=12,neural_dt_ms=.5,physics_dt_ms=2,
        groups=groups,replay=replay_groups,app_changed=False,learning=False,six_legs=False,
        strict_criterion='more than 24/30 regular joints; 24/30 alone fails')
    atomic(OUT/'report.json',result)
    if joint_rows:
        with (OUT/'joint_trials.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(joint_rows[0]));writer.writeheader();writer.writerows(joint_rows)
    print(json.dumps(result),flush=True)
    return result


def main():
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=('source','replay','closedloop','report'),required=True)
    a.add_argument('--seeds',type=int,nargs='+',default=SEEDS);args=a.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    task=dict(title='Reprodução premotora real e perna em malha fechada',detail='Sem novos ajustes: candidato congelado, seeds 2001–2010, física 2 ms; posição 4 e carga 12. Uma execução por vez.')
    atomic(LEG/'task.json',task)
    if not (OUT/'protocol.json').exists():atomic(OUT/'protocol.json',dict(seeds=SEEDS,candidate=CASES[CASE],position_gain=4,load_gain=12,seconds=20,strict_pass_more_than=24,created=time.time()))
    started=time.perf_counter()
    for seed in args.seeds:
        if args.stage=='report':break
        if time.perf_counter()-started>190:print('Budget reached; resume same command.',flush=True);break
        if args.stage in ('source','closedloop'):
            feedback=args.stage=='closedloop'
            if not leg_path(seed,feedback).exists():run(seed,'link',fixture='rail_open',motor_case=CASE,feedback=feedback,record_events=not feedback)
        else:
            for case in ('baseline',CASE):
                atomic(LEG/'progress.json',dict(pid=os.getpid(),profile='real_replay_'+case,seed=seed,state='executando',simulated_s=0,updated=time.time()))
                replay(seed,case)
    report()

if __name__=='__main__':main()
