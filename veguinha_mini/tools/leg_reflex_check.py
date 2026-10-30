"""20-second single-leg diagnostics. Never runs a free six-leg body or learning."""
import argparse
import gc
import os
from dataclasses import replace
from pathlib import Path
import time
import numpy as np
from brian2 import start_scope, NetworkOperation, StateMonitor, Clock, ms
from veguinha_mini.leg_circuit import build_leg, FixedTrunk, sense
from veguinha_mini.motor import MotorDecoder
from veguinha_mini.tools.mk1_neuralcheck import trial_config
from veguinha_mini.network import build_network
from veguinha_mini.mk1_validation import atomic,versions

OUT=Path(__file__).resolve().parents[1]/'results/leg_reflex_v1'


def require_complete(seconds,frames=None):
    if abs(seconds-20)>1e-9 or (frames is not None and frames!=1000):
        raise RuntimeError('Interrupted diagnostic: incomplete 20-second trials cannot be committed')


def joint_metrics(angles,torque,dt=.02):
    q=np.asarray(angles)[-500:];u=np.asarray(torque)[-500:]
    y=q-q.mean(axis=0);f=np.fft.fftfreq(len(y),dt)
    spec=np.fft.fft(y,axis=0);band=(f>=.8)&(f<=4)
    peak=np.argmax(abs(spec[band])**2,axis=0);freq=f[band][peak]
    phase=[];result=[]
    for j in range(q.shape[1]):
        filt=(f>=max(.5,freq[j]-.7))&(f<=freq[j]+.7)
        analytic=np.fft.ifft(spec[:,j]*(2*filt))
        phi=np.unwrap(np.angle(analytic));phase.append(phi)
        crosses=np.flatnonzero(np.diff(np.floor(phi[50:-50]/(2*np.pi)))>0)
        periods=np.diff(crosses)*dt
        amp=float(np.percentile(q[:,j],95)-np.percentile(q[:,j],5))
        total=abs(spec[(f>=.5)&(f<=8),j])**2
        concentration=float(abs(spec[band,j][peak[j]])**2/max(total.sum(),1e-20))
        lag=max(1,round((float(periods.mean()) if len(periods)>2 else 1/freq[j])/dt))
        acf=float(np.corrcoef(y[:-lag,j],y[lag:,j])[0,1]) if min(y[:-lag,j].std(),y[lag:,j].std())>1e-12 else 0.
        drift=float(abs(np.polyfit(np.arange(len(q))*dt,y[:,j],1)[0])*len(q)*dt/max(np.ptp(q[:,j]),1e-12))
        fractions=dict(oscillating=0.,stopped=0.,saturated=float(np.mean(abs(u[:,j])>=.98)),irregular=0.)
        for a,b in zip(np.array_split(q[:,j],5),np.array_split(u[:,j],5)):
            g=np.fft.rfftfreq(len(a),dt);p=abs(np.fft.rfft(a-a.mean()))**2
            good=(g>=.8)&(g<=4);wide=(g>=.5)&(g<=8)
            score=p[good].max()/max(p[wide].sum(),1e-20)
            local_acf=float(np.corrcoef(a[:-lag],a[lag:])[0,1]) if lag<len(a)//2 and min(a[:-lag].std(),a[lag:].std())>1e-12 else 0.
            category=('stopped' if np.ptp(a)<.01 else
                      'oscillating' if local_acf>=.6 else 'irregular')
            fractions[category]+=.2
        cv=float(periods.std()/periods.mean()) if len(periods)>2 else None
        measured=float(periods.mean()) if len(periods)>2 and amp>.01 else None
        valid=bool(measured is not None and .3<=measured<=1 and cv<.1
                   and acf>=.7 and drift<.25 and fractions['oscillating']>=.8)
        result.append(dict(period_s=float(periods.mean()) if valid else None,
             measured_period_s=measured,autocorrelation=acf,drift_fraction=drift,
             fft_candidate_period_s=float(1/freq[j]),period_cv=cv,amplitude_rad=amp,
             amplitude_deg=float(np.degrees(amp)),spectral_fraction=concentration,
             saturation=float(np.mean(abs(u[:,j])>=.98)),fractions=fractions,rhythmic=valid))
    phase=np.array(phase)[:,50:-50]
    for j,r in enumerate(result):
        z=np.exp(1j*(phase[j]-phase[0])).mean()
        r['relative_phase_deg']=float(np.angle(z,deg=True))
        r['phase_sd_deg']=float(np.degrees(np.sqrt(-2*np.log(np.clip(abs(z),1e-12,1)))))
    return result


def run(seed,profile,dt_ms=.5,physics_dt_ms=2,coupling=2,motor_gain=.18,fixture='fixed',motor_case=None,feedback=False,record_events=False,sensory_output=False,motor_excitability=False):
    gc.collect();start_scope();start=time.perf_counter();position_gain=0.;load_gain=0.
    case_cfg=None
    if motor_case is not None:
        from veguinha_mini.tools.motor_pool_bench import CASES,configure_leg
        case_cfg=CASES[motor_case]
        if profile!='link':raise ValueError('Motor bench validation uses the link control')
    if (feedback or record_events) and (profile!='link' or motor_case is None):
        raise ValueError('Real-input diagnostics require the unchanged link motor candidate')
    label=(('closedloop_' if feedback else 'realinput_' if record_events else 'poolleg_')+motor_case) if motor_case else profile
    if sensory_output:label='sensory_output_'+label
    if motor_excitability:label='excitability_'+label
    started_at=time.time()
    def progress(state,seconds=0):
        atomic(OUT/'progress.json',dict(pid=os.getpid(),state=state,profile=label,
            seed=seed,simulated_s=seconds,started_at=started_at,updated=time.time()))
    progress('preparando')
    if profile in ('official','legacy'):
        c=trial_config(seed,'coupled',isolated=True)
        regions=tuple(r for r in c.regions if not r.name.startswith('vnc_') or r.name=='vnc_lf')
        names={r.name for r in regions}
        c=replace(c,regions=regions,phase5_enabled=False,dt_ms=dt_ms,
                  motor_scaffold=profile=='legacy',
                  pathways=tuple(p for p in c.pathways if p.source in names and p.target in names))
        p=build_network(c,record_spikes=False);config=c;record=p.regions['vnc_lf']
    else:
        k=0 if profile in ('core','motor','reference2') else coupling
        p=build_leg(seed,joints=2 if profile=='reference2' else 3,coupling=k,dt_ms=dt_ms,
                    motor_strength=4 if profile in ('core','reference2') else 12,sensory_output=sensory_output,motor_excitability=motor_excitability,
                    position_gate=profile in ('gated','trace','coherent','load','shared','quiet','reset','loadonly','brake','brake_sham'),
                    gate_memory=profile in ('trace','coherent','load','shared','quiet','reset','loadonly','brake','brake_sham'),
                    coherent_reset=profile in ('coherent','load','shared','quiet','reset','loadonly','brake','brake_sham'),shared_inhibition=profile=='shared',reset_inhibition=profile=='reset',load_brake=profile in ('brake','brake_sham'),motor_heterogeneity=profile=='motorgraded',motor_size=8 if case_cfg is None else case_cfg['pool'])
        if profile=='quiet':p.group.noise=0
        if profile in ('motorquiet','motorgraded'):p.group.noise_gain[p.motors.ravel()]=0
        if profile=='brake_sham':p.synapses.w[p.kinds=='load_brake']=0
        if motor_case is not None:configure_leg(p,motor_case)
        config=p.config;record=np.arange(len(p.group))
        position_gain=12 if profile=='position' else 4 if profile in ('gated','trace','coherent','load','shared','quiet','reset','brake','brake_sham') else 0
        load_gain=12 if profile in ('load','shared','quiet','reset','loadonly','brake','brake_sham') else 0
    if feedback:position_gain=4;load_gain=12
    if profile in ('official','legacy'):config=replace(config,motor_gain=motor_gain)
    else:config.motor_gain=motor_gain
    body=FixedTrunk(dt=physics_dt_ms/1000,horizontal_free=fixture.startswith('rail'),walls=fixture!='rail_open')
    decoder=MotorDecoder(p,config,legs=('lf',))
    voltage=StateMonitor(p.group,'v',record=record,dt=1*ms,when='after_groups',name='leg_voltage')
    p.network.add(voltage);data=[];previous=np.zeros(len(p.group),dtype=np.int64)
    if record_events:
        from veguinha_mini.network import EmittedSpikeMonitor
        events=EmittedSpikeMonitor(p.group,'emit',record=True,name='real_input_events')
        p.network.add(events)

    def tick():
        nonlocal previous
        count=np.asarray(p.monitor.count[:],dtype=np.int64)
        command=decoder.update(count,.02)[0]
        applied=command if len(command)==3 else np.r_[0.,command]
        support,clearance=body.step(applied)
        q=body.angles-body.neutral
        if profile not in ('official','legacy'):
            sense(p,q,body.load,position_gain,load_gain)
        else:
            # The official phase-4 encoding, restricted to the present leg.
            currents=np.zeros(len(p.group))
            for j,name in enumerate(config.joint_names):
                currents[p.motor_pools['lf'][name+'_flex']]+=2*max(0,-body.angles[j])
                currents[p.motor_pools['lf'][name+'_extend']]+=2*max(0,body.angles[j])
            currents[p.interneurons['lf']['inhibitory']]+=float(body.load>.02)
            p.group.I_sense=currents
        data.append(dict(angles=q[-len(command):].copy(),torque=command.copy(),
                         support=support,clearance=clearance,load=body.load,
                         root_position=body.data.xpos[body.root].copy(),
                         wall_contact=any(int(c.geom1) in body.wall_ids or int(c.geom2) in body.wall_ids for c in body.data.contact),
                         rates=decoder.filtered_hz[0].copy(),counts=(count-previous)[record].copy()))
        previous=count.copy()
        if len(data)%250==0:progress('executando',len(data)*.02)
    p.network.add(NetworkOperation(tick,clock=Clock(dt=20*ms,name='leg_block_clock'),when='end',name='leg_tick'))
    print('compiling',profile,seed,flush=True)
    progress('compilando')
    p.network.run(0*ms);built=time.perf_counter()
    print('running',profile,seed,'build_s',round(built-start,2),flush=True)
    progress('executando')
    p.network.run(20000*ms)
    require_complete(float(p.network.t/ms)/1000,len(data))
    elapsed=time.perf_counter()-built
    arrays={k:np.asarray([d[k] for d in data]) for k in data[0]}
    if record_events:
        arrays.update(event_t=np.asarray(events.t/ms)/1000,event_i=np.asarray(events.i[:],int))
    metrics=joint_metrics(arrays['angles'],arrays['torque'])
    row=dict(seed=seed,profile=label,feedback=feedback,record_events=record_events,sensory_output=sensory_output,motor_excitability=motor_excitability,metric_version=2,device='controller 25078PC3EG',versions=versions(),
         seconds=20,dt_ms=dt_ms,physics_dt_ms=physics_dt_ms,cells=len(p.group),motor_gain=motor_gain,fixture=fixture,
         motor_case=motor_case,motor_case_config=case_cfg,motor_pool_size=decoder.pool_size,
         motor_indices=p.motors.tolist() if hasattr(p,'motors') else None,
         coupling=coupling if profile not in ('core','motor','reference2','official','legacy') else 0,
         motor_strength=4 if profile in ('core','reference2') else 12 if profile not in ('official','legacy') else None,
         position_gain=position_gain,load_gain=load_gain,joints=dict(zip(config.joint_names,metrics)),
         support=float(arrays['support'][-500:].mean()),
         max_clearance_mm=float(arrays['clearance'][-500:].max()*1000),
         air_fraction=float((arrays['clearance'][-500:]>.002).mean()),
         wall_contact=bool(arrays['wall_contact'].any()),
         motor_rates_hz=arrays['rates'][-500:].mean(axis=0).reshape(-1,2).tolist(),
         rate_max_hz=float(arrays['counts'][-500:].sum(axis=0).max()/10),
         voltage_min=float(np.min(voltage.v)),voltage_max=float(np.max(voltage.v)),
         build_wall_s=built-start,simulation_wall_s=elapsed)
    name=f'{profile}_{seed}_n{dt_ms:g}_p{physics_dt_ms:g}'
    if motor_case:name=f'{label}_{seed}_n{dt_ms:g}_p{physics_dt_ms:g}'
    if coupling!=2:name+=f'_k{coupling:g}'
    if motor_gain!=.18:name+=f'_g{motor_gain:g}'
    if fixture!='fixed':name+='_'+fixture
    OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT/(name+'.npz'),**arrays)
    atomic(OUT/(name+'.json'),row)
    progress('concluído',20)
    print(name,[(k,round(v['amplitude_deg'],2),v['period_s'],v['rhythmic']) for k,v in row['joints'].items()],
          'support',round(row['support'],3),'lift mm',round(row['max_clearance_mm'],2),
          'wall',round(elapsed,2),flush=True)
    return row


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--profile',choices=('official','legacy','reference2','core','motor','link','position','gated','trace','coherent','load','shared','quiet','reset','loadonly','brake','brake_sham','motorquiet','motorgraded'),required=True)
    a.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);a.add_argument('--dt',type=float,default=.5)
    a.add_argument('--physics-dt',type=int,default=2);a.add_argument('--coupling',type=float,default=2)
    a.add_argument('--motor-gain',type=float,default=.18)
    a.add_argument('--fixture',choices=('fixed','rail','rail_open'),default='fixed')
    a.add_argument('--motor-case',default=None)
    c=a.parse_args()
    for seed in c.seeds:run(seed,c.profile,c.dt,c.physics_dt,c.coupling,c.motor_gain,c.fixture,c.motor_case)
