"""Substep convergence and contact audit; no learning or prescribed neural phases."""
import argparse
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pyximport

from .body import Body, LEG_NAMES
from .world import World
from .phase_lock_audit import _save_atomic
from .tripod_body_diagnostic import TRIPOD_PHASE, PERIOD_S, _phase_contact_r, _phase_from_counts
from .metrics import NEIGHBORS

RESULTS = Path(__file__).parent / 'results'


def kernel():
    pyximport.install(setup_args={'include_dirs': np.get_include()}, language_level=3)
    from .fast_body import advance
    return advance


def summarize(body, origin, logs, dt, seconds, block_torques=None, counts=None):
    contacts = np.concatenate([x['contacts'] for x in logs])
    impulses = np.concatenate([x['impulses'] for x in logs])
    forward = np.concatenate([x['forward'] for x in logs])
    final = slice(-round(min(20, seconds/2)*1/dt), None)
    c = contacts[final]
    corr = lambda a,b: float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-9 and np.std(b)>1e-9 else 0.
    alternation = -np.mean([corr(c[:,a], c[:,b]) for a,b in NEIGHBORS])
    together = (np.all(c[:,[0,2,4]], axis=1).mean() + np.all(c[:,[3,1,5]], axis=1).mean())/2
    impulse_cycle = impulses[final].sum() / (len(c)*dt/PERIOD_S)
    result = dict(displacement=float(np.linalg.norm(body.position-origin)),
                  forward_speed=float(forward.mean()),
                  forward_distance=float(forward.sum()*dt),
                  final_forward_speed=float(forward[final].mean()),
                  contact_fraction=float(c.mean()), contacts_by_leg=c.mean(axis=0).tolist(),
                  all_three=float(together), alternation=float(alternation),
                  impulse_per_cycle=float(impulse_cycle), energy_spent=float(1-body.energy),
                  position=body.position.tolist())
    return result


def imposed(seed, dt, seconds=120, lag_deg=90, mode=0):
    advance = kernel()
    calibration = json.loads((RESULTS/'tripod_body_diagnostic_v2.json').read_text())['rows']
    reference = {r['seed']:r for r in calibration if r['condition']=='rede18'}
    # Additional seeds use the fixed population mean amplitude, without fitting to outcomes.
    amp = np.array(reference[seed]['calibration_amp'] if seed in reference else
                   np.mean([r['calibration_amp'] for r in reference.values()],axis=0))
    world = World(); body = Body(world); body.vertical_support = mode == 2
    origin = body.position.copy()
    logs = []
    for b in range(round(seconds/.02)):
        phase = 2*math.pi*b*.02/PERIOD_S + TRIPOD_PHASE
        torque = np.column_stack((amp[0]*np.sin(phase),
                                  amp[1]*np.sin(phase+math.radians(lag_deg))))
        logs.append(advance(body, world, dt, torque, round(.02/dt), mode))
    return summarize(body,origin,logs,dt,seconds)


def neural(seed, condition='rede18', seconds=40, params=None, trace_path=None):
    from .search_fast import FastEvaluator, LOCAL_OSCILLATOR
    from .motor import MotorDecoder, POOL_NAMES
    from .config import Config
    params = params or {}
    advance = kernel(); dt=.002
    evaluator = FastEvaluator(seed=seed, cross_force=0 if condition=='controle' else params.get('cross_force',18),
                              normalize_neighbors=True, cross_tau_ms=40)
    weights = evaluator.build_weights({**LOCAL_OSCILLATOR,'coupling':0.,
        'motor':params.get('motor_recruitment',LOCAL_OSCILLATOR['motor']),
        'motor_inhibition':params.get('motor_inhibition',LOCAL_OSCILLATOR['motor_inhibition'])})
    if params.get('joint_gain',0):
        weights = joint_links(evaluator,weights,params['joint_gain'])
    model = SimpleNamespace(group=evaluator.probe.neurons,
        motor_pools={leg:{name:evaluator.probe.pools[k].reshape(4,8)[i]
                         for i,name in enumerate(POOL_NAMES)} for k,leg in enumerate(LEG_NAMES)})
    decoder=MotorDecoder(model,Config(motor_gain=params.get('motor_gain',.18),
        motor_filter_ms=params.get('motor_filter_ms',120.)))
    world=World(); body=Body(world); body.vertical_support=True; origin=body.position.copy()
    state=None; previous=np.zeros(768,dtype=np.int64)
    rng=np.random.default_rng(seed+9000); permutation=rng.permutation(6)
    while np.any(permutation==np.arange(6)): permutation=rng.permutation(6)
    logs=[]; counts=[]; torques=[]; loads=[]
    for b in range(round(seconds/.02)):
        proprio_angles=body.angles.copy()
        proprio_angles[:,1] += params.get('knee_proprio_offset',0.)
        _,_,_,_,state=evaluator.simulate(evaluator.starts,evaluator.targets,weights,6,20,seed,
            params.get('drive',LOCAL_OSCILLATOR['drive']),
            params.get('proprio',LOCAL_OSCILLATOR['proprio']),
            params.get('noise_amp',2.5),state=state,
            external_angles=proprio_angles.ravel(),external_contacts=body.contacts.astype(float),
            return_state=True,edge_kinds=evaluator.kinds,cross_tau_ms=40,
            external_loads=np.clip(body.normal_load/(body.mass*body.gravity/6),0,2),
            load_reflex_gain=params.get('load_gain',0.),
            sensory_bias=params.get('sensory_bias',1.),
            load_reflex_knee_action=params.get('load_reflex_knee_action',0),
            inhibitory_bias=LOCAL_OSCILLATOR['inhibitory_bias'],
            motor_bias=params.get('motor_bias',0.),
            adaptation_a=LOCAL_OSCILLATOR['adaptation_a'],adaptation_d=LOCAL_OSCILLATOR['adaptation_d'])
        delta=state['total']-previous; previous=state['total'].copy()
        counts.append(delta[evaluator.probe.premotor].sum(axis=-1))
        torque=decoder.update(state['total'],.02)
        if condition=='embaralhado': torque=torque[permutation]
        torques.append(torque.copy()); loads.append(body.normal_load.copy())
        logs.append(advance(body,world,dt,torque,10,2))
    result=summarize(body,origin,logs,dt,seconds)
    counts=np.asarray(counts); contacts=np.concatenate([x['contacts'] for x in logs])
    phases=[]
    for j in range(2):
        phases.append(_phase_from_counts(counts[:,:,j:j+1,:]))
    usable=slice(round((seconds-min(20,seconds/2)+2)/.02),-50)
    phase_vector=np.exp(1j*(phases[1][usable]-phases[0][usable])).mean(axis=0)
    lag=np.angle(phase_vector,deg=True)
    t_neural=(np.arange(len(counts))+.5)*.02
    t_body=(np.arange(len(contacts))+.5)*dt
    interp=np.column_stack([np.interp(t_body,t_neural,np.unwrap(phases[0][:,k])) for k in range(6)])
    mask=(t_body>=seconds-min(20,seconds/2)+2)&(t_body<seconds-1)
    result['joint_phase_deg']=lag.tolist()
    result['joint_phase_lock_r']=np.abs(phase_vector).tolist()
    result['phase_contact_r']=[_phase_contact_r(interp[mask,k],contacts[mask,k]) for k in range(6)]
    result['motor_saturation']=decoder.saturation_fraction()
    sensory_ids=np.arange(768).reshape(6,128)[:,120:128]
    result['sensory_hz']=float(state['total'][sensory_ids].mean()/seconds)
    result['permutation']=permutation.tolist() if condition=='embaralhado' else None
    if trace_path:
        np.savez_compressed(trace_path,counts=counts,torques=np.asarray(torques),contacts=contacts,
                            normal_load=np.asarray(loads),neural_phases=np.asarray(phases))
    return result


def joint_links(evaluator, weights, gain):
    """Local disynaptic ring; retains all old weights and obeys Dale's law.

    Hip I inhibits homologous knee E; knee I inhibits opposite hip E.
    The circuit has no phase variable or target period. Source spikes alone
    deliver delayed inhibitory current through the existing 40-ms synapses.
    """
    source=np.repeat(np.arange(768),np.diff(evaluator.starts)).tolist()
    target=evaluator.targets.tolist(); kind=evaluator.kinds.tolist(); w=weights.tolist()
    rng=np.random.default_rng(evaluator.seed+17000)
    new_source=[]; new_target=[]
    for leg in range(6):
        for joint in range(2):
            for action in range(2):
                target_action=action if joint==0 else 1-action
                for cell in evaluator.probe.inhibitory[leg,joint,action]:
                    dst=rng.choice(evaluator.probe.premotor[leg,1-joint,target_action],3,replace=False)
                    new_source.extend([int(cell)]*3); new_target.extend(dst.tolist())
    degree=np.maximum(1,np.bincount(new_target,minlength=768))
    source.extend(new_source); target.extend(new_target); kind.extend([7]*len(new_target))
    w.extend((-gain/np.sqrt(degree[new_target])).tolist())
    order=np.argsort(source,kind='stable')
    evaluator.starts=np.r_[0,np.cumsum(np.bincount(source,minlength=768))].astype(np.int32)
    evaluator.targets=np.asarray(target,dtype=np.int32)[order]
    evaluator.kinds=np.asarray(kind,dtype=np.int8)[order]
    return np.asarray(w,dtype=float)[order]


def run(stage, seeds, budget=230):
    target = RESULTS/f'contact_{stage}_v1.json'
    output = json.loads(target.read_text()) if target.exists() else {'stage':stage,'rows':[]}
    done = {(r['seed'],r['dt'],r.get('lag_deg',90),r.get('condition','imposto')) for r in output['rows']}
    started = time.monotonic()
    cases = [(s,d,90,'imposto') for s in seeds for d in (.02,.01,.005,.002)]
    if stage in ('sweep','loaded_sweep'):
        cases = [(s,.002,lag,'imposto') for lag in range(0,360,30) for s in seeds]
    elif stage == 'network':
        cases=[(s,.002,90,c) for s in seeds for c in ('rede18','controle','embaralhado')]
    elif stage == 'pilot':
        cases=[(s,.002,90,c) for c in ('original','j3_l1','j6_l1','j10_l1','j6_l2') for s in seeds]
    elif stage == 'calibration':
        cases=[(s,.002,90,c) for c in ('gain032','gain027') for s in seeds]
    elif stage not in ('convergence','contact_convergence','loaded_convergence'): raise ValueError(stage)
    for seed,dt,lag,condition in cases:
        if (seed,dt,lag,condition) in done: continue
        if time.monotonic()-started > budget: break
        t=time.monotonic()
        if stage in ('network','pilot','calibration'):
            if stage=='pilot':
                candidates={'original':(0,0),'j3_l1':(3,1),'j6_l1':(6,1),'j10_l1':(10,1),'j6_l2':(6,2)}
                g,l=candidates[condition]; params={'joint_gain':g,'load_gain':l}
            elif stage=='calibration':
                params={'joint_gain':10,'load_gain':1,
                        'motor_gain':.32 if condition=='gain032' else .27}
            else:
                params=json.loads((RESULTS/'contact_selected_v1.json').read_text())['params']
            result=neural(seed,condition if stage=='network' else 'rede18',
                          seconds=40 if stage=='network' else 20,params=params,
                          trace_path=RESULTS/f'contact_{stage}_{seed}_{condition}.npz')
            result['params']=params
        else:
            mode=2 if stage.startswith('loaded') else int(stage!='convergence')
            result=imposed(seed,dt,lag_deg=lag,mode=mode)
        row={'seed':seed,'dt':dt,'lag_deg':lag,'condition':condition,**result}
        row['wall_s']=time.monotonic()-t
        output['rows'].append(row)
        _save_atomic(target,output)
        print(json.dumps(row),flush=True)
    print(json.dumps({'completed':len(output['rows']),'wall_s':time.monotonic()-started}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('stage'); p.add_argument('--seeds',nargs='+',type=int,default=list(range(42,52)))
    p.add_argument('--budget',type=float,default=230)
    args=p.parse_args(); run(args.stage,args.seeds,args.budget)
