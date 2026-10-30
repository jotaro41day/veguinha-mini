"""Export the complete motor-pool experiment without running a simulation."""
import csv
import json
from pathlib import Path
import numpy as np
from veguinha_mini.tools.motor_pool_bench import CASES,OUT,summary
from veguinha_mini.tools.motor_pool_validate import path,report,actual_variability
from veguinha_mini.mk1_validation import atomic


def write_csv(name,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def main():
    pool=summary();pilot=report([42,43,44]);held=report(list(range(1001,1011)))
    trials=[]
    for case in CASES:
        for seed in range(42,52):
            p=OUT/f'{case}_{seed}.json'
            if not p.exists():continue
            r=json.loads(p.read_text());trials.append(dict(case=case,seed=seed,**r['config'],
                input_cv=r['input_rate_cv'],motor_cv=r['rate_cv'],torque_cv=r['torque_cv'],
                torque_rms_nm=r['torque_rms_nm'],torque_amplitude_nm=r['torque_amplitude_nm'],
                mean_motor_hz=r['mean_rate_hz'],saturation=r['saturation'],engine=r.get('engine','brian2'),wall_s=r['simulation_wall_s']))
    write_csv('pool_trials.csv',trials)
    table=[]
    for p in pool:
        leg=next((r for r in pilot if r['case']==p['case']),None)
        table.append(dict(case=p['case'],parent=CASES[p['case']]['parent'],pool=CASES[p['case']]['pool'],
            filter_ms=CASES[p['case']]['filter_ms'],motor_cv=p['rate_cv']['mean'],motor_cv_sd=p['rate_cv']['sd'],
            torque_cv=p['torque_cv']['mean'],torque_cv_sd=p['torque_cv']['sd'],torque_rms_nm=p['torque_rms_nm']['mean'],
            torque_amplitude_nm=p['torque_amplitude_nm']['mean'],torque_amplitude_sd=p['torque_amplitude_nm']['sd'],regular=leg['regular'] if leg else None,
            total=leg['total'] if leg else None,min_movement_ratio=min(leg['amplitude_ratio']) if leg else None))
    write_csv('condition_summary.csv',table)
    physical=[]
    for cohort,seeds,cases in (('pilot',[42,43,44],list(CASES)),('confirmation',list(range(1001,1011)),['baseline','tonic_threshold'])):
        for case in cases:
            for seed in seeds:
                p=path(case,seed)
                if not p.exists():continue
                r=json.loads(p.read_text());v=actual_variability(case,seed)
                row=dict(cohort=cohort,case=case,seed=seed,regular=sum(x['rhythmic'] for x in r['joints'].values()),
                    motor_count_cv=float(np.mean(v['motor_cv'])),premotor_count_cv=float(np.mean(v['premotor_cv'])),
                    motor_rate_cv=float(np.mean(v['motor_rate_cv'])),premotor_rate_cv=float(np.mean(v['premotor_rate_cv'])),
                    torque_cv=float(np.mean(v['torque_cv'])),support=r['support'],wall_s=r['simulation_wall_s'],
                    engine=r.get('engine','Brian2+MuJoCo'),replay_source=r.get('replay_source',''))
                for joint,metrics in r['joints'].items():
                    row[joint+'_regular']=metrics['rhythmic'];row[joint+'_amplitude_deg']=metrics['amplitude_deg']
                    row[joint+'_saturation']=metrics['saturation'];row[joint+'_acf']=metrics['autocorrelation'];row[joint+'_period_s']=metrics['period_s'];row[joint+'_candidate_period_s']=metrics['measured_period_s']
                physical.append(row)
    write_csv('physical_trials.csv',physical)
    candidate=next((r for r in held if r['case']=='tonic_threshold'),None)
    independent=json.loads((OUT/'kernel_comparison.json').read_text())['trials']
    from veguinha_mini.tools.leg_reflex_report import signal_stats
    upstream=[]
    for seed in [42,43,44]+list(range(1001,1011)):
        f=path('tonic_threshold',seed)
        if not f.exists():continue
        counts=np.load(f.with_suffix('.npz'))['counts'];e=counts[:,48:120].reshape(-1,3,2,12).mean(3)
        e=np.stack([np.convolve(v,np.ones(5)/5,'same') for v in e.reshape(len(e),6).T],1).reshape(-1,3,2)
        upstream.append(dict(seed=seed,femur_regular=json.loads(f.read_text())['joints']['femur']['rhythmic'],**signal_stats(e[:,:,0]-e[:,:,1])))
    atomic(OUT/'upstream_phase_audit.json',dict(trials=upstream))
    result=dict(device='controller 25078PC3EG',pool_seeds=list(range(42,52)),pilot_seeds=[42,43,44],
        confirmation_seeds=list(range(1001,1011)),pool_trials=len(trials),pool=table,pilot=pilot,confirmation=held,
        confirmed_kinematics=bool(candidate and candidate['regular']==30),
        promoted_to_app=False,six_legs_started=False,learning_started=False,
        confirmation_seed_successes={c:sum(r['regular']==3 for r in physical if r['cohort']=='confirmation' and r['case']==c) for c in ('baseline','tonic_threshold')},
        holdout_parameters_unchanged=all(json.loads(path('tonic_threshold',s).read_text())['motor_case_config']==json.loads((OUT/'selection.json').read_text())['parameters'] for s in range(1001,1011)),
        status='No promotion: independent confirmation failed' if not candidate or candidate['regular']!=30 else 'Kinematic criterion passed; inspect variability before promotion',
        kernel_reference_trials=len(independent),kernel_max_count_error=max(r['count_max_error'] for r in independent),
        kernel_max_time_error_s=max(r['spike_time_max_error_s'] for r in independent),
        scope='One LF leg, guided trunk, position/load afferents disabled to isolate motor output; controlled periodic input only in the separate pool benchmark.')
    atomic(OUT/'report.json',result)
    print('pool trials',len(trials),'pilot',[(r['case'],r['regular']) for r in pilot],
          'confirmation',[(r['case'],r['regular'],r['total']) for r in held],flush=True)


if __name__=='__main__':main()
