"""Summarize bounded one-leg diagnostics without running another simulation."""
import csv
import json
from pathlib import Path
import numpy as np
from veguinha_mini.tools.leg_reflex_check import OUT
from veguinha_mini.mk1_validation import atomic


def stats(values):
    a=np.asarray(values,float)
    return dict(mean=float(a.mean()),sd=float(a.std(ddof=1)) if len(a)>1 else 0.)


def signal_stats(series):
    x=np.asarray(series)[-500:];x=x-x.mean(0)
    f=np.fft.fftfreq(len(x),.02);spec=np.fft.fft(x,axis=0)
    band=(f>=1)&(f<=4);peak=f[band][np.argmax(abs(spec[band])**2,axis=0)]
    phase=[];acf=[]
    for j,p in enumerate(peak):
        k=round(1/p/.02)
        acf.append(float(np.corrcoef(x[:-k,j],x[k:,j])[0,1]))
        mask=(f>=max(.5,p-.7))&(f<=p+.7)
        phase.append(np.angle(np.fft.ifft(spec[:,j]*mask))[50:-50])
    z=np.exp(1j*(np.asarray(phase)-phase[0])).mean(1)
    return dict(peak_period_s=(1/peak).tolist(),autocorrelation=acf,
                relative_phase_deg=np.angle(z,deg=True).tolist(),
                phase_sd_deg=np.degrees(np.sqrt(-2*np.log(np.clip(abs(z),1e-12,1)))).tolist())


def main():
    cases=[];rows=[]
    for profile in ('official','reference2','link','load','shared','quiet','reset','coherent','loadonly','brake','brake_sham','motorquiet','motorgraded'):
        for seed in (42,43,44):
            dt=1 if profile=='official' else .5
            path=OUT/f'{profile}_{seed}_n{dt:g}_p2_rail_open.json'
            if not path.exists():continue
            r=json.loads(path.read_text());a=np.load(path.with_suffix('.npz'))
            n=len(r['joints']);entry=dict(profile=profile,seed=seed,
                support=r['support'],lift_mm=r['max_clearance_mm'],wall_s=r['simulation_wall_s'],
                build_wall_s=r['build_wall_s'],joint_metrics=r['joints'],
                torque=signal_stats(a['torque']),angle=signal_stats(a['angles']))
            if profile!='official':
                e=a['counts'][:,16*n:40*n].reshape(-1,n,2,12).sum(3)/12/.02
                e=np.stack([np.convolve(v,np.ones(5)/5,'same') for v in e.reshape(len(e),-1).T],1).reshape(-1,n,2)
                entry['premotor']=signal_stats(e[:,:,0]-e[:,:,1])
            cases.append(entry)
            for j,v in r['joints'].items():
                rows.append(dict(profile=profile,seed=seed,joint=j,period_s=v['measured_period_s'],
                    amplitude_deg=v['amplitude_deg'],phase_deg=v['relative_phase_deg'],
                    phase_sd_deg=v['phase_sd_deg'],period_cv=v['period_cv'],
                    autocorrelation=v['autocorrelation'],oscillating=v['fractions']['oscillating'],
                    stopped=v['fractions']['stopped'],saturated=v['saturation'],rhythmic=v['rhythmic']))
    replay=json.loads((OUT/'replay_summary.json').read_text())['trials'];convergence=[]
    for dt in (4,2,1):
        rs=[r for r in replay if r['contact'] and r['physics_dt_ms']==dt]
        convergence.append(dict(dt_ms=dt,support=stats([r['support'] for r in rs]),
            amplitude_deg={j:stats([r['joints'][j]['amplitude_deg'] for r in rs]) for j in ('coxa','femur','tibia')},
            wall_s=sum(r['simulation_wall_s'] for r in rs)))
    exact={}
    for seed in (42,43,44):
        stem=f'link_{seed}_n0.5_p2_rail_open'
        a=np.load(OUT/(stem+'.npz'))['angles'];b=np.load(OUT/(stem+'_replay_p2.npz'))['angles']
        exact[seed]=float(np.max(abs(a-b)))
    contact={}
    for state in (True,False):
        rs=[r for r in replay if r['physics_dt_ms']==2 and r['contact']==state]
        contact[str(state)]=dict(regular_joints=sum(v['rhythmic'] for r in rs for v in r['joints'].values()),
            femur_autocorrelation=stats([r['joints']['femur']['autocorrelation'] for r in rs]))
    report=dict(device='controller 25078PC3EG',seeds=[42,43,44],seconds=20,window_s=[10,20],
        scope='One actuated LF leg, XY guide with fixed height/orientation, no walls; no learning.',
        cases=cases,convergence=convergence,exact_replay_max_error_rad=exact,contact_comparison=contact,
        motor_max_nm=[.08,.2,.2],promoted_to_app=False,one_leg_gate_passed=False,
        conclusion='Open-loop neural rhythm reaches torque. Identical torque replay passes 9/9 angles without floor, 6/9 with floor. Feedback candidates did not restore the gate. No six-leg or learning promotion.')
    atomic(OUT/'diagnostic_summary.json',report)
    with (OUT/'joint_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('saved',len(cases),'cases;',len(rows),'joint rows; contact',contact)


if __name__=='__main__':main()
