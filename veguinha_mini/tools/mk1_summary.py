"""Summaries of paired seeds; never pool frames as independent animals."""
import argparse
import json
from pathlib import Path
import numpy as np


def ci(values):
    x=np.asarray(values,dtype=float)
    if not len(x):return dict(n=0,mean=None,sd=None,ci95=None)
    mean=float(x.mean())
    if len(x)<2:return dict(n=1,mean=mean,sd=None,ci95=None)
    t={2:12.706205,3:4.302653,4:3.182446,5:2.776445,6:2.570582,
       7:2.446912,8:2.364624,9:2.306004,10:2.262157,11:2.228139,
       20:2.093024,30:2.045230}.get(len(x))
    sd=float(x.std(ddof=1))
    if t is None:return dict(n=len(x),mean=mean,sd=sd,ci95=None)
    half=t*sd/np.sqrt(len(x))
    return dict(n=len(x),mean=mean,sd=sd,ci95=[mean-half,mean+half])


def mechanics(root):
    rows=json.loads((root/'mechanics.json').read_text())['rows']
    selected={c:sorted([r for r in rows if r['condition']==c and not r['mirror']
                       and r['dt_ms']==2 and r['impulse'] is None],key=lambda r:r['seed'])
              for c in ('passive','random','tripod')}
    groups={}
    for name,group in selected.items():
        groups[name]=dict(forward=ci([r['forward_m_s'] for r in group]),
                         tilt=ci([r['tilt_deg'] for r in group]),
                         falls=sum(r['fall'] for r in group),
                         support=ci([np.mean(r['support']) for r in group]),
                         alternation=ci([r['alternation'] for r in group]),
                         three_feet=ci([r['three_feet'] for r in group]),
                         late_distance=ci([r['late_distance_m'] for r in group]))
    rand={r['seed']:r for r in selected['random']};trip={r['seed']:r for r in selected['tripod']}
    pairs=sorted(rand.keys() & trip.keys())
    difference=ci([trip[s]['forward_m_s']-rand[s]['forward_m_s'] for s in pairs])
    mirror=[]
    for r in rows:
        if r['mirror'] and r['seed'] in rand:
            mirror.append(r['forward_m_s']+rand[r['seed']]['forward_m_s'])
    convergence={}
    for dt in (4,1):
        for condition in ('random','tripod'):
            base={r['seed']:r for r in selected[condition]}
            variants=[r for r in rows if r['condition']==condition and r['dt_ms']==dt
                      and not r['mirror'] and r['impulse'] is None]
            convergence[f'{condition}_{dt}_minus_2']=ci([r['forward_m_s']-base[r['seed']]['forward_m_s']
                                                        for r in variants if r['seed'] in base])
    impulses=[r for r in rows if r['impulse'] is not None]
    return dict(groups=groups,paired_tripod_minus_random=difference,
                mirror_velocity_sum=ci(mirror),convergence=convergence,
                impulses=dict(n=len(impulses),recovered=sum(r['recovery_s'] is not None for r in impulses),
                              max_final_speed=max((r['final_speed_m_s'] for r in impulses),default=None)),
                seed_unit='one 20 s trial per seed and condition; late 10 s metrics')


def newborn(root):
    rows=[json.loads(p.read_text()) for p in root.glob('*.json') if p.name!='summary.json']
    rows=[r for r in rows if 'condition' in r]
    groups={}
    for cond in ('coupled','uncoupled','shuffled','old'):
        g=[r for r in rows if r['condition']==cond]
        if not g:continue
        groups[cond]=dict(forward=ci([r['forward_m_s'] for r in g]),
                          tilt=ci([r['tilt_deg'] for r in g]),falls=sum(r['fall'] for r in g),
                          viable=sum(r['viable'] for r in g),tripod=sum(r['tripod'] for r in g),
                          rhythmic_leg_fraction=ci([r['oscillating_leg_fraction'] for r in g]),
                          saturation=ci([r['saturation'] for r in g]),
                          wall_s_per_sim_s=ci([r['wall_s_per_sim_s'] for r in g]))
    coupled={r['seed']:r for r in rows if r['condition']=='coupled'}
    paired={}
    for cond in ('uncoupled','shuffled','old'):
        control={r['seed']:r for r in rows if r['condition']==cond}
        seeds=sorted(coupled.keys() & control.keys())
        paired[cond]=ci([coupled[s]['forward_m_s']-control[s]['forward_m_s'] for s in seeds])
    return dict(groups=groups,paired=paired,
                viability_pass=len(coupled)>=10 and all(r['viable'] for r in coupled.values()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=('mechanics','newborn'));p.add_argument('path',type=Path)
    a=p.parse_args();r={'mechanics':mechanics,'newborn':newborn}[a.kind](a.path)
    (a.path/'summary.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
