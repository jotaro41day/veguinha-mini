"""Summarize the structural correction; no simulation or parameter selection."""
import csv,json
import numpy as np
from veguinha_mini.tools.sensory_output_check import OUT,path,CASE,LEG
from veguinha_mini.mk1_validation import atomic


def variability(a,row):
 c=a['counts'];e=c[:,48:120].reshape(-1,3,2,12).mean(3)
 m=c[:,np.asarray(row['motor_indices'])].mean(3)
 y=e[:,0,0]-e[:,0,1];f=np.fft.fftfreq(len(y),.02)
 phase=np.unwrap(np.angle(np.fft.ifft(np.fft.fft(y)*((f>=2)&(f<=3.5)))))
 starts=np.flatnonzero(np.diff(np.floor((phase-np.pi/2)/(2*np.pi)))>0)
 starts=starts[(starts>=500)&(starts<950)];spans=list(zip(starts[:-1],starts[1:]))
 def cv(x):
  values=np.array([x[i:j].sum(0)/((j-i)*.02) for i,j in spans])
  return float(np.mean(values.std(0)/np.maximum(values.mean(0),1e-12)))
 return dict(motor_rate_cv=cv(m),premotor_rate_cv=cv(e))


def main():
 groups=[];joint_rows=[];trials=[];saved={}
 for cohort,seeds,conditions in [('pilot',[2001,2002,2003],('control','reroute','corrected')),('confirmation',list(range(3001,3011)),('control','corrected'))]:
  for condition in conditions:
   rows=[];values=[]
   for seed in seeds:
    f=path(seed,condition!='control',condition=='corrected')
    if not f.exists():continue
    row=json.loads(f.read_text());a=np.load(f.with_suffix('.npz'))
    assert a['angles'].shape==(1000,3) and a['counts'].shape==(1000,196)
    assert row['feedback'] and row['seconds']==20 and row['physics_dt_ms']==2
    assert row['position_gain']==4 and row['load_gain']==12
    assert row.get('sensory_output',False)==(condition!='control')
    assert row.get('motor_excitability',False)==(condition=='corrected')
    v=variability(a,row);rows.append(row);values.append(v);saved[(cohort,condition,seed)]=row
    trials.append(dict(cohort=cohort,condition=condition,seed=seed,regular=sum(j['rhythmic'] for j in row['joints'].values()),support=row['support'],clearance_mm=row['max_clearance_mm'],wall_s=row['simulation_wall_s'],**v))
    for joint,metrics in row['joints'].items():
     joint_rows.append(dict(cohort=cohort,condition=condition,seed=seed,joint=joint,**{k:metrics[k] for k in ('rhythmic','measured_period_s','period_cv','amplitude_deg','autocorrelation','saturation','phase_sd_deg')}))
   if not rows:continue
   def stats(x):
    x=[v for v in x if v is not None]
    return dict(mean=float(np.mean(x)) if x else None,sd=float(np.std(x,ddof=1)) if len(x)>1 else 0.)
   groups.append(dict(cohort=cohort,condition=condition,n=len(rows),regular=sum(v['rhythmic'] for r in rows for v in r['joints'].values()),total=3*len(rows),
    byjoint={j:sum(r['joints'][j]['rhythmic'] for r in rows) for j in ('coxa','femur','tibia')},
    all_three=sum(all(v['rhythmic'] for v in r['joints'].values()) for r in rows),
    support=stats([r['support'] for r in rows]),clearance_mm=stats([r['max_clearance_mm'] for r in rows]),
    motor_rate_cv=stats([v['motor_rate_cv'] for v in values]),premotor_rate_cv=stats([v['premotor_rate_cv'] for v in values]),
    joints={j:{k:stats([r['joints'][j][k] for r in rows]) for k in ('amplitude_deg','autocorrelation','saturation','measured_period_s','period_cv')} for j in ('coxa','femur','tibia')},
    simulation_wall_s=sum(r['simulation_wall_s'] for r in rows)))
 paired=[]
 if all(('confirmation',c,s) in saved for c in ('control','corrected') for s in range(3001,3011)):
  for metric in ('autocorrelation','amplitude_deg'):
   d=np.array([saved[('confirmation','corrected',s)]['joints']['femur'][metric]-saved[('confirmation','control',s)]['joints']['femur'][metric] for s in range(3001,3011)])
   h=2.262157*d.std(ddof=1)/np.sqrt(10);paired.append(dict(metric=metric,mean=float(d.mean()),ci95=[float(d.mean()-h),float(d.mean()+h)]))
 candidate=next((r for r in groups if r['cohort']=='confirmation' and r['condition']=='corrected'),None)
 result=dict(groups=groups,paired_femur=paired,all_trials_integral=True,
  confirmation_passed=bool(candidate and candidate['n']==10 and candidate['regular']>24),
  app_changed=False,full_body_validated=False,learning_started=False,
  protocol=json.loads((OUT/'protocol.json').read_text()))
 if (OUT/'verification.json').exists():result['verification']=json.loads((OUT/'verification.json').read_text())
 atomic(OUT/'report.json',result)
 for name,rows in [('joint_trials.csv',joint_rows),('trial_metrics.csv',trials)]:
  if rows:
   with (OUT/name).open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(result),flush=True)

if __name__=='__main__':main()
