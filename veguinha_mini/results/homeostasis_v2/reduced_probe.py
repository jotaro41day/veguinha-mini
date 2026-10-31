"""Physiological input transitions in the existing descending spiking fixture."""
import json,time,gc
from pathlib import Path
from dataclasses import replace
import numpy as np
from brian2 import ms
from veguinha_mini.tests.test_homeostatic_drive import intent_fixture
from veguinha_mini.tests.test_adaptive_homeostasis import AdaptiveHomeostasisTests
from veguinha_mini.sensory import inject_sensors
out=Path(__file__).resolve().parent;case=AdaptiveHomeostasisTests();rows=[]
for seed in (42,43,44):
 for enabled in (False,True):
  cfg=replace(case.config(),homeostasis_enabled=enabled)
  sensor=case.afferent_model(cfg);inputs={}
  for label,fatigue in (('fresh',0),('tired',1),('recovered',0)):
   sample=case.sample(energy=.95,fatigue=fatigue);inject_sensors(sensor,sample,config=cfg)
   v=np.asarray(sensor.group.I_sense[:]);inputs[label]=(float(v[sensor.intent.go].mean()),float(v[sensor.intent.hold].mean()))
  model,intent=intent_fixture(seed)
  start=time.perf_counter()
  for label in inputs:
   go,hold=inputs[label];model.group.I_sense[intent.go]=go;model.group.I_sense[intent.hold]=hold
   model.network.run(800*ms,namespace={});previous=np.asarray(model.monitor.count[:]).copy();model.network.run(400*ms,namespace={})
   rates=(np.asarray(model.monitor.count[:])-previous)/.4
   row=dict(seed=seed,enabled=enabled,condition=label,go_current=go,hold_current=hold,go_hz=float(rates[intent.go].mean()),hold_hz=float(rates[intent.hold].mean()),motor_hz=float(rates[128:].mean()))
   rows.append(row);print(json.dumps(row),flush=True)
  rows[-1]['real_s_for_3p6_sim']=time.perf_counter()-start
  (out/'reduced_probe.json').write_text(json.dumps(dict(engine='Brian2 Cython; descending competition and motor permission, no body',rows=rows),indent=2))
  del sensor,model,intent;gc.collect()
for seed in (42,43,44):
 a={r['condition']:r for r in rows if r['seed']==seed and r['enabled']}
 assert a['tired']['motor_hz']<a['fresh']['motor_hz']*.25,a
 assert a['recovered']['motor_hz']>a['tired']['motor_hz']+1,a
print('3 seeds: fatigue suppresses motor permission; recovery reopens it',flush=True)
