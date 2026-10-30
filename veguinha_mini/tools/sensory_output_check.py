"""Short, paired structural rerouting test; never trains or drives a gait."""
import argparse,json,time,os
from pathlib import Path
import numpy as np
from veguinha_mini.tools.leg_reflex_check import run,OUT as LEG
from veguinha_mini.mk1_validation import atomic

OUT=LEG.parent/'sensory_output_v1'
CASE='tonic_threshold'

def path(seed,reroute,excitability=False):return LEG/f'{"excitability_" if excitability else ""}{"sensory_output_" if reroute else ""}closedloop_{CASE}_{seed}_n0.5_p2_rail_open.json'

def main():
 a=argparse.ArgumentParser();a.add_argument('--seeds',type=int,nargs='+',required=True);a.add_argument('--control',action='store_true');a.add_argument('--excitability',action='store_true');a.add_argument('--routing-only',action='store_true');args=a.parse_args()
 if (args.control and args.excitability) or (args.routing_only and (args.control or args.excitability)):a.error('Choose control, routing-only, or corrected candidate')
 args.excitability=not(args.control or args.routing_only)
 OUT.mkdir(parents=True,exist_ok=True)
 atomic(LEG/'task.json',dict(title='Controle antigo pareado' if args.control else 'Confirmando a correção neural',detail='Uma perna LF;20s;ganhos4/12,corpo2ms. Seeds3001–3010 reservadas; sem aprendizado. Candidato: sensores na saída e b=.18–.22.'))
 start=time.perf_counter();last_done=None
 for seed in args.seeds:
  if time.perf_counter()-start>170:
   print('Budget reached; resume the same command.',flush=True);break
  reroute=not args.control
  f=path(seed,reroute,args.excitability)
  if not f.exists():run(seed,'link',fixture='rail_open',motor_case=CASE,feedback=True,sensory_output=reroute,motor_excitability=args.excitability)
  r=json.loads(f.read_text())
  assert r.get('sensory_output',False)==reroute and r.get('motor_excitability',False)==args.excitability
  assert r['position_gain']==4 and r['load_gain']==12 and r['physics_dt_ms']==2
  saved=np.load(f.with_suffix('.npz'));assert saved['angles'].shape==(1000,3)
  source=LEG/f'realinput_{CASE}_{seed}_n0.5_p2_rail_open.npz'
  exact=None
  if reroute and source.exists():
   ref=np.load(source);exact=bool(np.array_equal(saved['counts'][:,48:180],ref['counts'][:,48:180]))
   assert exact,'Sensory reroute changed the rhythm core despite no return connections'
  print(seed,'reroute',reroute,'regular',sum(v['rhythmic'] for v in r['joints'].values()),'core_identical_open',exact,flush=True)
  last_done=seed
  atomic(OUT/f'{"excitability" if args.excitability else "reroute" if reroute else "control"}_{seed}.json',dict(seed=seed,regular=sum(v['rhythmic'] for v in r['joints'].values()),core_identical_open=exact,source=str(f)))
 atomic(LEG/'progress.json',dict(pid=0,state='bloco concluído',profile='sensory_output',seed=last_done,simulated_s=20,updated=time.time()))

if __name__=='__main__':main()
