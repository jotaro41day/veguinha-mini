"""Serial, restartable launcher for 20 s newborn trials on the Mi 9T Pro."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=list(range(42,52)))
    p.add_argument('--conditions',nargs='+',default=['coupled','uncoupled','shuffled','old'])
    p.add_argument('--tag',default='newborn_v1');p.add_argument('--gain',type=float,default=.18)
    p.add_argument('--load-gain',type=float,default=1);p.add_argument('--contact-gain',type=float,default=1)
    p.add_argument('--motor-bias',type=float)
    p.add_argument('--cross-tau',type=float)
    p.add_argument('--isolated',action='store_true')
    p.add_argument('--seconds',type=float,default=20);p.add_argument('--budget',type=int,default=1800)
    a=p.parse_args()
    if a.tag!=Path(a.tag).name or a.tag in ('.','..'):p.error('invalid tag')
    root=Path(__file__).resolve().parents[1]/'results/mk1_validation_v1'/a.tag
    root.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    for seed in a.seeds:
        for condition in a.conditions:
            out=root/f'{seed}_{condition}.json'
            if out.exists():
                row=json.loads(out.read_text())
                if row.get('circuit','full') != ('isolated' if a.isolated else 'full'):
                    raise ValueError('existing trial has a different circuit')
                if row['seconds'] != a.seconds:
                    raise ValueError('existing trial has a different duration')
                expected=dict(motor_gain=a.gain,load_gain=a.load_gain,contact_gain=a.contact_gain,
                              motor_pool_bias=a.motor_bias,motor_cross_tau_ms=a.cross_tau)
                if any(row['config'].get(k)!=v for k,v in expected.items()):
                    raise ValueError('existing trial has a different configuration')
                continue
            remaining=a.budget-(time.monotonic()-start)
            if remaining<10:return
            cmd=[sys.executable,'-u','-m','veguinha_mini.tools.mk1_neuralcheck',
                 '--seed',str(seed),'--condition',condition,'--output',str(out),
                 '--seconds',str(a.seconds),'--gain',str(a.gain),
                 '--load-gain',str(a.load_gain),'--contact-gain',str(a.contact_gain)]
            if a.motor_bias is not None:cmd += ['--motor-bias',str(a.motor_bias)]
            if a.cross_tau is not None:cmd += ['--cross-tau',str(a.cross_tau)]
            if a.isolated:cmd += ['--isolated']
            log=out.with_suffix('.log')
            with log.open('w') as f:
                child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=child.wait(timeout=min(180,remaining))
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait();code=124
            print(json.dumps(dict(seed=seed,condition=condition,exit=code,
                                  batch_wall_s=time.monotonic()-start)),flush=True)
            if code:
                print(log.read_text()[-3000:],flush=True)
                raise SystemExit(code)


if __name__=='__main__':main()
