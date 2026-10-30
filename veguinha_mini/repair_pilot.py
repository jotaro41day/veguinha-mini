"""Short closed-loop parameter diagnostics; never trains or imposes gait."""
import argparse
import json
import time

from .contact_diagnostic import RESULTS, neural
from .phase_lock_audit import _save_atomic

CANDIDATES = {
    'baseline': {'joint_gain':10,'load_gain':1},
    'noise1': {'joint_gain':10,'load_gain':1,'noise_amp':1},
    'noise05': {'joint_gain':10,'load_gain':1,'noise_amp':.5},
    'filter60_gain30': {'joint_gain':10,'load_gain':1,'noise_amp':1,
                        'motor_filter_ms':60,'motor_gain':.30},
    'filter80_gain40': {'joint_gain':10,'load_gain':1,'noise_amp':1,
                        'motor_filter_ms':80,'motor_gain':.40},
    'force30_noise1': {'joint_gain':10,'load_gain':1,'noise_amp':1,
                       'cross_force':30},
    'force40_noise1': {'joint_gain':10,'load_gain':1,'noise_amp':1,
                       'cross_force':40},
    'load2_filter60': {'joint_gain':10,'load_gain':2,'noise_amp':1,
                       'motor_filter_ms':60,'motor_gain':.30},
    'noise05_gain30': {'joint_gain':10,'load_gain':1,'noise_amp':.5,
                       'motor_gain':.30},
    'noise05_force30': {'joint_gain':10,'load_gain':1,'noise_amp':.5,
                        'cross_force':30},
    'recruit8_biasneg1': {'joint_gain':10,'load_gain':1,'noise_amp':.5,
                          'motor_recruitment':8,'motor_bias':-1},
    'recruit12_biasneg2': {'joint_gain':10,'load_gain':1,'noise_amp':.5,
                           'motor_recruitment':12,'motor_bias':-2},
    'recruit8_gain30': {'joint_gain':10,'load_gain':1,'noise_amp':.5,
                        'motor_recruitment':8,'motor_bias':-1,'motor_gain':.30},
    'signfix_noise05': {'joint_gain':10,'load_gain':1,'noise_amp':.5,
                        'load_reflex_knee_action':1},
    'signfix_load2': {'joint_gain':10,'load_gain':2,'noise_amp':.5,
                      'load_reflex_knee_action':1},
    'signfix_recruit12': {'joint_gain':10,'load_gain':1,'noise_amp':.5,
                          'motor_recruitment':12,'motor_bias':-2,
                          'load_reflex_knee_action':1},
    'signfix_recruit12_load2': {'joint_gain':10,'load_gain':2,'noise_amp':.5,
                                'motor_recruitment':12,'motor_bias':-2,
                                'load_reflex_knee_action':1},
    'signfix_force30': {'joint_gain':10,'load_gain':1,'noise_amp':.5,
                        'cross_force':30,'load_reflex_knee_action':1},
    'offset08_oldsign': {'joint_gain':10,'load_gain':1,'noise_amp':.5,
                         'motor_recruitment':12,'motor_bias':-2,
                         'knee_proprio_offset':.08},
    'offset16_oldsign': {'joint_gain':10,'load_gain':1,'noise_amp':.5,
                         'motor_recruitment':12,'motor_bias':-2,
                         'knee_proprio_offset':.16},
    'offset24_oldsign': {'joint_gain':10,'load_gain':1,'noise_amp':.5,
                         'motor_recruitment':12,'motor_bias':-2,
                         'knee_proprio_offset':.24},
    'proprio4_oldsign': {'joint_gain':10,'load_gain':1,'noise_amp':.5,
                         'motor_recruitment':12,'motor_bias':-2,'proprio':4},
    'proprio8_oldsign': {'joint_gain':10,'load_gain':1,'noise_amp':.5,
                         'motor_recruitment':12,'motor_bias':-2,'proprio':8},
    'proprio12_oldsign': {'joint_gain':10,'load_gain':1,'noise_amp':.5,
                          'motor_recruitment':12,'motor_bias':-2,'proprio':12},
    'sensory4_p4': {'joint_gain':10,'load_gain':1,'noise_amp':.5,
                    'motor_recruitment':12,'motor_bias':-2,'proprio':4,
                    'sensory_bias':4},
    'sensory5_p4': {'joint_gain':10,'load_gain':1,'noise_amp':.5,
                    'motor_recruitment':12,'motor_bias':-2,'proprio':4,
                    'sensory_bias':5},
    'sensory5_p8': {'joint_gain':10,'load_gain':1,'noise_amp':.5,
                    'motor_recruitment':12,'motor_bias':-2,'proprio':8,
                    'sensory_bias':5},
    'sensory5_offset10': {'joint_gain':10,'load_gain':1,'noise_amp':.5,
                          'motor_recruitment':12,'motor_bias':-2,'proprio':4,
                          'sensory_bias':5,'knee_proprio_offset':.10},
    'sensory5_offset20': {'joint_gain':10,'load_gain':1,'noise_amp':.5,
                          'motor_recruitment':12,'motor_bias':-2,'proprio':4,
                          'sensory_bias':5,'knee_proprio_offset':.20},
    'motorfilter240': {'joint_gain':10,'load_gain':1,'noise_amp':.5,
                       'motor_recruitment':12,'motor_bias':-2,'proprio':4,
                       'sensory_bias':5,'motor_filter_ms':240,'motor_gain':.35},
    'motorfilter360': {'joint_gain':10,'load_gain':1,'noise_amp':.5,
                       'motor_recruitment':12,'motor_bias':-2,'proprio':4,
                       'sensory_bias':5,'motor_filter_ms':360,'motor_gain':.50},
}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--seconds',type=float,default=20)
    p.add_argument('--budget',type=float,default=250)
    a=p.parse_args()
    path=RESULTS/'repair_pilot_v1.json'
    data=json.loads(path.read_text()) if path.exists() else {'rows':[]}
    done={(r['seed'],r['condition'],r['seconds']) for r in data['rows']}
    started=time.monotonic()
    for name,params in CANDIDATES.items():
        for seed in a.seeds:
            if (seed,name,a.seconds) in done:continue
            if time.monotonic()-started>a.budget:return
            t=time.monotonic()
            row=neural(seed,seconds=a.seconds,params=params)
            row.update(seed=seed,condition=name,seconds=a.seconds,
                       params=params,wall_s=time.monotonic()-t)
            data['rows'].append(row);_save_atomic(path,data)
            print(json.dumps({key:row[key] for key in ('seed','condition','forward_speed',
                 'all_three','alternation','contact_fraction','wall_s')}),flush=True)


if __name__=='__main__':main()
