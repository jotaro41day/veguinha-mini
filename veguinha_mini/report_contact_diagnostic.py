"""Reproducible short tables and paired confirmation for the contact audit."""
import csv
import json
import math
import re
from pathlib import Path

import numpy as np

from .contact_diagnostic import RESULTS


def load(stage):
    return json.loads((RESULTS / f'contact_{stage}_v1.json').read_text())['rows']


def stat(values, digits=3, scale=1):
    a=np.asarray(values,dtype=float)*scale
    return f'{a.mean():.{digits}f} ± {a.std(ddof=1):.{digits}f}'


def paired(rows, reference, other, key):
    a={r['seed']:r[key] for r in rows if r['condition']==reference}
    b={r['seed']:r[key] for r in rows if r['condition']==other}
    seeds=sorted(set(a)&set(b))
    difference=np.array([a[s]-b[s] for s in seeds],dtype=float)
    if len(seeds)!=20:
        raise ValueError(f'expected 20 paired seeds, got {len(seeds)}')
    margin=2.093024054 * difference.std(ddof=1)/math.sqrt(20)
    return (difference.mean(),difference.std(ddof=1),difference.mean()-margin,
            difference.mean()+margin,sum(difference>0))


def torque_lag(seed,condition):
    path=RESULTS/f'contact_network_{seed}_{condition}.npz'
    torque=np.load(path)['torques']
    torque=torque[-1000:] - torque[-1000:].mean(axis=0)
    spectrum=np.fft.fft(torque,axis=0)
    frequencies=np.fft.fftfreq(len(torque),.02)
    band=(frequencies>=1)&(frequencies<=3.33)
    peak=np.flatnonzero(band)[np.abs(spectrum[band]).sum(axis=(1,2)).argmax()]
    return np.angle(spectrum[peak,:,1]/spectrum[peak,:,0],deg=True)


def trial_torque_rms(stage,seed,condition):
    torque=np.load(RESULTS/f'contact_{stage}_{seed}_{condition}.npz')['torques'][-500:]
    return np.sqrt(np.mean(torque*torque,axis=(0,1))).mean()


def main():
    old=load('convergence'); fixed=load('contact_convergence')
    loaded=load('loaded_convergence'); sweep=load('loaded_sweep')
    pilot=load('pilot'); calibration=load('calibration'); rows=load('network')
    seeds=set(range(52,72)); conditions={'rede18','controle','embaralhado'}
    if {(r['seed'],r['condition']) for r in rows} != {(s,c) for s in seeds for c in conditions}:
        raise ValueError('confirmation incomplete or duplicated')
    allconditions={c:[r for r in rows if r['condition']==c] for c in conditions}
    def table(headers, data):
        lines=['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']
        lines += ['| '+' | '.join(map(str,row))+' |' for row in data]
        return '\n'.join(lines)
    parts=[]
    parts.append(table(['Integrador','dt','n','Apoio/pé %','3 pés juntos %','Impulso/ciclo','Distância líquida'],[
        (label,f'{int(dt*1000)} ms',len(a),stat([r['contact_fraction'] for r in a],2,100),
         stat([r['all_three'] for r in a],2,100),stat([r['impulse_per_cycle'] for r in a]),
         stat([r['displacement'] for r in a],2))
        for label,rs in [('Máscara antiga',old),('Altura fixa',fixed),('Sustentação vertical',loaded)]
        for dt in (.02,.01,.005,.002) if (a:=[r for r in rs if r['dt']==dt])]))
    parts.append(table(['Defasagem quadril–joelho','n','Apoio/pé %','3 pés juntos %',
                        'Impulso/ciclo','Velocidade frontal','Alternância'],[
        (f'{lag}°',len(a),stat([r['contact_fraction'] for r in a],1,100),
         stat([r['all_three'] for r in a],1,100),stat([r['impulse_per_cycle'] for r in a],2),
         stat([r['forward_speed'] for r in a],2),stat([r['alternation'] for r in a],2))
        for lag in range(0,360,30) if (a:=[r for r in sweep if r['lag_deg']==lag])]))
    parts.append(table(['Candidato · seeds 42–44','n','Velocidade frontal','Torque RMS',
                        'Apoio %',
                        'Alternância','R fase→apoio'],[
        (name,len(a),stat([r['forward_speed'] for r in a]),
         stat([trial_torque_rms('pilot',r['seed'],name) for r in a]),
         stat([r['contact_fraction'] for r in a],1,100),
         stat([r['alternation'] for r in a]),
         stat([np.mean(r['phase_contact_r']) for r in a]))
        for name in ('original','j3_l1','j6_l1','j10_l1','j6_l2')
        if (a:=[r for r in pilot if r['condition']==name])]))
    parts.append(table(['Calibração motora · seeds 42–44','n','Torque RMS',
                        'Velocidade frontal','Alternância','R fase→apoio'],[
        (name,len(a),stat([trial_torque_rms('calibration',r['seed'],name) for r in a]),
         stat([r['forward_speed'] for r in a]),stat([r['alternation'] for r in a]),
         stat([np.mean(r['phase_contact_r']) for r in a]))
        for name in ('gain032','gain027') if (a:=[r for r in calibration if r['condition']==name])]))
    parts.append(table(['Confirmação · seeds 52–71','Velocidade frontal 0–40 s',
                        'Velocidade 20–40 s','Distância líquida euclidiana','Distância frontal assinada',
                        'Apoio %','3 pés juntos %','Alternância','R fase→apoio'],[
        (name,stat([r['forward_speed'] for r in a]),
         stat([r['final_forward_speed'] for r in a]),
         stat([r['displacement'] for r in a],2),
         stat([r['forward_distance'] for r in a],2),
         stat([r['contact_fraction'] for r in a],1,100),
         stat([r['all_three'] for r in a],1,100),
         stat([r['alternation'] for r in a]),
         stat([np.mean(r['phase_contact_r']) for r in a]))
        for name,a in ((n,allconditions[n]) for n in ('rede18','controle','embaralhado'))]))
    comparisons=[]
    for other in ('controle','embaralhado'):
        for key,label in [('forward_speed','Velocidade frontal'),
                          ('forward_distance','Distância frontal assinada')]:
            mean,sd,lo,hi,wins=paired(rows,'rede18',other,key)
            comparisons.append((f'Rede 18 − {other}',label,f'{mean:+.3f} ± {sd:.3f}',
                                f'[{lo:+.3f}; {hi:+.3f}]',f'{wins}/20'))
    parts.append(table(['Diferença pareada','Desfecho','Média ± DP','IC95% t','Seeds superiores'],comparisons))
    by_leg=[]
    for i,leg in enumerate(('LF','LM','LH','RF','RM','RH')):
        a=allconditions['rede18']; tor=[torque_lag(r['seed'],'rede18')[i] for r in a]
        tor_mean=float(np.angle(np.exp(1j*np.deg2rad(tor)).mean(),deg=True))
        neural=[r['joint_phase_deg'][i] for r in a]
        neural_mean=float(np.angle(np.exp(1j*np.deg2rad(neural)).mean(),deg=True))
        by_leg.append((leg,f'{neural_mean:+.1f}°',f'{tor_mean:+.1f}°',
            stat([r['joint_phase_lock_r'][i] for r in a]),
            stat([r['phase_contact_r'][i] for r in a])))
    parts.append(table(['Perna · rede 18','Fase neural joelho−quadril','Fase torque joelho−quadril',
                        'Travamento neural R','R fase→apoio'],by_leg))
    passive=json.loads((RESULTS/'contact_passive_v1.json').read_text())
    parts.append(table(['Corpo sem torque, 40 s','Medição'],[
        ('Velocidade frontal 0–40 s',f"{passive['forward_speed']:+.4f}"),
        ('Velocidade frontal 20–40 s',f"{passive['final_forward_speed']:+.5f}"),
        ('Distância líquida euclidiana',f"{passive['displacement']:.3f}"),
        ('Carga normal final / peso',f"{passive['final_normal_load']:.4f} / {passive['mass_gravity']:.4f}"),
    ]))
    d=paired(rows,'rede18','controle','forward_speed')
    shuffled=paired(rows,'rede18','embaralhado','forward_speed')
    mean=allconditions['rede18']; forward=np.mean([r['forward_speed'] for r in mean]); alt=np.mean([r['alternation'] for r in mean])
    gate=d[2]>0 and shuffled[2]>0 and forward>0 and alt>0
    parts.append(table(['Decisão','Valor / limite'],[
        ('Vantagem sobre controle sem acoplamento (limite inferior IC95%)',f'{d[2]:+.3f} / >0'),
        ('Vantagem sobre sinais embaralhados (limite inferior IC95%)',f'{shuffled[2]:+.3f} / >0'),
        ('Velocidade frontal média da rede',f'{forward:+.3f} / >0'),
        ('Alternância média de apoio',f'{alt:+.3f} / >0'),
        ('Liberar rede Brian2 completa e aprendizagem','Sim' if gate else 'Não'),
        ('Escopo da física','2,5D: altura e carga normais; sem roll/pitch ou queda tridimensional'),
        ('Motor','Circuito Izhikevich rápido 768 neurônios, sem plasticidade; blocos 20 ms, corpo 2 ms'),
    ]))
    log=(RESULTS/'contact_full_tests.log')
    if log.exists():
        match=re.search(r'Ran (\d+) tests in ([0-9.]+)s\s+OK',log.read_text())
        if match:
            parts.append(table(['Verificação','Resultado'],[
                ('Testes existentes e novos',f'{match.group(1)}/{match.group(1)} aprovados em {match.group(2)} s'),
                ('Física compilada vs. Body.step','Paridade nos quatro dt e em vários ciclos de tripé'),
            ]))
    out=RESULTS/'contact_diagnostic_report_v1.md'
    out.write_text('\n\n'.join(parts)+'\n')
    with (RESULTS/'contact_network_per_seed_v1.csv').open('w',newline='') as f:
        columns=['seed','condition','forward_speed','final_forward_speed','displacement',
                 'forward_distance','contact_fraction','all_three','alternation',
                 'impulse_per_cycle','phase_contact_r_mean','motor_saturation']
        writer=csv.DictWriter(f,columns);writer.writeheader()
        for r in rows:
            writer.writerow({c:np.mean(r['phase_contact_r']) if c=='phase_contact_r_mean'
                             else r.get(c) for c in columns})
    print(json.dumps({'report':str(out),'gate':bool(gate),'rows':len(rows),
                      'paired_control':[float(value) for value in d]}))


if __name__=='__main__':
    main()
