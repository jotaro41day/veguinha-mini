"""Summarize paired short body probes as tables, with signed-speed intervals."""
import csv
import json
import math

import numpy as np

from .contact_diagnostic import RESULTS

N = 10
SEEDS = tuple(range(52, 62))
T95_DF9 = 2.2621571627409915


def _mean_sd(values):
    values = np.asarray(values, float)
    return f"{values.mean():+.3f} ± {values.std(ddof=1):.3f}"


def _ci(values):
    values = np.asarray(values, float)
    center = values.mean()
    margin = T95_DF9 * values.std(ddof=1)/math.sqrt(len(values))
    return f"[{center-margin:+.3f}; {center+margin:+.3f}]"


def _table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "|" + "|".join("---" for _ in headers) + "|"]
                    + ["| " + " | ".join(map(str,row)) + " |" for row in rows])


def main():
    source=json.loads((RESULTS/'transplant_diagnostic_v1.json').read_text())
    data={(r['condition'],r['seed']):r for r in source['rows']}
    names=[('replay','Rede original'),('network_hip_imposed_knee','Quadril rede + joelho imposto'),
           ('imposed_hip_network_knee','Quadril imposto + joelho rede'),
           ('sine_network_phases','Seno + fases neurais'),
           ('network_template_tripod_rms','Onda neural + fases 0°/180°'),
           ('imposed_matched','Seno tripé, RMS pareado'),
           ('imposed_original','Seno tripé, amplitude original'),
           ('no_coupling','Sem acoplamento'),('motor_shuffled','Motor embaralhado'),
           ('sine_random_phases','Seno, fases aleatórias'),
           ('random_mirror','Seno, corpo espelhado'),
           ('random_isotropic','Seno, atrito isotrópico'),
           ('random_friction_swapped','Seno, atrito trocado'),
           ('shuffled_mirror','Motor embaralhado, corpo espelhado')]
    missing=[(name,s) for name,_ in names for s in SEEDS if (name,s) not in data]
    if missing:raise ValueError(f"incomplete paired trials: {missing[:4]}")
    def values(name,key):return [data[name,s][key] for s in SEEDS]
    parts=[]
    parts.append(_table(['Protocolo','Valor'],[
        ('Seeds pareadas','52–61 (10)'),('Duração por condição','40 s simulados'),
        ('Corpo corrigido','sustentação vertical e contato por altura/carga; dt 2 ms'),
        ('Comando motor','mantido por blocos de 20 ms'),
        ('Métrica oficial','velocidade frontal assinada média em 0–40 s'),
        ('Apoio, três pés, alternância','janela final 20–40 s')]))
    parts.append(_table(['Condição','Velocidade (unid./s), média ± DP','IC95%','Apoio %',
                         '3 pés juntos %','Alternância'],[
        (label,_mean_sd(values(name,'forward_speed')),_ci(values(name,'forward_speed')),
         f"{100*np.mean(values(name,'contact_fraction')):.1f}",
         f"{100*np.mean(values(name,'all_three')):.1f}",
         f"{np.mean(values(name,'alternation')):+.3f}") for name,label in names]))
    comparisons=[('network_hip_imposed_knee','replay'),
                 ('imposed_hip_network_knee','replay'),
                 ('sine_network_phases','replay'),
                 ('network_template_tripod_rms','replay'),
                 ('imposed_matched','replay'),('replay','no_coupling'),
                 ('replay','motor_shuffled')]
    parts.append(_table(['Diferença pareada de velocidade','Média ± DP','IC95%'],[
        (f'{a} − {b}',_mean_sd(np.array(values(a,'forward_speed'))-
                             np.array(values(b,'forward_speed'))),
         _ci(np.array(values(a,'forward_speed'))-
             np.array(values(b,'forward_speed')))) for a,b in comparisons]))
    residual=max(abs(data['sine_random_phases',s]['forward_speed']+
                     data['random_mirror',s]['forward_speed']) for s in SEEDS)
    parts.append(_table(['Hipótese','Medição / decisão'],[
        ('Reprodução de torque',f"erro máximo de velocidade: {max(abs(data['replay',s]['forward_speed']-next(r['forward_speed'] for r in json.loads((RESULTS/'contact_network_v1.json').read_text())['rows'] if r['seed']==s and r['condition']=='rede18')) for s in SEEDS):.2e}"),
        ('Simetria por espelhamento',f"resíduo máximo |v + v_espelho| = {residual:.2e}"),
        ('Atrito invertido',f"velocidade {np.mean(values('random_friction_swapped','forward_speed')):+.3f}; mantém sinal positivo"),
        ('Onda neural',"causa dominante do recuo; trocar por seno com fases neurais aumenta a velocidade"),
        ('Fase entre pernas',"contribui; template neural com 0°/180° inverte o sinal, mas apoio alternado ainda insuficiente"),
        ('Controle DAN embaralhado da fase 5',"não definido neste ensaio sem aprendizagem/DAN; motor embaralhado é controle mecânico distinto")]))
    pilot=json.loads((RESULTS/'repair_pilot_v1.json').read_text())['rows']
    groups={}
    for row in pilot:groups.setdefault(row['condition'],[]).append(row)
    parts.append(_table(['Ajuste da rede, 3 seeds × 20 s','Velocidade','Apoio %',
                         '3 pés %','Alternância'],[
        (name,_mean_sd([r['forward_speed'] for r in groups[name]]),
         f"{100*np.mean([r['contact_fraction'] for r in groups[name]]):.1f}",
         f"{100*np.mean([r['all_three'] for r in groups[name]]):.1f}",
         f"{np.mean([r['alternation'] for r in groups[name]]):+.3f}")
        for name in ('baseline','noise05','recruit12_biasneg2','signfix_noise05',
                     'signfix_recruit12_load2','offset08_oldsign','offset16_oldsign',
                     'offset24_oldsign','proprio4_oldsign','proprio8_oldsign',
                     'proprio12_oldsign','sensory4_p4','sensory5_p4','sensory5_p8',
                     'sensory5_offset10','sensory5_offset20',
                     'motorfilter240','motorfilter360')
        if name in groups and len(groups[name])==3]))
    parts.append(_table(['Entrada sensorial no modo rápido','Taxa sensorial (Hz)',
                         '3 pés %','Alternância'],[
        (name,f"{np.mean([r['sensory_hz'] for r in groups[name]]):.3f}",
         f"{100*np.mean([r['all_three'] for r in groups[name]]):.1f}",
         f"{np.mean([r['alternation'] for r in groups[name]]):+.3f}")
        for name in ('proprio4_oldsign','proprio8_oldsign','proprio12_oldsign',
                     'sensory4_p4','sensory5_p4','sensory5_p8',
                     'sensory5_offset10','sensory5_offset20',
                     'motorfilter240','motorfilter360')
        if name in groups and len(groups[name])==3]))
    closest=groups['proprio12_oldsign']
    parts.append(_table(['Porta para promover a correção','Observado','Critério','Decisão'],[
        ('Velocidade frontal (triagem 3 × 20 s)',
         f"{np.mean([r['forward_speed'] for r in closest]):+.3f}",'>0','provisório'),
        ('Alternância de apoio',
         f"{np.mean([r['alternation'] for r in closest]):+.3f}",'>0,300','passou na triagem'),
        ('Três pés juntos',
         f"{100*np.mean([r['all_three'] for r in closest]):.1f}%",'>30%','falhou'),
        ('Aferência nessa condição',
         f"{np.mean([r['sensory_hz'] for r in closest]):.3f} Hz",'ativa','quase muda'),
        ('IC95% de vantagem nas 10 seeds','não calculado após falha da triagem',
         'limite inferior >0','não avançar'),
        ('App principal, Brian2 completo, aprendizagem','não executados',
         'somente após aprovação da porta','parados')]))
    out=RESULTS/'transplant_diagnostic_report_v1.md'
    out.write_text("\n\n".join(parts)+"\n")
    with (RESULTS/'transplant_per_seed_v1.csv').open('w',newline='') as handle:
        fields=['seed','condition','forward_speed','contact_fraction','all_three',
                'alternation','torque_rms']
        writer=csv.DictWriter(handle,fields);writer.writeheader()
        for name,_ in names:
            for seed in SEEDS:
                writer.writerow({field:data[name,seed][field] for field in fields})
    print(out)


if __name__=='__main__':main()
