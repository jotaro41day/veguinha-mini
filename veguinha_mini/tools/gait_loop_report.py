"""Post-process recorded closed-loop trials; replay torques only to audit measurements.

This script is not a controller. Every replay must reproduce the recorded free
MuJoCo state exactly before its additional measurements are accepted.
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from veguinha_mini.mujoco_body import MujocoBody
from veguinha_mini.mk1_validation import atomic

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/gait_loop_v1'
A=[0,4,2];B=[3,1,5]


def summarize_contacts(samples):
    c=np.asarray(samples,bool).reshape(-1,6)
    a=c[:,A].sum(1);b=c[:,B].sum(1)
    av=a/3;bv=b/3
    return dict(support=c.mean(0).tolist(),
                exclusive_tripod=float((((a==3)&(b<=1))|((b==3)&(a<=1))).mean()),
                all_six=float(c.all(1).mean()),flight=float((~c.any(1)).mean()),
                alternation=-float(np.corrcoef(av,bv)[0,1]) if min(av.std(),bv.std())>1e-10 else 0.)


def inspect(path):
    r=json.loads(path.read_text());d=np.load(path.with_suffix('.npz'))
    n=round(r['seconds']/.02);late=slice(n//2,n)
    body=MujocoBody(variant='mk1')
    if r['motor_circuit']=='modular':
        for _ in range(100):body.step(.02,np.zeros((6,3)))
        body.energy=1.
    forward=[];height=[];errors=[];world_speed=[]
    for torque,qpos in zip(d['torque'],d['qpos']):
        body.step(.02,torque)
        errors.append(float(np.max(abs(body.data.qpos-qpos))))
        forward.append(float(body.velocity@[np.cos(body.heading),np.sin(body.heading)]))
        height.append(body.data.site_xpos[body.toe_ids,2].copy())
        world_speed.append(float(np.linalg.norm(body.velocity)))
    error=max(errors)
    if error!=0:raise AssertionError(f'{path.name}: replay qpos error {error}; measurements rejected')
    measured={**summarize_contacts(d['subsupport'][late]),
              'forward_m_s':float(np.mean(forward[late])),
              'speed_world_rms_m_s':float(np.sqrt(np.mean(np.square(world_speed[late])))),
              'foot_height_span_p95_p5_mm':(np.diff(np.percentile(np.array(height)[late],[5,95],axis=0),axis=0)[0]*1000).tolist(),
              'motor_limit_fraction':float((abs(d['torque'][late])>=.99).mean()),
              'replay_qpos_max_error':error,
              'window_s':[n//2*.02,n*.02]}
    row={**{k:v for k,v in r.items() if k not in ('joints','motor_rates_hz','motor_torque_rms')},**measured,'file':path.name}
    if r.get('silence_tail_s'):
        row['silence_result']={**r['silence_result'],'last_second_world_speed_rms_m_s':float(np.sqrt(np.mean(np.square(world_speed[-50:]))))}
        if r['silence_tail_s']>=5:
            speeds=[];relax=[]
            for block in range(1000):
                body.step(.02,np.zeros((6,3)))
                speeds.append(float(np.linalg.norm(body.velocity)))
                if (block+1)%50==0:
                    relax.append(dict(seconds_after_ablation=r['silence_tail_s']+(block+1)*.02,
                                      planar_speed_rms_m_s=float(np.sqrt(np.mean(np.square(speeds[-50:])))),
                                      tilt_deg=float(np.degrees(np.arccos(np.clip(body.upright,-1,1))))))
            atomic(OUT/'passive_relaxation.json',dict(seed=r['seed'],command='zero joint torque; continued physical settling only',samples=relax))
            row['silence_result']['passive_speed_rms_after_25s_m_s']=relax[-1]['planar_speed_rms_m_s']
    return row,np.asarray(forward)


def ci(values):
    x=np.asarray(values,float);n=len(x);sd=float(x.std(ddof=1)) if n>1 else 0.
    # All primary comparisons are paired n=3, df=2. No Gaussian 1.96 shortcut.
    if n!=3:raise ValueError('This report prespecifies three seeds')
    half=4.302652729911275*sd/np.sqrt(n)
    return dict(n=n,mean=float(x.mean()),sd=sd,ci95=[float(x.mean()-half),float(x.mean()+half)])


def font(size):
    from PIL import ImageFont
    for path in ('/data/data/com.termux/files/usr/share/fonts/TTF/DejaVuSans.ttf','/system/fonts/Roboto-Regular.ttf'):
        try:return ImageFont.truetype(path,size)
        except OSError:pass
    return ImageFont.load_default(size=size)


def measured_poses(path):
    """Draw measured mesh poses with an orthographic camera; no integration."""
    import mujoco
    from PIL import Image,ImageDraw
    data=np.load(path);body=MujocoBody(variant='mk1');m=body.model
    image=Image.new('RGB',(1500,650),'#f7f8f7');draw=ImageDraw.Draw(image)
    draw.text((35,24),'mk-1 · poses reais da malha neural → MuJoCo',font=font(32),fill='#202a33')
    draw.text((35,72),'Sem aprendizagem · seed 42 · cada pose extraída do estado físico registrado',font=font(22),fill='#52616b')
    view=np.array([.6,-.8,.65]);view/=np.linalg.norm(view)
    right=np.cross([0,0,1],view);right/=np.linalg.norm(right);up=np.cross(view,right)
    basis=np.array([right,up,view]);scale=570
    for col,t in enumerate((6.10,6.28,6.46)):
        index=round(t/.02)-1;body.data.qpos[:]=data['qpos'][index];mujoco.mj_forward(m,body.data)
        origin=body.data.qpos[:3].copy();origin[2]=0;center=np.array([250+500*col,425]);faces=[]
        def project(p):
            q=np.asarray(p)@basis.T
            return q,np.column_stack((center[0]+q[:,0]*scale,center[1]-q[:,1]*scale))
        _,floor=project(np.array([[-.4,-.4,0],[-.4,.4,0],[.4,.4,0],[.4,-.4,0]]))
        draw.polygon([tuple(v) for v in floor],fill='#e7ece9')
        for geom in range(m.ngeom):
            if m.geom_type[geom]!=mujoco.mjtGeom.mjGEOM_MESH:continue
            mesh=int(m.geom_dataid[geom]);v0=m.mesh_vertadr[mesh];nv=m.mesh_vertnum[mesh]
            f0=m.mesh_faceadr[mesh];nf=m.mesh_facenum[mesh]
            vertices=m.mesh_vert[v0:v0+nv]@body.data.geom_xmat[geom].reshape(3,3).T+body.data.geom_xpos[geom]-origin
            rotated,pixels=project(vertices);indices=m.mesh_face[f0:f0+nf];tris=rotated[indices]
            normals=np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0]);normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
            shade=.65+.35*np.abs(normals[:,2]);colors=np.clip(m.geom_rgba[geom,:3]*shade[:,None]*255,0,255).astype('uint8')
            for depth,poly,color in zip(tris[:,:,2].mean(1),pixels[indices],colors):
                faces.append((depth,poly,tuple(map(int,color))))
        for _,poly,color in sorted(faces,key=lambda f:f[0]):draw.polygon([tuple(v) for v in poly],fill=color)
        _,feet=project(body.data.site_xpos[body.toe_ids]-origin)
        for xy,loaded in zip(feet,data['subsupport'][index].any(0)):
            x,y=xy;draw.ellipse((x-5,y-5,x+5,y+5),fill='#16846e' if loaded else '#c07225')
        draw.text((40+500*col,530),f't = {t:.2f} s',font=font(26),fill='#202a33')
        draw.text((40+500*col,568),f'x = {body.position[0]:.3f} m',font=font(22),fill='#52616b')
    draw.text((35,610),'Marcadores: verde = carga no bloco de 20 ms; laranja = sem carga. Não são alvos do controlador.',font=font(20),fill='#52616b')
    image.save(OUT/'measured_walking_poses.png')


def render_report(summary):
    from PIL import Image,ImageDraw
    rows=summary['trials'];im=Image.new('RGB',(1400,1240),'#fafbf9');draw=ImageDraw.Draw(im)
    dark='#233038';muted='#596971';blue='#176c9b';orange='#c47620'
    draw.text((50,28),'Veguinha mini · avanço e apoio medidos',font=font(36),fill=dark)
    draw.text((50,82),'Corpo livre mk-1 · seed 42 · sem aprendizagem · física 2 ms',font=font(24),fill=muted)
    draw.text((50,138),'Trajetória real em 12 s',font=font(27),fill=dark)
    trajectories=[]
    for row in (rows[0],rows[3]):
        data=np.load(OUT/row['file'].replace('.json','.npz'));trajectories.append(data['position'][:600])
    pts=np.concatenate(trajectories);lo=pts.min(0)-.12;hi=pts.max(0)+.12
    scale=min(1120/(hi[0]-lo[0]),320/(hi[1]-lo[1]));center=(lo+hi)/2
    def xy(p):return np.array([720,350])+np.column_stack(((p[:,0]-center[0])*scale,-(p[:,1]-center[1])*scale))
    draw.rectangle((120,190,1310,530),outline='#cad3d0',width=2)
    for data,color in zip(trajectories,('#999f9c',blue)):
        points=xy(data);draw.line([tuple(p) for p in points],fill=color,width=4);x,y=points[-1];draw.ellipse((x-6,y-6,x+6,y+6),fill=color)
    draw.text((130,540),'Cinza: anterior · azul: corrigido. Escala igual nos eixos; marcador = posição final.',font=font(22),fill=muted)
    x0,y0=1250,500;draw.line((x0-100*scale/1000,y0,x0,y0),fill=dark,width=3)
    draw.text((1170,465),'0,10 m',font=font(19),fill=dark)
    draw.text((50,600),'Apoios reais de 6 a 8 s · carga > 0,02 N em cada passo de 2 ms',font=font(26),fill=dark)
    data=np.load(OUT/rows[3]['file'].replace('.json','.npz'));c=data['subsupport'][:600].reshape(-1,6)[3000:4000]
    for leg,name in enumerate(('LF','LM','LH','RF','RM','RH')):
        y=660+leg*39;draw.text((60,y),name,font=font(23),fill=dark)
        switches=np.flatnonzero(np.diff(np.r_[False,c[:,leg],False].astype(int)))
        for a,b in switches.reshape(-1,2):draw.rectangle((140+a*1.18,y,140+b*1.18,y+27),fill=blue if leg in A else orange)
    for tick in range(5):
        x=140+1180*tick/4;draw.text((x-20,900),f'{6+tick*.5:.1f}',font=font(20),fill=muted)
    draw.text((50,958),'Velocidade frontal oficial · média ± DP entre as seeds 42, 43 e 44',font=font(25),fill=dark)
    for j,(key,label) in enumerate((('legacy_velocity','Anterior'),('modular_velocity','Corrigido'),('uncoupled_velocity','Sem acoplamento'),('shuffled_velocity','Motores embaralhados'))):
        v=summary[key];draw.text((70,1006+42*j),f'{label}: {v["mean"]:+.5f} ± {v["sd"]:.5f} m/s',font=font(23),fill=blue if key=='modular_velocity' else dark)
    draw.text((50,1190),'Janela métrica: 6–12 s. Resultados curtos não demonstram marcha aprendida ou navegação.',font=font(21),fill=muted)
    im.save(OUT/'gait_comparison.png')
    measured_poses(OUT/rows[3]['file'].replace('.json','.npz'))


def main():
    paths=[]
    for circuit in ('legacy','modular'):
        for seed in (42,43,44):
            suffix='_silence2' if circuit=='modular' and seed==42 else ''
            paths.append(OUT/f'v2_app_{circuit}_{seed}_s12_normal{suffix}.json')
    patterns=[f'v2_app_modular_{s}_s12_normal_k0.json' for s in (42,43,44)]
    patterns += [f'v2_app_modular_{s}_s12_shuffled.json' for s in (42,43,44)]
    patterns += ['v2_app_modular_42_s6_normal_silence5.json']
    for pattern in patterns:
        p=OUT/pattern
        if p.exists():paths.append(p)
    rows=[];series={}
    for p in paths:
        r,v=inspect(p);rows.append(r);series[p.stem]=v
    baseline=rows[:3];candidate=rows[3:6]
    summary={'protocol':{'seeds':[42,43,44],'simulated_s':12,'stable_window_s':[6,12],
                         'neural_dt_ms':{'legacy':1,'modular':.5},'physics_dt_ms':2,
                         'motor_block_ms':20,'plasticity':False,'device':'25078PC3EG',
                         'velocity_definition':'mean(vx*cos(heading)+vy*sin(heading)), planar body frame',
                         'contact_threshold_N':.02,'tripod_definition':'all 3 feet of A supported and at most 1 of B, or vice versa; measured each 2ms',
                         'auxiliary_controls':'same three seeds; all available files included; silencing: seed42 only',
                         'inference_limit':'three short trials; no claim of learned gait, endurance, navigation or new-seed generalization'},
             'trials':rows,
             'legacy_velocity':ci([r['forward_m_s'] for r in baseline]),
             'modular_velocity':ci([r['forward_m_s'] for r in candidate]),
             'paired_velocity_difference':ci([c['forward_m_s']-b['forward_m_s'] for b,c in zip(baseline,candidate)]),
             'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                              [ROOT/f for f in ('config.py','network.py','cord.py','leg_circuit.py','sensory.py','sim.py','motor.py','mujoco_body.py','metrics.py','assets/hexapod_mk1.xml')]}}
    for condition,group in [('uncoupled',[r for r in rows if r['cross_gain']==0]),('shuffled',[r for r in rows if r['shuffled']])]:
        if len(group)==3:
            summary[condition+'_velocity']=ci([r['forward_m_s'] for r in group])
            summary['paired_modular_minus_'+condition]=ci([c['forward_m_s']-b['forward_m_s'] for b,c in zip(group,candidate)])
    atomic(OUT/'report.json',summary)
    lines=['# Marcha neural no mk-1 — ensaios curtos','',
           'Celular 25078PC3EG; Brian2/Cython sem viewer. Seeds 42–44, 12 s, janela final 6–12 s. Sem aprendizagem. Física MuJoCo 2 ms e blocos motores de 20 ms. Avanço oficial planar: média de vx·cos(heading) + vy·sin(heading).',
           '', '| Seed | Condição | Avanço m/s | Líquido m | Inclinação ° | Tripé exclusivo % | Alternância | Queda |',
           '|---|---|---:|---:|---:|---:|---:|---|']
    for r in rows:
        if r['seconds']!=12:continue
        name='anterior' if r['motor_circuit']=='legacy' else 'embaralhado' if r['shuffled'] else 'sem acoplamento' if r['cross_gain']==0 else 'corrigido'
        lines.append(f"|{r['seed']}|{name}|{r['forward_m_s']:.5f}|{r['net_m']:.4f}|{r['tilt_deg']:.2f}|{100*r['exclusive_tripod']:.1f}|{r['alternation']:.3f}|{'sim' if r['fell'] else 'não'}|")
    lines+=['','| Comparação pareada | Δ avanço m/s | IC95% t (n=3) |','|---|---:|---|']
    for key,label in [('paired_velocity_difference','corrigido − anterior'),('paired_modular_minus_uncoupled','corrigido − sem acoplamento'),('paired_modular_minus_shuffled','corrigido − embaralhado')]:
        if key in summary:
            v=summary[key];lines.append(f"|{label}|{v['mean']:+.5f}|[{v['ci95'][0]:+.5f}; {v['ci95'][1]:+.5f}]|")
    lines+=['','Tripé exclusivo: os três pés de A em carga e no máximo um de B, ou o inverso, medido em cada 2 ms. Todos os seis pés plantados não contam. Alternância: correlação negativa entre as frações de pés apoiados em A e B.',
            '', 'Os replays físicos servem apenas para medir a velocidade planar: cada qpos deve ser idêntico ao registro neural (erro máximo zero). O replay não gera os dados neurais nem participa do controle.',
            '', 'O candidato reutiliza os módulos neurais locais corrigidos: sensores na saída motora, diversidade de excitabilidade, inibição homóloga entre vizinhos e leitura de carga integrada nos substeps. O restante do cérebro permanece conectado. VNC196 células/perna, rede4024, 320976sinapses; anterior3712/480096. O candidato usa passo neural0,5ms e assentamento passivo inicial2s; o controle anterior preserva exatamente seu passo1ms/inicialização original. A comparação mede o conjunto das correções, não isola o efeito de cada uma.',
            '', 'Limites: três seeds curtas, sem validação nova de generalização, navegação ou durabilidade. Os pés traseiros ainda apoiam mais que os dianteiros e a saída passa parte do tempo no limite de torque. Nem todas as juntas passam o critério estrito de autocorrelação. Não há R-STDP neste ensaio; coordenação inata não demonstra marcha aprendida. Arquivos exploratórios sem prefixo v2 foram excluídos porque o primeiro canal I_cross não entrava na equação de voltagem.']
    (OUT/'relatorio.md').write_text('\n'.join(lines)+'\n')
    fields=['seed','motor_circuit','cross_gain','shuffled','seconds','window_s','forward_m_s','net_m','path_m','tilt_deg','fell','wall_contact','exclusive_tripod','alternation','flight','all_six','regular_joints','phase_locked','wall_s','device','file']
    with (OUT/'trials.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    render_report(summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('trials','source_sha256')},indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-only',action='store_true',help='redesenha os dados já conferidos sem repetir a física')
    if parser.parse_args().render_only:render_report(json.loads((OUT/'report.json').read_text()))
    else:main()
