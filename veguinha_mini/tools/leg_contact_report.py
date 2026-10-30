"""Reproduce the contact/waveform diagnostic tables without running simulations."""
import json
import numpy as np
from veguinha_mini.tools.leg_reflex_check import OUT
from veguinha_mini.mk1_validation import atomic


def cycle_variability(counts):
    e=counts[:,48:120].reshape(-1,3,2,12).sum(3)/12
    m=counts[:,:48].reshape(-1,3,2,8).sum(3)/8
    y=e[:,0,0]-e[:,0,1];f=np.fft.fftfreq(len(y),.02)
    z=np.fft.ifft(np.fft.fft(y)*((f>=2)&(f<=3.5)))
    if abs(z).max()<1e-12:return dict(cycles=0,premotor_cv=None,motor_cv=None)
    phase=np.unwrap(np.angle(z))
    # Place cycle boundaries between antagonist bursts, not through a burst peak.
    starts=np.flatnonzero(np.diff(np.floor((phase-np.pi/2)/(2*np.pi)))>0)
    starts=starts[(starts>=500)&(starts<950)]
    def cv(a):
        values=np.array([a[i:j].sum(0) for i,j in zip(starts[:-1],starts[1:])])
        return (values.std(0)/np.maximum(values.mean(0),1e-12)).tolist()
    if len(starts)<4:return dict(cycles=0,premotor_cv=None,motor_cv=None)
    return dict(cycles=len(starts)-1,premotor_cv=cv(e),motor_cv=cv(m))


def summarize(rows):
    acf=np.array([r['joints']['femur']['autocorrelation'] for r in rows])
    return dict(regular_joints=sum(v['rhythmic'] for r in rows for v in r['joints'].values()),
        femur_acf_mean=float(acf.mean()),femur_acf_sd=float(acf.std(ddof=1)),
        support=float(np.mean([r['support'] for r in rows])),
        simulation_wall_s=float(sum(r['simulation_wall_s'] for r in rows)))


def main():
    profiles=('link','coherent','loadonly','load','brake','brake_sham','motorquiet','motorgraded')
    groups={};variability={};identity={};rows={}
    for profile in profiles:
        rows[profile]=[json.loads((OUT/f'{profile}_{s}_n0.5_p2_rail_open.json').read_text()) for s in (42,43,44)]
        groups[profile]=summarize(rows[profile]);variability[profile]=[]
        for seed in (42,43,44):
            a=np.load(OUT/f'{profile}_{seed}_n0.5_p2_rail_open.npz')['counts']
            variability[profile].append(dict(seed=seed,**cycle_variability(a)))
            if profile in ('motorquiet','motorgraded'):
                b=np.load(OUT/f'link_{seed}_n0.5_p2_rail_open.npz')['counts']
                identity[f'{profile}_{seed}']=int(np.max(abs(a[:,48:180]-b[:,48:180])))
    probes={}
    for rms,suffix in ((.35/np.sqrt(2),''),(.5,'_rms0.5')):
        probes[str(rms)]={}
        for condition in ('neural_matched','sine_same','sine_quadrature'):
            a=[json.loads((OUT/f'probe_{condition}_{s}{suffix}.json').read_text()) for s in (42,43,44)]
            probes[str(rms)][condition]=summarize(a)
    force={}
    for condition in ('original','frictionless','compliant'):
        a=[json.loads((OUT/f'contact_{condition}_{s}.json').read_text()) for s in (42,43,44)]
        force[condition]=summarize(a)
        force[condition]['constraint_rms_nm']=np.mean([x['torque_constraint_rms_nm'] for x in a],0).tolist()
    difference=np.array([a['joints']['femur']['autocorrelation']-b['joints']['femur']['autocorrelation']
                         for a,b in zip(rows['motorgraded'],rows['link'])])
    mean=float(difference.mean());error=float(4.3026527299*difference.std(ddof=1)/np.sqrt(3))
    report=dict(device='controller 25078PC3EG',seeds=[42,43,44],seconds=20,window_s=[10,20],
        scope='One LF leg, guided trunk. No six-leg body, full Brian2 or learning validation.',
        groups=groups,probes=probes,force=force,cycle_variability=variability,
        max_upstream_spike_difference=identity,
        paired_femur_acf_graded_minus_link=dict(mean=mean,ci95=[mean-error,mean+error]),
        one_leg_gate_passed=False,promoted_to_app=False,
        conclusion='Periodic sines work under contact at matched RMS, while neural commands remain irregular. Motor readout variability and tonic load feedback remain unresolved; no robust improvement from the last adaptation-diversity test.')
    atomic(OUT/'diagnostic_round2.json',report)
    print('groups',groups)
    print('matched RMS0.5',probes['0.5'])
    print('paired femur ACF',report['paired_femur_acf_graded_minus_link'])
    print('upstream identity',identity)


if __name__=='__main__':main()
