"""Replay recorded motor output in the one-leg fixture; no network or learning."""
import time
import numpy as np
from veguinha_mini.leg_circuit import FixedTrunk
from veguinha_mini.tools.leg_reflex_check import OUT, joint_metrics
from veguinha_mini.mk1_validation import atomic


def run(source,dt_ms,contact=True):
    a=np.load(source);u=a['torque']
    if u.shape!=(1000,3):raise ValueError('Expected a 20-second three-joint recording')
    b=FixedTrunk(dt=dt_ms/1000,horizontal_free=True,walls=False)
    if not contact:
        # Diagnostic only: remove the floor from this in-memory copy.
        b.model.geom_contype[b.floor]=0;b.model.geom_conaffinity[b.floor]=0
    start=time.perf_counter();angles=[];support=[];clearance=[]
    for torque in u:
        stance,z=b.step(torque);angles.append(b.angles-b.neutral)
        support.append(stance);clearance.append(z)
    angles=np.asarray(angles)
    row=dict(source=source.name,physics_dt_ms=dt_ms,contact=contact,seconds=20,
             device='controller 25078PC3EG',simulation_wall_s=time.perf_counter()-start,
             motor_rms_nm=np.sqrt(np.mean((u*np.array([.08,.2,.2]))**2,axis=0)).tolist(),
             support=float(np.mean(support[-500:])),
             joints=dict(zip(('coxa','femur','tibia'),joint_metrics(angles,u))))
    name=source.stem+f'_replay_p{dt_ms}'+('' if contact else '_air')
    np.savez_compressed(OUT/(name+'.npz'),angles=angles,torque=u,support=support,clearance=clearance)
    atomic(OUT/(name+'.json'),row)
    print(name,[(j,round(v['autocorrelation'],3),v['rhythmic']) for j,v in row['joints'].items()],flush=True)
    return row


if __name__=='__main__':
    rows=[]
    for seed in (42,43,44):
        source=OUT/f'link_{seed}_n0.5_p2_rail_open.npz'
        for dt in (4,2,1):rows.append(run(source,dt))
        rows.append(run(source,2,contact=False))
    atomic(OUT/'replay_summary.json',dict(trials=rows,scope='Fixed height/orientation XY guide; only LF actuated. Torques identical across physics steps.'))
