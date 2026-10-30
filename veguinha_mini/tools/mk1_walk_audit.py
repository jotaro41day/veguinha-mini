"""Bounded mechanical probes. Sines never enter the neural/app controller."""
import argparse
import time
from pathlib import Path
import numpy as np
import mujoco
from veguinha_mini.mujoco_body import MujocoBody
from veguinha_mini.mk1_validation import atomic, versions

OUT = Path(__file__).resolve().parents[1]/'results/mk1_walk_audit_v1'


def probe_commands(seed, condition, seconds, period=.38):
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2*np.pi, 6)
    if condition == 'tripod':
        phase = np.array([0, np.pi, 0, np.pi, 0, np.pi])+rng.uniform(0, 2*np.pi)
    elif condition != 'random':
        raise ValueError('unknown mechanical probe')
    t = np.arange(round(seconds/.02))*.02
    command = np.sin(2*np.pi*t[:, None, None]/period+phase[None, :, None]
                     +np.array([0, 0, np.pi/2]))
    command *= (.35/np.sqrt(2))/np.sqrt((command**2).mean(axis=0))[None]
    return command


def support_metrics(feet):
    feet = np.asarray(feet, bool)
    a, b = feet[:, [0, 4, 2]], feet[:, [3, 1, 5]]
    x, y = a.mean(axis=1), b.mean(axis=1)
    alt = -float(np.corrcoef(x, y)[0, 1]) if min(x.std(), y.std()) > 1e-10 else 0.
    exclusive = (a.all(axis=1) & ~b.any(axis=1)) | (b.all(axis=1) & ~a.any(axis=1))
    return dict(support=feet.mean(axis=0).tolist(), alternation=alt,
                three_feet=float((a.all(axis=1)|b.all(axis=1)).mean()),
                exclusive_tripod=float(exclusive.mean()))


def run(seed, condition, femur_gear=.08, seconds=20, period=.38, dt=.002):
    if not 4 <= seconds <= 20 or femur_gear <= 0 or period <= 0:
        raise ValueError('probe outside short diagnostic bounds')
    b = MujocoBody(variant='mk1', dt=dt)
    # One mechanical factor per comparison; the MJCF on disk is untouched.
    b.model.actuator_gear[1::3, 0] = femur_gear
    for _ in range(100):
        b.step(.02, np.zeros((6, 3)))
    initial = b.position
    command = probe_commands(seed, condition, seconds, period)
    sampled = []; fine_contact = []; fine_loaded = []; max_clearance = np.zeros(6)
    start = time.perf_counter(); fell = False
    for k, torque in enumerate(command):
        for substep in range(round(.02/dt)):
            b.step(dt, torque)
            clearance = b.data.site_xpos[b.toe_ids, 2]-b.model.geom_size[b.foot_ids, 0]
            if k >= len(command)//2:
                fine_contact.append(b.contacts.copy())
                fine_loaded.append(b.normal_load > .02)
                max_clearance = np.maximum(max_clearance, clearance)
            fell |= b.fell
        mat = b.data.xmat[b.body_id].reshape(3, 3)
        sampled.append(dict(qpos=b.data.qpos.copy(), qvel=b.data.qvel.copy(),
                            angles=b.angles, torque=torque,
                            forward=float(b.data.qvel[:3] @ mat[:, 0]),
                            tilt=float(np.degrees(np.arccos(np.clip(b.upright, -1, 1)))),
                            load=b.normal_load.copy()))
    elapsed = time.perf_counter()-start
    arrays = {k: np.asarray([r[k] for r in sampled]) for k in sampled[0]}
    late = slice(len(command)//2, None)
    row = dict(seed=seed, condition=condition, femur_gear=femur_gear, seconds=seconds,
               period=period, physics_dt_ms=1000*dt, versions=versions(), device='controller 25078PC3EG',
               forward_m_s=float(arrays['forward'][late].mean()),
               tilt_deg=float(arrays['tilt'][late].mean()), fall=bool(fell),
               max_clearance_m=max_clearance.tolist(),
               angle_range_rad=np.ptp(arrays['angles'][late], axis=0).tolist(),
               distance_m=float(np.linalg.norm(b.position-initial)),
               torque_rms_Nm=np.sqrt(((command*b.model.actuator_gear[:, 0].reshape(6, 3))**2)
                                     .mean(axis=0)).tolist(),
               contact=support_metrics(fine_contact), loaded=support_metrics(fine_loaded),
               wall_s=elapsed)
    name = f'{seed}_{condition}_f{femur_gear:g}_p{period:g}'
    if dt != .002:
        name += f'_dt{1000*dt:g}'
    OUT.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(OUT/f'{name}.npz', **arrays)
    atomic(OUT/f'{name}.json', row)
    print({k: row[k] for k in ('seed', 'condition', 'femur_gear', 'forward_m_s', 'fall', 'wall_s')},
          row['contact'], 'clearance', row['max_clearance_m'], flush=True)
    return row


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--conditions', nargs='+', choices=['random', 'tripod'], default=['tripod'])
    p.add_argument('--femur-gears', type=float, nargs='+', default=[.08, .2])
    p.add_argument('--seconds', type=float, default=20)
    p.add_argument('--period', type=float, default=.38)
    p.add_argument('--physics-dt-ms', type=int, choices=[1, 2, 4], default=2)
    a = p.parse_args()
    for seed in a.seeds:
        for condition in a.conditions:
            for gear in a.femur_gears:
                run(seed, condition, gear, a.seconds, a.period, a.physics_dt_ms/1000)
