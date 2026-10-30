"""Bounded, resumable mechanical/host validation. Never trains a controller."""
import argparse
import cProfile
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import pstats
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from .mujoco_body import MujocoBody, MJCF_PATH

ROOT = Path(__file__).parent
OUT = ROOT / "results/mk1_validation_v1"


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def versions():
    return dict(host=platform.node(), python=platform.python_version(),
                packages={k: importlib.metadata.version(k) for k in
                          ("numpy", "brian2", "Cython", "mujoco")})


def code_hash():
    files = sorted([*ROOT.glob("*.py"), *ROOT.glob("*.pyx"),
                    *ROOT.glob("assets/**/*.xml"), *ROOT.glob("assets/**/*.obj")])
    h = hashlib.sha256()
    for p in files:
        h.update(p.relative_to(ROOT).as_posix().encode() + b"\0" + p.read_bytes())
    return h.hexdigest()


def reproduce():
    from .config import Config
    from .network import build_network
    from brian2 import ms
    b = MujocoBody(variant="mk1")
    rng = np.random.default_rng(42)
    q = []
    for _ in range(100):
        b.step(.02, rng.uniform(-.4, .4, (6, 3)))
        q.append(np.r_[b.data.qpos, b.data.qvel])
    net = build_network(Config(body_model="mk1", seed=42), record_spikes=False)
    net.network.run(1000 * ms)
    result = dict(versions=versions(), code_sha256=code_hash(), seed=42,
                  physical_state=np.asarray(q).tolist(),
                  neural_counts=np.asarray(net.monitor.count[:], dtype=int).tolist(),
                  neural_v=np.asarray(net.group.v[:]).tolist())
    atomic(OUT / "reproduce.json", result)
    print(json.dumps(dict(versions=result["versions"], hash=result["code_sha256"],
                         spikes=sum(result["neural_counts"])), indent=2), flush=True)


def profile_body():
    rows = {}
    for chunk in (.002, .02):
        b = MujocoBody(variant="mk1")
        rng = np.random.default_rng(42)
        commands = rng.uniform(-.5, .5, (150, 6, 3))
        profiler = cProfile.Profile(); start = time.perf_counter()
        profiler.enable()
        for command in commands:
            for _ in range(round(.02 / chunk)):
                b.step(chunk, command)
        profiler.disable()
        stats = pstats.Stats(profiler)
        top = sorted(stats.stats.items(), key=lambda x: x[1][3], reverse=True)[:18]
        rows[str(chunk)] = dict(wall_s=time.perf_counter()-start,
                               components=[dict(function=f"{k[0]}:{k[1]}:{k[2]}",
                                                calls=v[1], self_s=v[2], cumulative_s=v[3])
                                           for k, v in top],
                               final_qpos=b.data.qpos.tolist())
    atomic(OUT / "body_profile.json", dict(versions=versions(), rows=rows))
    return rows


def thermal():
    from .config import Config
    from .sim import Simulation
    s = Simulation(Config(body_model="mk1", block_ms=20), headless=True,
                   physics="mujoco", body_substep_ms=2)
    s.run_headless(.2)
    start = time.perf_counter(); rows = []
    while time.perf_counter()-start < 600:
        tick = time.perf_counter(); s.run_headless(5)
        elapsed = time.perf_counter()-tick
        rows.append(dict(wall_age_s=time.perf_counter()-start,
                         wall_s_per_sim_s=elapsed/5,
                         simulated_s=s.simulated_ms/1000))
        atomic(OUT / "thermal.json", dict(versions=versions(), complete=False, rows=rows))
        print(json.dumps(rows[-1]), flush=True)
    tick = time.perf_counter(); s.run_headless(5)
    rows.append(dict(wall_age_s=time.perf_counter()-start,
                     wall_s_per_sim_s=(time.perf_counter()-tick)/5,
                     simulated_s=s.simulated_ms/1000))
    atomic(OUT / "thermal.json", dict(versions=versions(), complete=True, rows=rows))
    profile_body()


def afterload():
    """Repeat the initial workload from seed 42, avoiding learning/state drift."""
    from .config import Config
    from .sim import Simulation
    s=Simulation(Config(body_model='mk1',block_ms=20),headless=True,
                 physics='mujoco',body_substep_ms=2)
    s.run_headless(.2)
    start=time.perf_counter();s.run_headless(5);elapsed=time.perf_counter()-start
    old=json.loads((OUT/'thermal.json').read_text())
    cp=cProfile.Profile();cp.enable();s.run_headless(1);cp.disable()
    stats=pstats.Stats(cp)
    rows=sorted(stats.stats.items(),key=lambda kv:kv[1][2],reverse=True)[:25]
    report=dict(versions=versions(),initial_wall_s_per_sim_s=old['rows'][0]['wall_s_per_sim_s'],
                after_10min_fresh_wall_s_per_sim_s=elapsed/5,
                profile_total_self_s=stats.total_tt,
                profile_self=[dict(function=f'{k[0]}:{k[1]}:{k[2]}',calls=v[1],
                                   self_s=v[2],cumulative_s=v[3]) for k,v in rows])
    atomic(OUT/'afterload.json',report);print(json.dumps(report),flush=True)


def trial(seed, condition, dt=.002, seconds=20, mirror=False, impulse=None):
    b = MujocoBody(dt=dt, variant="mk1")
    if mirror:
        xml = ET.parse(b.mjcf_path)
        # Reflection x -> -x: polar positions reflect X; axial hinge axes
        # transform with det(R) R, hence Y/Z reverse. No gait is changed.
        for e in xml.iter():
            for name in ("pos", "fromto"):
                if e.get(name):
                    a = np.fromstring(e.get(name), sep=" "); a[::3] *= -1
                    e.set(name, " ".join(map(str, a)))
            if e.tag == "joint" and e.get("axis"):
                a=np.fromstring(e.get("axis"),sep=" ");a[1:] *= -1
                e.set("axis", " ".join(map(str,a)))
        # Visual meshes do not contact or contribute inertia; leave them hidden
        # in this numerical symmetry check. Mesh paths resolve from assets.
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".xml", dir=b.mjcf_path.parent) as f:
            xml.write(f.name)
            m = mujoco.MjModel.from_xml_path(f.name)
        m.opt.timestep=dt; b.model=m; b.data=mujoco.MjData(m)
    rng=np.random.default_rng(seed)
    phase=rng.uniform(0,2*np.pi,6)
    if condition == "tripod":
        phase=np.array([0,np.pi,0,np.pi,0,np.pi],dtype=float)+rng.uniform(0,2*np.pi)
    sample_times=np.arange(round(seconds/.02))*.02
    commands=np.zeros((len(sample_times),6,3))
    if condition!='passive':
        commands=np.sin(2*np.pi*sample_times[:,None,None]/.38+
                        phase[None,:,None]+np.array([0,0,np.pi/2]))
        # Exact finite-window RMS per actuator, not just equal sine amplitudes.
        commands *= (.35/np.sqrt(2))/np.sqrt(np.mean(commands**2,axis=0))[None]
    initial = b.position.copy()
    traces=[]; wall=time.perf_counter()
    body_geoms={b.model.geom(n).id for n in ('shell','head','tail')}
    for index in range(round(seconds/.02)):
        t=index*.02
        # Mechanical probe only, never passed to a neural controller.
        command=commands[index]
        if impulse is not None and index==100:
            # An impulse changes initial momentum once, with no sustained force.
            b.data.qvel[:2] += np.asarray(impulse)/.316
        b.step(.02,command)
        mat=b.data.xmat[b.body_id].reshape(3,3)
        body_floor=any((int(con.geom1)==b.floor_id and int(con.geom2) in body_geoms)
                       or (int(con.geom2)==b.floor_id and int(con.geom1) in body_geoms)
                       for con in b.data.contact)
        traces.append(np.r_[t+.02,np.dot(b.data.qvel[:3],mat[:,0]),
                            np.degrees(np.arccos(np.clip(b.upright,-1,1))),
                            b.contacts,bool(b.fell or body_floor),b.data.qpos[:3],b.data.qvel[:3]])
    trace=np.asarray(traces); late=trace[-min(len(trace),500):]
    a=late[:,3:9][:,[0,4,2]];c=late[:,3:9][:,[3,1,5]]
    x=a.mean(axis=1);y=c.mean(axis=1)
    alt=-float(np.corrcoef(x,y)[0,1]) if np.std(x)>1e-10 and np.std(y)>1e-10 else 0.
    row=dict(seed=seed,condition=condition,dt_ms=dt*1000,mirror=mirror,
             impulse=impulse, seconds=seconds, wall_s=time.perf_counter()-wall,
             forward_m_s=float(late[:,1].mean()),tilt_deg=float(late[:,2].mean()),
             fall=bool(trace[:,9].any()),support=late[:,3:9].mean(axis=0).tolist(),
             alternation=alt,three_feet=float((a.all(axis=1)|c.all(axis=1)).mean()),
             net_distance_m=float(np.linalg.norm(b.position-initial)),
             late_distance_m=float(np.linalg.norm(late[-1,10:12]-late[0,10:12])),
             final_speed_m_s=float(np.linalg.norm(b.data.qvel[:3])))
    if impulse is not None:
        recovered=None
        for i in range(150,len(trace)):
            if (np.max(np.linalg.norm(trace[i-49:i+1,13:16],axis=1))<.001
                    and np.max(trace[i-49:i+1,2])<1):
                recovered=float(trace[i,0]-2);break
        row['recovery_s']=recovered
    row['motor_command_rms']=float(np.sqrt(np.mean(commands**2)))
    return row,trace


def mechanics():
    # Resume only complete atomic rows; each trial is one seed and condition.
    path=OUT/"mechanics.json"
    saved=json.loads(path.read_text()) if path.exists() else dict(versions=versions(),rows=[])
    if not (OUT/'joint_calibration.json').exists():
        b=MujocoBody(variant='mk1');parameters=[]
        for leg_i,leg in enumerate(('lf','lm','lh','rf','rm','rh')):
            for j,joint in enumerate(b.joint_names):
                item=b.model.joint(f'{leg}_{joint}');idx=item.id
                mujoco.mj_resetData(b.model,b.data);mujoco.mj_forward(b.model,b.data)
                toe=b.data.site_xpos[b.toe_ids[leg_i]].copy()
                origin=b.data.xpos[b.model.body(f'{leg}_{joint}').id].copy()
                b.data.qpos[item.qposadr]=.1;mujoco.mj_forward(b.model,b.data)
                diff=b.data.site_xpos[b.toe_ids[leg_i]]-toe
                effect=float(diff[0]) if j==0 else float(np.dot(diff[:2],(toe-origin)[:2])/
                                                         np.linalg.norm((toe-origin)[:2]))
                parameters.append(dict(leg=leg,joint=joint,stiffness=float(b.model.jnt_stiffness[idx]),
                    damping=float(b.model.dof_damping[item.dofadr][0]),
                    springref=float(b.model.qpos_spring[item.qposadr][0]),
                    positive_rotation_effect_m=effect))
        b.model.opt.gravity[:]=0;b.model.jnt_stiffness[:]=0;b.model.dof_damping[:]=0
        for index,row in enumerate(parameters):
            mujoco.mj_resetData(b.model,b.data);b.data.qpos[2]+=.5
            command=np.zeros((6,3));command.reshape(-1)[index]=.1
            b.step(.01,command)
            row['positive_pulse_joint_rad']=float(b.angles.reshape(-1)[index])
        atomic(OUT/'joint_calibration.json',dict(versions=versions(),parameters=parameters,
            all_positive=all(p['positive_rotation_effect_m']>0 and p['positive_pulse_joint_rad']>0
                             for p in parameters)))
    jobs=[(seed,c,.002,False,None) for seed in range(42,52) for c in ("random","tripod","passive")]
    jobs += [(seed,"passive",.002,False,j) for seed in range(42,52) for j in ([.015,0],[0,.015])]
    jobs += [(seed,c,dt,False,None) for seed in range(42,52)
             for c in ('random','tripod') for dt in (.004,.001)]
    jobs += [(seed,"random",.002,True,None) for seed in range(42,52)]
    for seed,c,dt,mirror,impulse in jobs:
        key=f"{seed}_{c}_{dt}_{mirror}_{impulse}"
        if any(r["key"]==key for r in saved["rows"]):continue
        row,trace=trial(seed,c,dt,mirror=mirror,impulse=impulse)
        row["key"]=key; saved["rows"].append(row)
        atomic(path,saved)
        np.savez_compressed(OUT/(hashlib.sha256(key.encode()).hexdigest()[:12]+".npz"),trace=trace)
        print(json.dumps(row),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("reproduce","thermal","mechanics","profile","afterload"))
    args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    {"reproduce":reproduce,"thermal":thermal,"mechanics":mechanics,"profile":profile_body,
     "afterload":afterload}[args.mode]()
