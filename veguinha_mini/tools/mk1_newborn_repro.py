"""One-second cross-device check of the exact newborn experiment circuit."""
import json
import time
import numpy as np
from pathlib import Path
from veguinha_mini.config import Config
from veguinha_mini.tools.mk1_neuralcheck import Observed
from veguinha_mini.mk1_validation import OUT, atomic, versions, code_hash


def run():
    c=Config(seed=42,body_model='mk1',block_ms=20,motor_scaffold=True,
             motor_gain=.18,foot_contact_gain=1,phase5_enabled=True,
             neural_value_channels=True,motor_plastic_scope='afferent_dn_interleg',
             vnc_hebb_rate=0,learning_rate=.00001)
    s=Observed(c,headless=True,physics='mujoco',body_substep_ms=2,load_gain=1)
    s.plasticity=False;decode=s.motor.update
    def update(count,dt):
        torque=decode(count,dt);s.applied=torque.copy();return torque
    s.motor.update=update
    before=np.asarray(s.model.synapses.w[:]).copy()
    start=time.perf_counter();s.run_headless(1)
    result=dict(versions=versions(),code_sha256=code_hash(),wall_s=time.perf_counter()-start,
                spikes=np.asarray(s.model.monitor.count[:],dtype=int).tolist(),
                v=np.asarray(s.model.group.v[:]).tolist(),
                qpos=s.body.data.qpos.tolist(),qvel=s.body.data.qvel.tolist(),
                weights_unchanged=bool(np.array_equal(before,np.asarray(s.model.synapses.w[:]))),
                dan_external_current=np.asarray(s.model.group.I_sense[:])[s.model.regions['dan']].tolist())
    atomic(OUT/'reproduce_newborn.json',result)
    print(json.dumps(dict(cells=len(s.model.group),spikes=sum(result['spikes']),
                         weights_unchanged=result['weights_unchanged'],wall_s=result['wall_s'])),flush=True)


if __name__=='__main__':run()
