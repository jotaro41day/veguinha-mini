"""Moving-body throughput, server renderer on/off at 3 fps (no WebGL GPU claim)."""
import json
from pathlib import Path
import threading
import time
import numpy as np
from veguinha_mini.mujoco_body import MujocoBody
from veguinha_mini.mjviser_bridge import MjviserBridge
from veguinha_mini.world import World
from veguinha_mini.mk1_validation import atomic, versions, OUT


class Moving:
    def __init__(self):
        self.body=MujocoBody(variant='mk1');self.lock=threading.Lock();self.seq=0
        self.world=World()

    def latest(self):
        with self.lock:
            return dict(body=self.body.snapshot(self.world),world=self.world.snapshot(),
                        epoch=0,seq=self.seq,speed=20)


def run():
    source=Moving();bridge=MjviserBridge(source,port=8062,fps=3)
    bridge.force_render=True;bridge.active=False;bridge.start();rows=[]
    phases=np.random.default_rng(42).uniform(0,2*np.pi,(6,3))
    try:
        for enabled in (False,True,False,True):
            bridge.active=enabled
            start=time.perf_counter();sim_start=source.body.data.time
            frames=bridge.frames;cpu=bridge.render_seconds
            while time.perf_counter()-start<4:
                with source.lock:
                    torque=.35*np.sin(2*np.pi*source.body.data.time/.38+phases)
                    source.body.step(.02,torque);source.seq+=1
                time.sleep(0)  # Let the independent viewer thread acquire its snapshot.
            wall=time.perf_counter()-start;seconds=source.body.data.time-sim_start
            rendered=bridge.frames-frames
            row=dict(viewer=enabled,wall_s=wall,simulated_s=seconds,wall_s_per_sim_s=wall/seconds,
                     fps=rendered/wall,ms_per_render=1000*(bridge.render_seconds-cpu)/max(1,rendered))
            rows.append(row);print(json.dumps(row),flush=True)
    finally:bridge.stop()
    atomic(OUT/'viewer_cost.json',dict(versions=versions(),rows=rows,
           scope='server-side mjviser pose updates with physical motion at 3 fps; browser GPU not measured'))


if __name__=='__main__':run()
