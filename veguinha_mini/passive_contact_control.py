"""Check whether gravity/contact alone makes the silent 2.5D body translate."""
import json
from pathlib import Path

import numpy as np

from .body import Body
from .contact_diagnostic import RESULTS, kernel
from .world import World


def main():
    advance=kernel(); world=World(); body=Body(world)
    body.vertical_support=True
    origin=body.position.copy()
    torque=np.zeros((6,2))
    flow=[]; loads=[]
    for _ in range(round(40/.02)):
        result=advance(body,world,.002,torque,10,2)
        flow.extend(result['forward'])
        loads.append(body.normal_load.sum())
    result=dict(seconds=40,forward_speed=float(np.mean(flow)),
                final_forward_speed=float(np.mean(flow[-10000:])),
                displacement=float(np.linalg.norm(body.position-origin)),
                final_normal_load=float(np.mean(loads[-500:])),
                mass_gravity=body.mass*body.gravity,
                final_height=body.height)
    output=RESULTS/'contact_passive_v1.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
