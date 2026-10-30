import unittest,tempfile,pickle
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import mujoco
from brian2 import Clock,Network,ms,prefs,start_scope
from veguinha_mini.config import Config
from veguinha_mini.perception import ExternalSenses
from veguinha_mini.world import World


def textured_plane(distance):
    rng=np.random.default_rng(387)
    geoms=[]
    for i,y in enumerate(np.arange(-2.4,2.4,.15)):
        for j,z in enumerate(np.arange(-1.4,1.6,.15)):
            color=' '.join(str(v) for v in rng.uniform(.1,.9,3))
            geoms.append(f'<geom type="box" size=".002 .075 .075" pos="{.25+distance} {y} {z}" rgba="{color} 1" contype="0" conaffinity="0"/>')
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom name="floor" type="plane" size="5 5 .01" pos="0 0 -2"/><body name="thorax"/>'+''.join(geoms)+'</worldbody></mujoco>')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    return SimpleNamespace(model=model,data=data,body_id=model.body('thorax').id,floor_id=model.geom('floor').id)


class BinocularRenderTests(unittest.TestCase):
    def test_180p_rendered_stereo_measures_real_plane_distance(self):
        from veguinha_mini.binocular import BinocularCircuit
        self.results=[]
        for seed in (42,43,44):
            start_scope();prefs.codegen.target='numpy'
            config=Config(seed=seed,vision_enabled=True,binocular_enabled=True,vision_width=320,vision_height=180,vision_pitch_deg=0)
            net=Network();circuit=BinocularCircuit(config,Clock(dt=1*ms),net)
            for distance in (.6,1.,1.8):
                body=textured_plane(distance);sense=ExternalSenses(config);sense.bind_model(body.model);sense.scan(body,World(food_enabled=False))
                circuit.reset_readout();circuit.set_images(sense.rgb);net.run(400*ms,namespace={});result=circuit.read(.4)
                measured=np.array([v for row in result['axial_depth_m'] for v in row if v is not None])
                self.results.append(dict(seed=seed,truth=distance,valid=len(measured),median=float(np.median(measured)) if len(measured) else None))
                print(self.results[-1],flush=True)
                self.assertGreaterEqual(len(measured),3)
                self.assertLess(abs(np.median(measured)-distance)/distance,.2)
                self.assertEqual(sense.snapshot()['camera']['encoding'],'jpeg')

    def test_arena_grid_ambiguity_is_rejected_and_accepted_distances_are_geometric(self):
        from veguinha_mini.binocular import BinocularCircuit
        from veguinha_mini.mujoco_body import MujocoBody
        body=MujocoBody(variant='mk1')
        for _ in range(100):body.step(.02,np.zeros((6,3)))
        for seed in (42,43,44):
            start_scope();prefs.codegen.target='numpy'
            config=Config(seed=seed,vision_enabled=True,binocular_enabled=True,vision_width=320,vision_height=180)
            sense=ExternalSenses(config);sense.bind_model(body.model);sense.scan(body,World(food_enabled=False))
            net=Network();circuit=BinocularCircuit(config,Clock(dt=1*ms),net);circuit.set_images(sense.rgb)
            net.run(400*ms,namespace={});result=circuit.read(.4)
            rotation=body.data.xmat[body.body_id].reshape(3,3)
            origin=body.data.xpos[body.body_id]+rotation@sense.eye_origins[0]
            midpoint=body.data.xpos[body.body_id]+rotation@sense.eye_origins.mean(0)
            gid=np.array([-1],np.int32);accepted=0
            for row in range(circuit.rows):
                for col in range(circuit.cols):
                    estimate=result['distance_m'][row][col]
                    if estimate is None:continue
                    index=(row*circuit.cols+col)*circuit.disparities;y,x=circuit.y[index],circuit.x[index]
                    ray=rotation@sense.directions[0,y,x]
                    true_distance=mujoco.mj_ray(body.model,body.data,origin,ray,sense.groups,True,body.body_id,gid)
                    true_range=np.linalg.norm(origin+true_distance*ray-midpoint)
                    self.assertGreater(true_distance,0)
                    self.assertLess(abs(estimate-true_range)/true_range,.1,(row,col,estimate,true_range))
                    accepted+=1
            self.assertGreaterEqual(accepted,4)
            print(dict(arena_seed=seed,accepted=accepted,total=circuit.rows*circuit.cols),flush=True)

    def test_moving_food_while_paused_refreshes_images_without_simulating_neurons(self):
        from veguinha_mini.sim import Simulation
        start_scope();config=Config(scale=.05,backend='numpy',vision_enabled=True,binocular_enabled=True)
        sim=Simulation(config,headless=True,physics='mujoco',body_substep_ms=2);sim.paused=True
        sequence=sim.body.senses.sequence;pose=sim.body.data.qpos.copy()
        point=sim.body.data.xpos[sim.body.body_id]+np.array([1.,0,0])
        sim.submit({'cmd':'food','value':[(point[0]+3)*100,(2-point[1])*100]});sim._commands()
        self.assertGreater(sim.body.senses.sequence,sequence)
        self.assertEqual(sim.simulated_ms,0)
        self.assertEqual(int(np.asarray(sim.binocular.monitor.count[:]).sum()),0)
        np.testing.assert_array_equal(sim.body.data.qpos,pose)
        self.assertIsNone(sim.body.senses.binocular)

    def test_optional_visual_circuit_migrates_without_changing_old_weights(self):
        from veguinha_mini.sim import Simulation
        start_scope();old_config=Config(scale=.05,backend='numpy',vision_enabled=True)
        source=Simulation(old_config,headless=True,physics='mujoco',body_substep_ms=2)
        source.run_headless(.02)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'old';source.save_checkpoint(path)
            with path.with_suffix('.pkl').open('rb') as handle:state=pickle.load(handle)
            config=replace(old_config,binocular_enabled=True);state['config']=config
            path.with_suffix('.pkl').write_bytes(pickle.dumps(state,pickle.HIGHEST_PROTOCOL))
            target=Simulation(config,headless=True,physics='mujoco',body_substep_ms=2);target.load_checkpoint(path)
            np.testing.assert_array_equal(target.model.synapses.w[:],source.model.synapses.w[:])
            self.assertEqual(target.simulated_ms,source.simulated_ms)
            target.run_headless(.04)
            self.assertTrue(np.isfinite(target.binocular.group.v[:]).all())
            target.submit({'cmd':'ablate','region':'binocular','value':True});target._commands()
            self.assertFalse(np.any(target.binocular.group.spike_gate[:]))
            target.save_checkpoint(Path(directory)/'new')
            restored=Simulation(config,headless=True,physics='mujoco',body_substep_ms=2);restored.load_checkpoint(Path(directory)/'new')
            np.testing.assert_array_equal(restored.binocular.group.v[:],target.binocular.group.v[:])
            np.testing.assert_array_equal(restored.binocular.rates,target.binocular.rates)
            np.testing.assert_array_equal(restored.body.senses.rgb,target.body.senses.rgb)
            self.assertEqual(restored.body.senses.age_s,target.body.senses.age_s)
            self.assertEqual(restored.body.senses.sequence,target.body.senses.sequence)

if __name__=='__main__':unittest.main()
