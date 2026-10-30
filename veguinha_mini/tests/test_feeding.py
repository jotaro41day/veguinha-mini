import unittest
from types import SimpleNamespace
import numpy as np
from veguinha_mini.world import World
from veguinha_mini.mujoco_body import MujocoBody


class FeedingTests(unittest.TestCase):
    def test_mouth_sensor_is_inside_visual_mandibles(self):
        from veguinha_mini.perception import mouth_position
        body=MujocoBody(variant='mk1')
        local=mouth_position(body)-body.data.xpos[body.body_id]
        self.assertLess(local[0],.376)
        self.assertGreater(local[0],.229)

    def test_reset_restarts_episode_and_relaxes_muscles_without_erasing_weights(self):
        from brian2 import start_scope,ms
        from veguinha_mini.config import Config
        from veguinha_mini.sim import Simulation
        start_scope();s=Simulation(Config(scale=.05,backend='numpy',phase5_enabled=True))
        s.speed='turbo';s.model.network.run(100*ms)
        weights=np.array(s.model.synapses.w[:]);s.submit({'cmd':'reset'});s._commands();s._publish(0)
        self.assertTrue(s.paused)
        self.assertEqual(s.latest()['episode_t_ms'],0)
        self.assertEqual(s.simulated_ms,100)
        np.testing.assert_array_equal(s.model.synapses.w[:],weights)
        np.testing.assert_array_equal(s.motor.filtered_hz,0)

    def test_food_requires_mouth_contact_and_neural_drive(self):
        from veguinha_mini.perception import mouth_position,feed
        body=MujocoBody(variant='mk1');body.energy=.2
        mouth=mouth_position(body)
        world=World(odor_x=(mouth[0]+3)*100,odor_y=(2-mouth[1])*100,food_energy=.5)
        unchanged,taken=feed(body,world,0,.1,.25)
        self.assertEqual(taken,0);self.assertEqual(body.energy,.2)
        fed,taken=feed(body,world,1,.1,.25)
        self.assertGreater(taken,0)
        self.assertAlmostEqual(body.energy,.2+taken)
        self.assertAlmostEqual(fed.food_energy,.5-taken)
        self.assertEqual(fed.snapshot()['odor'],world.snapshot()['odor'])
        distant=World(odor_x=20,odor_y=20)
        self.assertEqual(feed(body,distant,1,.1,.25)[1],0)

    def test_neural_mouth_silence_has_zero_drive(self):
        from veguinha_mini.motor import NeuralMouth
        model=SimpleNamespace(group=np.zeros(16),regions={'seg':np.arange(16)},inhibitory=np.zeros(16,bool))
        mouth=NeuralMouth(model);counts=np.zeros(16,int)
        self.assertEqual(mouth.update(counts,.02)[0],0)
        counts[:8]+=1
        self.assertGreater(mouth.update(counts,.02)[0],0)

    def test_finite_stock_and_coordinates(self):
        world=World(food_energy=.01)
        consumed,taken=world.consume(2.)
        self.assertAlmostEqual(taken,.01)
        self.assertEqual(consumed.food_energy,0)
        self.assertIsNone(consumed.snapshot()['odor'])
        self.assertEqual(consumed.consume(2)[1],0)
        np.testing.assert_allclose(world.food_position_m,[1.5,.1,.065])

    def test_paused_checkpoint_preserves_world_and_mouth(self):
        import tempfile
        from pathlib import Path
        from brian2 import start_scope,ms
        from veguinha_mini.config import Config
        from veguinha_mini.sim import Simulation
        start_scope();config=Config(scale=.05,backend='numpy',feeding_enabled=True,vision_enabled=True,nociception_enabled=True)
        first=Simulation(config,headless=True,physics='mujoco',body_substep_ms=2)
        first.world=World(odor_x=123,odor_y=321,food_energy=.23);first.paused=True
        first.model.network.run(100*ms)
        first.mouth.rates[:]=[12,4]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'paused';first.save_checkpoint(path)
            second=Simulation(config,headless=True,physics='mujoco',body_substep_ms=2);second.load_checkpoint(path)
        self.assertEqual(second.world,first.world)
        np.testing.assert_array_equal(second.mouth.rates,[12,4])
        np.testing.assert_array_equal(second.model.synapses.w[:],first.model.synapses.w[:])


if __name__=='__main__':unittest.main()
