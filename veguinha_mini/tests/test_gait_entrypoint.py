import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from veguinha_mini.tools.gait_loop_report import summarize_contacts


class GaitEntrypointTests(unittest.TestCase):
    def launch(self,*args):
        from veguinha_mini.main import main
        sim=MagicMock();sim.frame={}
        with patch('sys.argv',['main.py','--serve',*args]),patch('veguinha_mini.sim.Simulation',return_value=sim) as construct,patch('veguinha_mini.server.serve_local'):
            main()
        return construct.call_args.args[0],sim

    def test_mk1_server_uses_confirmed_circuit_and_can_start_paused(self):
        config,sim=self.launch('--body-model','mk1','--start-paused')
        self.assertEqual(config.motor_circuit,'modular')
        self.assertEqual(config.dt_ms,.5)
        self.assertTrue(config.phase5_enabled)
        self.assertTrue(config.neural_value_channels)
        self.assertTrue(config.homeostasis_enabled)
        self.assertEqual(config.motor_plastic_scope,'afferent_dn_interleg')
        self.assertTrue(sim.paused)
        self.assertTrue(sim.frame['paused'])

    def test_legacy_homeostasis_remains_available_for_paired_controls(self):
        config,_=self.launch('--body-model','mk1','--legacy-homeostasis')
        self.assertFalse(config.homeostasis_enabled)

    def test_phase4_preserves_frozen_reference(self):
        config,_=self.launch('--body-model','mk1','--phase4')
        self.assertFalse(config.phase5_enabled)

    def test_controls_remain_accessible(self):
        for args in ((),('--body-model','mk1','--motor-circuit','legacy'),('--body-model','mk1','--scale','.5')):
            config,_=self.launch(*args)
            self.assertEqual(config.motor_circuit,'legacy')
            self.assertEqual(config.dt_ms,1.)

    def test_all_feet_planted_is_not_alternating_tripod(self):
        r=summarize_contacts(np.ones((10,6),bool))
        self.assertEqual(r['exclusive_tripod'],0)
        self.assertEqual(r['all_six'],1)
        self.assertEqual(r['alternation'],0)
        c=np.zeros((10,6),bool);c[:5,[0,4,2]]=1;c[5:,[3,1,5]]=1
        r=summarize_contacts(c)
        self.assertEqual(r['exclusive_tripod'],1)
        self.assertAlmostEqual(r['alternation'],1)


if __name__=='__main__':unittest.main()
