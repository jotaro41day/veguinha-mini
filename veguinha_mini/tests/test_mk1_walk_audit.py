"""Mechanical probes are isolated from the neural controller."""
import unittest
from dataclasses import replace
import numpy as np
from veguinha_mini.tools.mk1_walk_audit import probe_commands, support_metrics, run
from veguinha_mini.tools.mk1_neuralcheck import trial_config
from veguinha_mini.network import build_network


class WalkAuditTests(unittest.TestCase):
    def test_single_leg_scaffold_builds_without_other_five_modules(self):
        c = trial_config(42, 'coupled', isolated=True)
        regions = tuple(r for r in c.regions if not r.name.startswith('vnc_') or r.name == 'vnc_lf')
        names = {r.name for r in regions}
        c = replace(c, backend='numpy', regions=regions, phase5_enabled=False,
                    pathways=tuple(p for p in c.pathways if p.source in names and p.target in names))
        m = build_network(c, record_spikes=False)
        self.assertEqual(set(m.half_centers), {'lf'})
        self.assertEqual(len(m.regions['vnc_lf']), 144)
        self.assertEqual(len(m.motor_pools['lf']), 6)
        self.assertEqual(len(m.plastic_edges), 0)

    def test_unsupported_timestep_rejected_before_simulating(self):
        with self.assertRaises(ValueError):
            run(42, 'tripod', seconds=4, dt=.003)

    def test_isolated_trial_keeps_three_joints_without_learning(self):
        c = trial_config(42, 'coupled', isolated=True, motor_bias=0)
        self.assertEqual(c.joint_names, ('coxa', 'femur', 'tibia'))
        self.assertEqual(c.motor_pool_bias, 0)
        self.assertTrue(c.motor_scaffold)
        self.assertEqual(sum(r.cells for r in c.regions if r.name == 'kc'), 1)
        self.assertEqual(c.motor_plastic_scope, 'afferent_dn_interleg')

    def test_paired_probes_have_equal_per_motor_rms(self):
        for condition in ('random', 'tripod'):
            command = probe_commands(42, condition, 20)
            self.assertEqual(command.shape, (1000, 6, 3))
            np.testing.assert_allclose(np.sqrt((command**2).mean(axis=0)),
                                       .35/np.sqrt(2), rtol=1e-12)

    def test_continuous_stance_is_not_tripod_alternation(self):
        self.assertEqual(support_metrics(np.ones((100, 6), bool))['alternation'], 0)
        feet = np.zeros((100, 6), bool)
        feet[:50, [0, 4, 2]] = True
        feet[50:, [3, 1, 5]] = True
        m = support_metrics(feet)
        self.assertAlmostEqual(m['alternation'], 1)
        self.assertEqual(m['exclusive_tripod'], 1)


if __name__ == '__main__':
    unittest.main()
