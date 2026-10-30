"""Contratos do ensaio motor curto."""

import unittest
from unittest.mock import patch

import numpy as np
from brian2 import ms

from veguinha_mini.fast_motor import _leg_phase_lags, build_probe, measure_probe


class FastMotorTests(unittest.TestCase):
    def test_contact_phase_reset_tracks_stance_across_network_blocks(self):
        import pyximport
        pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)
        from veguinha_mini.fast_kernel import simulate

        starts = np.zeros(129, dtype=np.int32)
        targets = np.array([], dtype=np.int32)
        weights = np.array([], dtype=float)
        args = (starts, targets, weights, 1)
        kwargs = dict(return_state=True, external_angles=np.zeros(2),
                      contact_gain=0, contact_phase_reset_gain=2.0,
                      noise_amp=0)
        state = simulate(*args, 59, 42, 5.346912, .795704,
                         external_contacts=np.ones(1), **kwargs)[-1]
        self.assertAlmostEqual(state["contact_age"][0], .059)
        self.assertEqual(state["stance_end"][0], 0)
        state = simulate(*args, 1, 42, 5.346912, .795704, state=state,
                         external_contacts=np.ones(1), **kwargs)[-1]
        self.assertGreater(state["stance_end"][0], 0)
        state = simulate(*args, 1, 42, 5.346912, .795704, state=state,
                         external_contacts=np.zeros(1), **kwargs)[-1]
        self.assertEqual(state["contact_age"][0], 0)
        self.assertGreater(state["unload"][0], 0)

    def test_pair_audit_passes_contact_coupling_to_kernel(self):
        import pyximport
        pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)
        import veguinha_mini.fast_kernel as fast_kernel
        from veguinha_mini.fast_sensor_audit import _probe_result
        from veguinha_mini.search_fast import LOCAL_OSCILLATOR

        params = {**LOCAL_OSCILLATOR, "contact_gain": 1.0,
                  "interleg_contact_gain": 1.0}
        with patch.object(fast_kernel, "simulate", side_effect=RuntimeError("captured")) as call:
            with self.assertRaisesRegex(RuntimeError, "captured"):
                _probe_result(42, 2, "neighbor", .1, params)
        self.assertEqual(call.call_args.kwargs["interleg_contact_gain"], 1.0)

    def test_kernel_matches_brian2_with_restoring_sensors(self):
        import pyximport
        pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)
        from veguinha_mini.fast_kernel import simulate
        probe = build_probe(legs=1, backend="numpy", seed=42, noise=0)
        order = np.argsort(probe.pre, kind="stable")
        starts = np.r_[0, np.cumsum(np.bincount(probe.pre, minlength=128))].astype(np.int32)
        _, _, _, _, state = simulate(
            starts, probe.post[order], np.asarray(probe.recurrent.w[:])[order],
            1, 500, 42, 5.346912, .795704, 0,
            return_state=True, edge_kinds=probe.edge_kinds[order],
            inhibitory_bias=.371140, motor_bias=0,
            adaptation_a=.005423742, adaptation_d=11.211611, contact_gain=1)
        probe.network.run(500 * ms)
        np.testing.assert_array_equal(state["total"], probe.spikes.count[:])
        np.testing.assert_allclose(state["theta"], probe.joints.theta[:], atol=1e-10)

    def test_angles_restore_and_foot_contact_targets_local_inhibition(self):
        probe = build_probe(legs=2, backend="numpy", seed=42, noise=0)
        probe.joints.theta = [.2, -.3, -.4, .5]
        probe.joints.omega = 0
        probe.network.run(1 * ms)
        sense = np.asarray(probe.neurons.I_sense[:]).reshape(2, 128)
        channels = sense[:, 120:].reshape(2, 2, 2, 2).mean(axis=-1)
        self.assertEqual(channels[0, 0, 0], 0)
        self.assertGreater(channels[0, 0, 1], 0)
        self.assertGreater(channels[1, 0, 0], 0)
        self.assertEqual(channels[1, 0, 1], 0)
        self.assertTrue((sense[0, 80:120] > 0).all())
        self.assertTrue((sense[1, 80:120] == 0).all())

    def test_foot_contact_recruits_neighbor_flexor_inhibition(self):
        probe = build_probe(legs=2, backend="numpy", seed=42, noise=0,
                            interleg_contact_gain=1.0)
        probe.joints.theta = [0, -.3, 0, .3]
        probe.joints.omega = 0
        probe.network.run(1 * ms)
        sense = np.asarray(probe.neurons.I_sense[:]).reshape(2, 128)
        np.testing.assert_allclose(sense[0, 80:120], 1.0)
        for joint in range(2):
            np.testing.assert_allclose(sense[1, 80 + joint * 20:90 + joint * 20], 1.0)
            np.testing.assert_allclose(sense[1, 90 + joint * 20:100 + joint * 20], 0.0)

    def test_two_leg_kernel_matches_brian2_with_contact_and_mutual_inhibition(self):
        import pyximport
        pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)
        from veguinha_mini.fast_kernel import simulate
        probe = build_probe(legs=2, backend="numpy", seed=42, noise=0,
                            interleg_contact_gain=1.0, mutual_inhibition=8)
        order = np.argsort(probe.pre, kind="stable")
        starts = np.r_[0, np.cumsum(np.bincount(probe.pre, minlength=256))].astype(np.int32)
        _, _, _, _, state = simulate(
            starts, probe.post[order], np.asarray(probe.recurrent.w[:])[order],
            2, 500, 42, 5.346912, .795704, 0,
            return_state=True, edge_kinds=probe.edge_kinds[order],
            inhibitory_bias=.371140, motor_bias=0,
            adaptation_a=.005423742, adaptation_d=11.211611,
            contact_gain=1, interleg_contact_gain=1)
        probe.network.run(500 * ms)
        np.testing.assert_array_equal(state["total"], probe.spikes.count[:])

    def test_one_leg_has_128_neurons_and_two_joints(self):
        probe = build_probe(legs=1, backend="numpy", seed=42)
        self.assertEqual(len(probe.neurons), 128)
        self.assertEqual(len(probe.joints), 2)
        self.assertEqual(probe.pools.shape, (1, 2, 2, 8))

    def test_short_probe_reports_oscillation_and_speed(self):
        result = measure_probe(legs=1, seconds=.5, backend="numpy", seed=42)
        self.assertEqual(result["neurons"], 128)
        self.assertIn("antagonist_correlation", result)
        self.assertIn("realtime_ratio", result)

    def test_phase_lag_detects_alternating_legs(self):
        bins = np.arange(400)
        signal = np.sin(2 * np.pi * bins / 8)
        counts = np.zeros((2, 2, 2, len(bins)))
        for leg, sign in ((0, 1), (1, -1)):
            counts[leg, 0, 0] = np.maximum(0, sign * signal)
            counts[leg, 0, 1] = np.maximum(0, -sign * signal)
        lags = _leg_phase_lags(counts, [(0, 1)])[0]
        self.assertTrue(all(abs(lag) > 170 for lag in lags))

    def test_interleg_projection_samples_both_half_centers(self):
        probe = build_probe(legs=2, backend="numpy", seed=42)
        for source_leg, target_leg in ((0, 1), (1, 0)):
            for joint in range(2):
                for action in range(2):
                    sources = probe.premotor[source_leg, joint, action]
                    mask = ((probe.edge_kinds == 5)
                            & np.isin(probe.pre, sources)
                            & (probe.post // 128 == target_leg))
                    target_actions = ((probe.post[mask] % 128 - 32) // 12) % 2
                    self.assertEqual(set(target_actions.tolist()), {0, 1})

    def test_interleg_opposite_bias_changes_only_target_action(self):
        for probability in (0.0, 1.0):
            probe = build_probe(legs=2, backend="numpy", seed=42,
                                interleg_opposite_prob=probability)
            mask = probe.edge_kinds == 5
            source_action = ((probe.pre[mask] % 128 - 32) // 12) % 2
            target_action = ((probe.post[mask] % 128 - 32) // 12) % 2
            np.testing.assert_array_equal(target_action,
                                          (source_action + int(probability)) % 2)

    def test_homologous_mutual_inhibition_has_negative_weight_and_correct_target(self):
        probe = build_probe(legs=2, backend="numpy", seed=42,
                            coupling=0, mutual_inhibition=8)
        mask = probe.edge_kinds == 7
        self.assertTrue(mask.any())
        self.assertTrue((probe.pre[mask] // 128 != probe.post[mask] // 128).all())
        source = (probe.pre[mask] % 128 - 80) // 10
        target = (probe.post[mask] % 128 - 32) // 12
        np.testing.assert_array_equal(source, target)
        self.assertTrue((np.asarray(probe.recurrent.w[:])[mask] < 0).all())

    def test_cross_inhibition_decay_is_independent_of_local_inhibition(self):
        import pyximport
        pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)
        from veguinha_mini.fast_kernel import simulate
        probe = build_probe(legs=2, backend="numpy", seed=42,
                            coupling=0, mutual_inhibition=10)
        order = np.argsort(probe.pre, kind="stable")
        starts = np.r_[0, np.cumsum(np.bincount(probe.pre, minlength=256))].astype(np.int32)
        args = (starts, probe.post[order], np.asarray(probe.recurrent.w[:])[order],
                2, 2000, 42, 5.346912, .795704, 2.5)
        kwargs = {"return_state": True, "edge_kinds": probe.edge_kinds[order],
                  "inhibitory_bias": .371140, "motor_bias": 0,
                  "adaptation_a": .005423742, "adaptation_d": 11.211611}
        slow = simulate(*args, **kwargs, cross_tau_ms=150)[-1]["total"]
        middle = simulate(*args, **kwargs, cross_tau_ms=40)[-1]["total"]
        self.assertFalse(np.array_equal(slow, middle))

    def test_two_leg_kernel_matches_brian2_at_40_ms_cross_decay(self):
        import pyximport
        pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)
        from veguinha_mini.fast_kernel import simulate
        probe = build_probe(legs=2, backend="numpy", seed=42, noise=0,
                            coupling=0, mutual_inhibition=20, cross_tau_ms=40)
        order = np.argsort(probe.pre, kind="stable")
        starts = np.r_[0, np.cumsum(np.bincount(probe.pre, minlength=256))].astype(np.int32)
        _, _, _, _, state = simulate(
            starts, probe.post[order], np.asarray(probe.recurrent.w[:])[order],
            2, 500, 42, 5.346912, .795704, 0,
            return_state=True, edge_kinds=probe.edge_kinds[order],
            inhibitory_bias=.371140, motor_bias=0,
            adaptation_a=.005423742, adaptation_d=11.211611,
            cross_tau_ms=40)
        probe.network.run(500 * ms)
        np.testing.assert_array_equal(state["total"], probe.spikes.count[:])


if __name__ == "__main__":
    unittest.main()
