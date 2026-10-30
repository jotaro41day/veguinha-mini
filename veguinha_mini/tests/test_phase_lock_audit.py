"""Métrica de travamento de fase usada nos ensaios curtos."""

import unittest

import numpy as np

from veguinha_mini.phase_lock_audit import _kernel_kinds, phase_metrics
from veguinha_mini.tripod_audit import (EDGE_PAIRS, TRIPOD_A, TRIPOD_B,
                                         _tripod_metrics, neighbor_normalizers,
                                         phase_series)


def synthetic_pair(first_hz=2.6, second_hz=2.6, lag=np.pi, seconds=12):
    samples = round(seconds / .05)
    time = np.arange(samples) * .05
    counts = np.zeros((2, 2, 2, samples))
    for leg, (frequency, shift) in enumerate(((first_hz, 0), (second_hz, lag))):
        wave = np.sin(2 * np.pi * frequency * time + shift)
        counts[leg, 0, 0] = np.maximum(0, wave) * 10
        counts[leg, 0, 1] = np.maximum(0, -wave) * 10
    return counts


class PhaseLockAuditTests(unittest.TestCase):
    def test_tripod_series_and_final_window_detect_known_tripod(self):
        single = synthetic_pair(seconds=60)
        counts = np.zeros((6, 2, 2, single.shape[-1]))
        for leg in TRIPOD_A:
            counts[leg] = single[0]
        for leg in TRIPOD_B:
            counts[leg] = single[1]
        series = phase_series(counts)
        self.assertEqual(len(series), 11)
        self.assertEqual(len(series[-1]["phase_deg"]), 6)
        totals = np.zeros((6, 128))
        totals[:, :32] = 30
        measured = _tripod_metrics(counts[..., -400:], totals.ravel(), 20)
        self.assertTrue(measured["tripod_locked"])
        self.assertEqual(measured["same_pairs_locked"], 6)
        self.assertEqual(len(measured["same_phase_sd_deg"]), 6)

    def test_tripod_groups_are_bipartite_for_all_seven_connections(self):
        self.assertEqual(set(TRIPOD_A), {0, 2, 4})
        self.assertEqual(set(TRIPOD_B), {1, 3, 5})
        self.assertEqual(len(EDGE_PAIRS), 7)
        self.assertTrue(all((a in TRIPOD_A) != (b in TRIPOD_A)
                            for a, b in EDGE_PAIRS))
        np.testing.assert_allclose(neighbor_normalizers(),
                                   [1/np.sqrt(2), 1/np.sqrt(3), 1/np.sqrt(2),
                                    1/np.sqrt(2), 1/np.sqrt(3), 1/np.sqrt(2)])

    def test_fast_cross_inhibition_only_changes_its_decay_channel(self):
        kinds = np.array([0, 3, 7], dtype=np.int8)
        np.testing.assert_array_equal(_kernel_kinds(kinds, "slow"), [0, 3, 7])
        np.testing.assert_array_equal(_kernel_kinds(kinds, "fast"), [0, 3, 0])
        np.testing.assert_array_equal(kinds, [0, 3, 7])

    def test_antiphase_locks_and_inphase_does_not(self):
        opposite = phase_metrics(synthetic_pair())
        same = phase_metrics(synthetic_pair(lag=0))
        self.assertTrue(opposite["locked"])
        self.assertLess(abs(abs(opposite["phase_deg"]) - 180), 10)
        self.assertLess(opposite["phase_sd_deg"], 10)
        self.assertFalse(same["locked"])
        self.assertTrue(phase_metrics(synthetic_pair(lag=0), target="same")["locked"])

    def test_frequency_mismatch_does_not_count_as_lock(self):
        drifting = phase_metrics(synthetic_pair(second_hz=2.85))
        self.assertFalse(drifting["locked"])
        self.assertGreater(drifting["phase_sd_deg"], 35)


if __name__ == "__main__":
    unittest.main()
