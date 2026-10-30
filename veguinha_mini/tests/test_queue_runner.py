import csv
import tempfile
import unittest
from pathlib import Path

from veguinha_mini.queue_runner import baseline_statistics, decision_1500, first_jobs


def write_rows(root, seed, condition, rows):
    keys = ("t_s", "net_path", "net_energy", "contralateral_phase",
            "contralateral_contact_correlation", "window_start_s",
            "window_net_path", "window_net_energy",
            "window_contralateral_phase",
            "window_contralateral_contact_correlation")
    with (root / f"{condition}_seed{seed}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row.get(key, 0) for key in keys} for row in rows)


class QueueRunnerTests(unittest.TestCase):
    def test_first_jobs_prioritize_short_off_controls(self):
        self.assertEqual(first_jobs(), [
            (42, "off", 600), (43, "off", 600), (44, "off", 600),
            (42, "shuffled", 900), (43, "shuffled", 900),
            (42, "learning", 1500), (43, "learning", 1500),
            (44, "learning", 1500),
        ])

    def test_early_stop_uses_equal_length_windows_and_two_seed_rule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed, control, early, late in ((42, .1, .08, .09),
                                               (43, .2, .15, .16),
                                               (44, .3, .25, .24)):
                write_rows(root, seed, "off", [
                    {"t_s": 600, "net_path": control, "net_energy": 10}])
                write_rows(root, seed, "learning", [
                    {"t_s": 600, "net_path": early, "net_energy": 10},
                    {"t_s": 1500, "window_start_s": 900,
                     "window_net_path": late, "window_net_energy": 11},
                ])
            stats = baseline_statistics(root)
            self.assertAlmostEqual(stats["net_path_mean"], .2)
            self.assertAlmostEqual(stats["net_path_min"], .1)
            self.assertAlmostEqual(stats["net_path_max"], .3)
            result = decision_1500(root)
            self.assertTrue(result["stop"])
            self.assertEqual(result["wins_vs_off"], 0)
            self.assertEqual(result["upward_trends"], 0)
            write_rows(root, 42, "learning", [
                {"t_s": 600, "net_path": .08},
                {"t_s": 1500, "net_path": .01, "window_start_s": 900,
                 "window_net_path": .12, "window_net_energy": 11},
            ])
            write_rows(root, 43, "learning", [
                {"t_s": 600, "net_path": .15},
                {"t_s": 1500, "net_path": .01, "window_start_s": 900,
                 "window_net_path": .22, "window_net_energy": 11},
            ])
            result = decision_1500(root)
            self.assertFalse(result["stop"])
            self.assertEqual(result["wins_vs_off"], 2)


if __name__ == "__main__":
    unittest.main()
