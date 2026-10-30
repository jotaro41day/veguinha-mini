import csv
import tempfile
import unittest
from pathlib import Path

from veguinha_mini.report_scaffold import comparison_rows


class ScaffoldReportTests(unittest.TestCase):
    def test_requires_all_nine_conditions_at_the_same_simulated_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (42, 43, 44):
                for condition in ("learning", "off", "shuffled"):
                    with (root / f"{condition}_seed{seed}.csv").open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, ["t_s", "net_path", "net_energy",
                                                         "contralateral_phase",
                                                         "contralateral_contact_correlation"])
                        writer.writeheader()
                        writer.writerow({"t_s": 3000, "net_path": .2,
                                         "net_energy": 3.0,
                                         "contralateral_phase": -.2,
                                         "contralateral_contact_correlation": -.3})
            rows = comparison_rows(root, (42, 43, 44), 3000)
            self.assertEqual(len(rows), 9)
            self.assertTrue(all(row["alternating"] for row in rows))
            self.assertTrue(all(row["t_s"] == 3000 for row in rows))
            with (root / "off_seed44.csv").open("a", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow([3300, .3, 4, -.4, -.4])
            self.assertEqual(len(comparison_rows(root, (42, 43, 44), 3000)), 9)
            (root / "off_seed44.csv").unlink()
            with self.assertRaises(FileNotFoundError):
                comparison_rows(root, (42, 43, 44), 3000)


if __name__ == "__main__":
    unittest.main()
