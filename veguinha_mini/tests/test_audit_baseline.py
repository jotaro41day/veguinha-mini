import csv
import json
import pickle
import tempfile
import unittest
from pathlib import Path

from veguinha_mini.audit_baseline import (
    _load_phase5_history, collect, config_for, official_ratio, write_comparison,
)


class BaselineAuditTests(unittest.TestCase):
    def test_one_metric_and_contact_only_variants(self):
        self.assertAlmostEqual(official_ratio({"displacement": 8, "path_length": 100}), .08)
        self.assertAlmostEqual(official_ratio({"displacement": 8, "path_length": 100,
                                               "net_path": .075}), .075)
        phase4 = config_for("phase4_default", 42)
        phase4_contact = config_for("phase4_contact_zero", 42)
        phase5 = config_for("phase5_off_default", 42)
        phase5_contact = config_for("phase5_off_contact_035", 42)
        self.assertEqual((phase4.block_ms, phase4.contact_threshold), (10, .35))
        self.assertEqual((phase4_contact.block_ms, phase4_contact.contact_threshold), (10, 0))
        self.assertEqual((phase5.block_ms, phase5.contact_threshold), (20, 0))
        self.assertEqual((phase5_contact.block_ms, phase5_contact.contact_threshold), (20, .35))

    def test_legacy_phase4_reuse_marks_unrecorded_observables(self):
        with tempfile.TemporaryDirectory() as directory:
            row = collect("phase4_default", 42, 300, Path(directory), Path(directory),
                          run_missing=False)
            self.assertAlmostEqual(row["net_path"], 8.448 / 758.1223)
            self.assertIsNone(row["contact_fraction"])
            self.assertIsNone(row["mean_forward_flow"])
            self.assertIn("legado", row["provenance_note"])

    def test_phase5_reuse_requires_matching_checkpoint_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = config_for("phase5_off_default", 43)
            stem = root / "off_seed43"
            with stem.with_suffix(".csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=(
                    "condition", "seed", "t_s", "displacement", "path_length",
                    "energy_spent", "contact_fraction", "mean_forward_flow"))
                writer.writeheader()
                writer.writerow(dict(condition="off", seed=43, t_s=300,
                                     displacement=12, path_length=200, energy_spent=.4,
                                     contact_fraction=.5, mean_forward_flow=.1))
            state = {"config": config, "sim": {"plasticity": False,
                                                "simulated_ms": 300000}}
            with stem.with_suffix(".pkl").open("wb") as handle:
                pickle.dump(state, handle)
            row = _load_phase5_history("phase5_off_default", 43, 300, config, root)
            self.assertAlmostEqual(row["net_path"], .06)
            self.assertEqual(row["contact_fraction"], .5)
            wrong = config_for("phase5_off_contact_035", 43)
            self.assertIsNone(_load_phase5_history("phase5_off_default", 43, 300, wrong, root))

    def test_comparison_uses_same_seed_control(self):
        rows = [dict(variant="phase4_default", seed=42, net_path=.01),
                dict(variant="phase5_off_default", seed=42, net_path=.07)]
        with tempfile.TemporaryDirectory() as directory:
            report = write_comparison(rows, [], Path(directory))
            self.assertAlmostEqual(rows[0]["delta_vs_phase5_off_pp"], -6.0)
            self.assertEqual(report["mean_by_variant"]["phase4_default"]["seeds"], 1)
            self.assertTrue((Path(directory) / "comparison.csv").exists())
            self.assertEqual(json.loads((Path(directory) / "comparison.json").read_text())
                             ["official_metric"], report["official_metric"])


if __name__ == "__main__":
    unittest.main()
