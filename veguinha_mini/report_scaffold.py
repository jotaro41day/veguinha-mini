"""Tabela pareada do treino motor isolado, sempre no mesmo tempo simulado."""

import argparse
import csv
from pathlib import Path


CONDITIONS = ("learning", "off", "shuffled")
FIELDS = ("seed", "condition", "t_s", "net_path", "net_energy",
          "contralateral_phase", "contralateral_contact_correlation", "alternating")


def comparison_rows(destination, seeds=(42, 43, 44), seconds=3000):
    root = Path(destination)
    rows = []
    for seed in seeds:
        for condition in CONDITIONS:
            source = root / f"{condition}_seed{seed}.csv"
            with source.open(newline="", encoding="utf-8") as handle:
                matches = [row for row in csv.DictReader(handle)
                           if abs(float(row["t_s"]) - seconds) < 1e-6]
            if not matches:
                raise ValueError(f"checkpoint de {seconds:g} s ausente: {source.name}")
            observed = matches[-1]
            phase = float(observed["contralateral_phase"])
            contact = float(observed["contralateral_contact_correlation"])
            rows.append({
                "seed": seed, "condition": condition, "t_s": seconds,
                "net_path": float(observed["net_path"]),
                "net_energy": float(observed["net_energy"]),
                "contralateral_phase": phase,
                "contralateral_contact_correlation": contact,
                "alternating": phase < -0.1 and contact < -0.1,
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent / "results" / "scaffold_train_v1")
    parser.add_argument("--seconds", type=float, default=3000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    rows = comparison_rows(args.output_dir, args.seeds, args.seconds)
    destination = args.output_dir / "comparison.csv"
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(destination)


if __name__ == "__main__":
    main()
