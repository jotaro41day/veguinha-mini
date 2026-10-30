"""Compara a linha de base da fase 4 com a fase 5 sem plasticidade.

Métrica oficial: soma dos deslocamentos líquidos por episódio dividida pela
soma dos deslocamentos físicos entre passos. Reinícios não entram no caminho.
Os quatro protocolos usam a mesma arena, duração e seed. A diferença de bloco
físico (10 ms na fase 4, 20 ms na fase 5) é preservada e aparece no relatório.
"""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np

from .config import Config
from .metrics import net_path_ratio
from .sim import Simulation


HERE = Path(__file__).resolve().parent
VARIANTS = (
    "phase4_default",
    "phase4_contact_zero",
    "phase5_off_default",
    "phase5_off_contact_035",
)
REQUIRED_VARIANTS = VARIANTS[:3]
FIELDS = (
    "variant", "seed", "simulated_seconds", "block_ms", "contact_threshold",
    "displacement", "path_length", "net_path", "energy_spent",
    "contact_fraction", "mean_forward_flow", "mean_abs_lateral_flow",
    "mean_abs_rotation", "wall_collisions", "wall_seconds", "source",
)
SCAFFOLD_FIELDS = (
    "value_flow_tau_ms",
    "motor_scaffold", "half_center_recurrence", "half_center_recruitment",
    "half_center_motor", "half_center_cross_inhibition",
    "half_center_motor_inhibition", "half_center_dn_drive",
    "half_center_interleg_strength", "half_center_adaptation_a",
    "half_center_adaptation_d", "half_center_inhibitory_bias",
)


def config_for(variant, seed, backend="cython"):
    """Mantém todas as condições iguais dentro de cada fase, salvo contato."""
    if variant not in VARIANTS:
        raise ValueError(f"variante inválida: {variant}")
    options = {"seed": seed, "backend": backend}
    if variant.startswith("phase5"):
        options.update(phase5_enabled=True, block_ms=20)
    if variant == "phase4_contact_zero":
        options["foot_contact_angle"] = 0.0
    if variant == "phase5_off_contact_035":
        options["foot_contact_angle"] = 0.35
    return Config(**options)


def fingerprint(config):
    settings = asdict(config)
    # A auditoria reproduz as variantes anteriores ao andaime. Excluir os
    # campos novos de valor padrão mantém válidas as medições já concluídas.
    if not config.motor_scaffold:
        for name in SCAFFOLD_FIELDS:
            settings.pop(name)
    payload = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def official_ratio(record):
    if record.get("net_path") not in (None, ""):
        return float(record["net_path"])
    return net_path_ratio(record["displacement"], record["path_length"])


def _optional_number(record, key):
    value = record.get(key)
    return None if value in (None, "") else float(value)


def _record(variant, seed, seconds, config, observed, source, wall_seconds=None):
    contacts = observed.get("contact_fraction")
    if isinstance(contacts, (list, tuple, np.ndarray)):
        contacts = float(np.mean(contacts))
    return {
        "variant": variant, "seed": seed, "simulated_seconds": seconds,
        "block_ms": config.block_ms, "contact_threshold": config.contact_threshold,
        "displacement": float(observed["displacement"]),
        "path_length": float(observed["path_length"]),
        "net_path": official_ratio(observed),
        "energy_spent": float(observed["energy_spent"]),
        "contact_fraction": None if contacts in (None, "") else float(contacts),
        "mean_forward_flow": _optional_number(observed, "mean_forward_flow"),
        "mean_abs_lateral_flow": _optional_number(observed, "mean_abs_lateral_flow"),
        "mean_abs_rotation": _optional_number(observed, "mean_abs_rotation"),
        "wall_collisions": _optional_number(observed, "wall_collisions"),
        "wall_seconds": wall_seconds, "source": source,
        "config_fingerprint": fingerprint(config),
    }


def _load_local(out_dir, variant, seed, seconds, config):
    path = out_dir / f"{variant}_seed{seed}.json"
    if not path.exists():
        return None
    row = json.loads(path.read_text(encoding="utf-8"))
    if (row.get("simulated_seconds") == seconds
            and row.get("config_fingerprint") == fingerprint(config)):
        return row
    return None


def _load_phase4_history(variant, seed, seconds, config):
    if variant != "phase4_default" or seconds != 300 or config.backend != "cython":
        return None
    path = HERE / "results" / f"baseline_seed{seed}.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("seed") != seed or record.get("simulated_seconds") != seconds:
        return None
    row = _record(variant, seed, seconds, config, record, str(path),
                  record.get("wall_seconds"))
    row["provenance_note"] = (
        "CSV/JSON legado sem impressão digital da configuração; a fase 4 usou "
        "Config(seed=seed). Contato e fluxo não foram gravados.")
    return row


def _load_phase5_history(variant, seed, seconds, config, source_dir):
    if variant != "phase5_off_default" or seconds != 300:
        return None
    candidates = [Path(source_dir) / f"off_seed{seed}"]
    if seed == 42:
        candidates.append(HERE / "results" / "phase5_full_probe" / "off_seed42")
    for stem in candidates:
        path = stem.with_suffix(".csv")
        checkpoint = stem.with_suffix(".pkl")
        if not path.exists() or not checkpoint.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            matches = [row for row in csv.DictReader(handle)
                       if float(row["t_s"]) == seconds and row.get("condition") == "off"
                       and int(row["seed"]) == seed]
        if not matches:
            continue
        # Checkpoints are local files produced by this simulator. The exact
        # configuration and disabled plasticity must match before reuse.
        with checkpoint.open("rb") as handle:
            state = pickle.load(handle)
        if (state.get("config") != config or state.get("sim", {}).get("plasticity") is not False
                or abs(state.get("sim", {}).get("simulated_ms", -1) / 1000 - seconds) > 1e-6):
            continue
        observed = matches[-1]
        wall_seconds = (_optional_number(observed, "wall_s_per_sim_min") or 0) * seconds / 60
        return _record(variant, seed, seconds, config, observed, str(path), wall_seconds)
    return None


def collect(variant, seed, seconds, out_dir, source_dir, backend="cython",
            run_missing=True, reuse_legacy=True):
    """Reusa evidência compatível ou mede uma condição em um processo novo."""
    config = config_for(variant, seed, backend)
    row = _load_local(out_dir, variant, seed, seconds, config)
    if row is None and reuse_legacy:
        row = (_load_phase4_history(variant, seed, seconds, config)
               or _load_phase5_history(variant, seed, seconds, config, source_dir))
    if row is None and run_missing:
        started = time.perf_counter()
        sim = Simulation(config, headless=True)
        sim.plasticity = variant.startswith("phase4")
        result = sim.run_headless(seconds)
        row = _record(variant, seed, seconds, config, result, "new_run",
                      round(time.perf_counter() - started, 2))
    if row is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{variant}_seed{seed}.json").write_text(
            json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
    return row


def write_comparison(rows, missing, output):
    output.mkdir(parents=True, exist_ok=True)
    lookup = {(row["variant"], row["seed"]): row for row in rows}
    for row in rows:
        reference = lookup.get(("phase5_off_default", row["seed"]))
        row["delta_vs_phase5_off_pp"] = (
            100 * (row["net_path"] - reference["net_path"]) if reference else None)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*FIELDS, "delta_vs_phase5_off_pp",
                                                   "config_fingerprint", "provenance_note"])
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in writer.fieldnames} for row in rows)
    grouped = {}
    for variant in VARIANTS:
        matched = [row["net_path"] for row in rows if row["variant"] == variant]
        grouped[variant] = {"seeds": len(matched),
                            "mean_net_path": float(np.mean(matched)) if matched else None}
    report = {
        "official_metric": "sum(per_episode_endpoint_distance) / sum(per_physics_step_distance)",
        "units": "fraction; multiply by 100 for percent",
        "protocol": "300 s per seed; same arena; phase 4 body block 10 ms; phase 5 block 20 ms",
        "historical_limit": "Legacy phase-4 files have no config fingerprint, contacts or flow.",
        "rows": rows, "mean_by_variant": grouped, "missing": missing,
    }
    (output / "comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS,
                        default=list(REQUIRED_VARIANTS))
    parser.add_argument("--backend", choices=("cython", "numpy"), default="cython")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "baseline_audit")
    parser.add_argument("--phase5-source", type=Path,
                        default=HERE / "results" / "audit_phase5_off")
    parser.add_argument("--summary-only", action="store_true",
                        help="só reúne resultados existentes; não inicia simulações")
    parser.add_argument("--fresh", action="store_true",
                        help="ignora arquivos históricos, mas preserva cache da auditoria")
    args = parser.parse_args()
    if args.seconds <= 0 or not args.seeds:
        parser.error("duração e seeds devem ser positivas")
    rows, missing = [], []
    for variant in args.variants:
        for seed in args.seeds:
            row = collect(variant, seed, args.seconds, args.output, args.phase5_source,
                          args.backend, not args.summary_only, not args.fresh)
            if row is None:
                missing.append({"variant": variant, "seed": seed})
                print(f"PENDENTE {variant} seed={seed}", flush=True)
            else:
                rows.append(row)
                print(f"{variant} seed={seed}: net/path={100 * row['net_path']:.3f}% "
                      f"fonte={row['source']}", flush=True)
            write_comparison(rows, missing, args.output)
    print(json.dumps({"complete": not missing, "missing": missing,
                      "comparison": str(args.output / 'comparison.csv')}, ensure_ascii=False))


if __name__ == "__main__":
    main()
