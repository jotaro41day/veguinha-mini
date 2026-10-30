"""VNC/descendentes isolados com suporte sensorial mínimo e o mesmo corpo 2D."""

import argparse
import json
import math
import resource
import time

from .config import Config
from .regions import Pathway, Region
from .sim import Simulation


LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
COUPLINGS = (("lf", "lm"), ("lm", "lh"), ("rf", "rm"), ("rm", "rh"),
             ("lf", "rf"), ("lm", "rm"), ("lh", "rh"))


def isolated_config(seed=42, tonic_drive=2.0, backend="cython", **changes):
    """Configura 640 células DN/VNC e 229 células mínimas de suporte.

    O drive tônico é uma corrente basal uniforme apenas nos descendentes.
    Não há padrão de marcha ou comando de velocidade embutido.
    """
    if not math.isfinite(tonic_drive) or not -4 <= tonic_drive <= 8:
        raise ValueError("drive descendente deve ser finito e ficar entre -4 e 8")
    if changes.get("motor_scaffold"):
        changes.setdefault("vnc_hebb_rate", 0.0)
        changes.setdefault("motor_gain", 0.045)
    regions = (
        Region("al_glom", 8, 0, 1.0),
        Region("kc", 1, 0, -1.0),
        Region("cx", 8, 3, 4.7),
        Region("dn", 64, 16, 4.7 + tonic_drive),
        *(Region(f"vnc_{leg}", 96, 32, 5.0) for leg in LEGS),
        Region("seg", 4, 2, 4.2),
        Region("lh", 4, 2, 4.7),
        Region("value", 8, 3, 4.5),
        Region("dan", 8, 0, 4.0),
        Region("comparator", 12, 4, 4.3),
    )
    pathways = [Pathway("dn", "dn", 48, 0.45),
                Pathway("value", "dan", 8, 0.55),
                Pathway("comparator", "dan", 12, 0.45),
                Pathway("dn", "comparator", 32, 0.4),
                Pathway("cx", "dn", 8, 0.6)]
    for leg in LEGS:
        pathways.extend((Pathway("dn", f"vnc_{leg}", 48, 0.45),
                         Pathway(f"vnc_{leg}", f"vnc_{leg}", 80, 0.65)))
    for a, b in COUPLINGS:
        pathways.extend((Pathway(f"vnc_{a}", f"vnc_{b}", 30, 0.35),
                         Pathway(f"vnc_{b}", f"vnc_{a}", 30, 0.35)))
    changes.setdefault("block_ms", 20)
    return Config(seed=seed, backend=backend, phase5_enabled=True,
                  regions=regions, pathways=tuple(pathways), **changes)


def isolated_simulation(seed=42, tonic_drive=2.0, backend="cython",
                        plasticity=False, reward_mode="coupled", headless=True,
                        **config_changes):
    sim = Simulation(isolated_config(seed=seed, tonic_drive=tonic_drive,
                                     backend=backend, **config_changes),
                     headless=headless, reward_mode=reward_mode)
    sim.plasticity = plasticity
    return sim


def main():
    parser = argparse.ArgumentParser(description="Benchmark do cordão ventral isolado")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tonic-drive", type=float, default=2)
    parser.add_argument("--backend", choices=("numpy", "cython"), default="cython")
    parser.add_argument("--warmup-s", type=float, default=.1,
                        help="compila o código Brian2 antes de iniciar o cronômetro")
    args = parser.parse_args()
    if args.seconds <= 0 or args.warmup_s <= 0:
        parser.error("--seconds e --warmup-s devem ser positivos")
    started = time.perf_counter()
    sim = isolated_simulation(seed=args.seed, tonic_drive=args.tonic_drive,
                              backend=args.backend)
    built = time.perf_counter()
    sim.run_headless(args.warmup_s)
    warmed = time.perf_counter()
    cpu_started = time.process_time()
    result = sim.run_headless(args.seconds)
    elapsed = time.perf_counter() - warmed
    cpu_elapsed = time.process_time() - cpu_started
    result.update({"cells": len(sim.model.group), "motor_core_cells": 640,
                   "synapses": len(sim.model.synapses),
                   "construction_wall_s": round(built - started, 3),
                   "warmup_wall_s": round(warmed - built, 3),
                   "warmup_sim_s": args.warmup_s,
                   "simulation_wall_s": round(elapsed, 3),
                   "simulation_cpu_s": round(cpu_elapsed, 3),
                   "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
                   "realtime_percent": round(100 * args.seconds / elapsed, 2)})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
