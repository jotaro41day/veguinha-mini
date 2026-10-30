"""Ensaios sensoriais curtos, com parâmetros fixos e limite de tempo real."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import tempfile
import time


SEEDS = (42, 43, 44)
SOURCES = ("fast_sensor_audit.py", "fast_motor.py", "fast_kernel.pyx",
           "fast_tune.py", "search_fast.py", "body.py", "motor.py",
           "metrics.py", "config.py", "sensory.py", "network.py")


def _save_atomic(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _probe_result(seed, legs, pair_kind, seconds, params):
    import numpy as np
    import pyximport

    from .fast_motor import _leg_phase_lags, build_probe
    from .fast_tune import oscillation_score

    pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)
    from .fast_kernel import simulate

    probe = build_probe(legs=legs, seed=seed, backend="cython",
                        pair_kind=pair_kind, **params)
    order = np.argsort(probe.pre, kind="stable")
    starts = np.r_[0, np.cumsum(np.bincount(probe.pre, minlength=128 * legs))]
    counts, angles, contacts, _, state = simulate(
        starts.astype(np.int32), probe.post[order].astype(np.int32),
        np.asarray(probe.recurrent.w[:], dtype=float)[order],
        legs, round(seconds * 1000), seed, params["drive"], params["proprio"],
        2.5, return_state=True, inhibitory_bias=params["inhibitory_bias"],
        motor_bias=params["motor_bias"], adaptation_a=params["adaptation_a"],
        adaptation_d=params["adaptation_d"], edge_kinds=probe.edge_kinds[order],
        contact_gain=params["contact_gain"],
        interleg_contact_gain=params["interleg_contact_gain"])
    scores = [oscillation_score(counts[leg:leg + 1], seconds,
                                state["total"][leg * 128:(leg + 1) * 128])
              for leg in range(legs)]
    return {"neurons": 128 * legs, "synapses": len(order),
            "pass_local": all(score["pass"] for score in scores),
            "oscillation_by_leg": scores,
            "interleg_lag_deg": _leg_phase_lags(counts, [(0, 1)] if legs == 2 else []),
            "foot_contact_correlation": (
                float(np.corrcoef(contacts[1], contacts[3])[0, 1])
                if legs == 2 and np.std(contacts[1]) and np.std(contacts[3]) else None),
            "joint_angle_sd": np.std(angles, axis=1).round(4).tolist(),
            "contact_fraction": np.mean(contacts, axis=1).round(4).tolist()}


def run_audit(stage, seconds, proprio, contact_gain, coupling,
              interleg_contact_gain, opposite_prob, budget_s, output):
    """Executa seeds sequencialmente; preserva resultados completos no timeout."""
    if (stage not in ("local", "pairs", "body") or not 10 <= seconds <= 20
            or not 0 < budget_s <= 240
            or not all(math.isfinite(gain) and gain >= 0
                       for gain in (proprio, contact_gain, coupling, interleg_contact_gain))
            or not math.isfinite(opposite_prob) or not 0 <= opposite_prob <= 1):
        raise ValueError("estágio, duração, orçamento ou ganhos inválidos")
    started = time.perf_counter()
    root = Path(__file__).resolve().parent
    output = Path(output)
    report = {"stage": stage, "seeds": list(SEEDS), "simulated_seconds": seconds,
              "budget_seconds": budget_s, "status": "running", "rows": [],
              "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                for name in SOURCES}}
    previous_handler = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise TimeoutError("limite de tempo real atingido")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, budget_s)
    try:
        _save_atomic(output, report)
        from .search_fast import FastEvaluator, LOCAL_OSCILLATOR

        params = {**LOCAL_OSCILLATOR, "proprio": proprio,
                  "contact_gain": contact_gain, "coupling": coupling,
                  "interleg_contact_gain": interleg_contact_gain,
                  "interleg_opposite_prob": opposite_prob}
        report["params"] = params
        _save_atomic(output, report)
        conditions = ("neighbor", "contralateral") if stage == "pairs" else (stage,)
        for seed in SEEDS:
            for condition in conditions:
                row_started = time.perf_counter()
                report["in_progress"] = {"seed": seed, "condition": condition}
                _save_atomic(output, report)
                if stage == "body":
                    result = FastEvaluator(
                        seed=seed, interleg_opposite_prob=opposite_prob
                    ).evaluate(params, seconds=seconds)
                else:
                    result = _probe_result(seed, 2 if stage == "pairs" else 1,
                                           condition if stage == "pairs" else "neighbor",
                                           seconds, params)
                wall = time.perf_counter() - row_started
                row = {"seed": seed, "condition": condition,
                       "simulated_seconds": seconds, "wall_seconds": round(wall, 4),
                       "realtime_ratio": round(seconds / wall, 2), **result}
                report["rows"].append(row)
                report.pop("in_progress", None)
                report["wall_seconds"] = round(time.perf_counter() - started, 4)
                _save_atomic(output, report)
                outcome = (f"net/path={100 * row['net_path']:.2f}%"
                           if stage == "body" else f"local_pass={row['pass_local']}")
                print(f"seed={seed} {condition}: {outcome}; {wall:.2f}s reais", flush=True)
        report["status"] = "complete"
    except TimeoutError as exc:
        report["status"] = "timeout"
        report["error"] = str(exc)
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        report["wall_seconds"] = round(time.perf_counter() - started, 4)
        _save_atomic(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("local", "pairs", "body"), required=True)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--proprio", type=float, default=.795704)
    parser.add_argument("--contact-gain", type=float, default=1.0)
    parser.add_argument("--coupling", type=float, default=2.0)
    parser.add_argument("--interleg-contact-gain", type=float, default=0.0)
    parser.add_argument("--opposite-prob", type=float, default=0.5)
    parser.add_argument("--budget-s", type=float, default=120)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not 10 <= args.seconds <= 20 or not 0 < args.budget_s <= 240:
        parser.error("episódios devem ter 10–20 s e orçamento real deve ser >0 e <=240 s")
    if not all(math.isfinite(gain) and gain >= 0
               for gain in (args.proprio, args.contact_gain, args.coupling,
                            args.interleg_contact_gain)):
        parser.error("ganhos devem ser finitos e não negativos")
    if not math.isfinite(args.opposite_prob) or not 0 <= args.opposite_prob <= 1:
        parser.error("probabilidade oposta deve ficar entre 0 e 1")
    output = args.output or Path(__file__).parent / "results" / f"fast_sensor_{args.stage}_v1.json"
    result = run_audit(args.stage, args.seconds, args.proprio, args.contact_gain,
                       args.coupling, args.interleg_contact_gain, args.opposite_prob,
                       args.budget_s, output)
    print(f"{result['status']}: {len(result['rows'])} resultados em {output}", flush=True)
    return 124 if result["status"] == "timeout" else 0


if __name__ == "__main__":
    raise SystemExit(main())
