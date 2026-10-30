"""Ajusta somente a oscilação local, usando ensaios curtos de uma perna."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .fast_motor import build_probe


def oscillation_score(counts, seconds, total=None):
    """Período, potência rítmica e fase flexor/extensor em duas metades."""
    rows = []
    series = np.asarray(counts[0], dtype=float)
    for joint in range(2):
        for half in (0, 1):
            window = series[joint, :, half * series.shape[-1] // 2:
                            (half + 1) * series.shape[-1] // 2].copy()
            window -= window.mean(axis=1, keepdims=True)
            spectrum = np.fft.rfft(window * np.hanning(window.shape[-1]), axis=1)
            frequencies = np.fft.rfftfreq(window.shape[-1], .05)
            band = (frequencies >= 1) & (frequencies <= 3.33)
            if not band.any():
                return {"pass": False, "score": -1e9, "windows": []}
            joint_power = (np.abs(spectrum) ** 2).sum(axis=0)
            index = np.flatnonzero(band)[np.argmax(joint_power[band])]
            spectral_total = (np.abs(spectrum[:, (frequencies >= .5) & (frequencies <= 8)]) ** 2).sum()
            # A tapered oscillation occupies the peak and its adjacent FFT bins.
            rhythm = float((np.abs(spectrum[:, max(1, index-1):index+2]) ** 2).sum()
                           / max(spectral_total, 1e-9))
            phase = float(np.angle(spectrum[0, index] * spectrum[1, index].conjugate(), deg=True))
            period = float(1 / frequencies[index])
            rows.append({"joint": joint, "half": half, "period_s": round(period, 3),
                         "phase_deg": round(phase, 1), "rhythm": round(rhythm, 4)})
    rates = series.sum(axis=-1) / (12 * seconds)
    # The cadence criterion is scored here, never used as the search fitness
    # for whole-body motion.
    phase_quality = [max(0.0, -np.cos(np.deg2rad(row["phase_deg"])))
                     for row in rows]
    rhythm_quality = [min(1.0, row["rhythm"] / .1) for row in rows]
    rate_ok = bool((rates > 1).all() and (rates < 80).all())
    pool_rates = None
    if total is not None:
        pool_rates = {
            "motor": np.asarray(total[:32]).reshape(4, 8).sum(axis=1) / 8 / seconds,
            "premotor": np.asarray(total[32:80]).reshape(4, 12).sum(axis=1) / 12 / seconds,
            "inhibitory": np.asarray(total[80:120]).reshape(4, 10).sum(axis=1) / 10 / seconds,
        }
        rate_ok &= all((values > 1).all() and (values < 80).all()
                       for values in pool_rates.values())
    passed = (rate_ok and all(abs(row["phase_deg"]) >= 120 for row in rows)
              and all(row["rhythm"] >= .12 for row in rows))
    return {"pass": passed, "score": (round(float(min(phase_quality) +
                                            min(rhythm_quality) +
                                            np.mean(phase_quality) / 2), 4)
                                       if rate_ok else -1000.0),
            "windows": rows, "rates_hz": np.round(rates, 2).tolist(),
            "pool_rates_hz": ({key: np.round(value, 2).tolist()
                               for key, value in pool_rates.items()}
                              if pool_rates is not None else None)}


def tune(candidates=350, seconds=10, budget_s=150, seed=42):
    import pyximport
    pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)
    from .fast_kernel import simulate
    started = time.perf_counter()
    probe = build_probe(legs=1, seed=seed, backend="cython")
    order = np.argsort(probe.pre, kind="stable")
    starts = np.r_[0, np.cumsum(np.bincount(probe.pre, minlength=128))].astype(np.int32)
    targets = probe.post[order].astype(np.int32)
    kinds = probe.edge_kinds[order]
    signs = probe.fixed_sign[order]
    degrees = probe.edge_degrees[order]
    rng = np.random.default_rng(seed + 4000)
    rows = []
    baseline = {"drive": 5.5, "recurrence": 5.0, "recruitment": 5.0,
                "cross_inhibition": .5, "proprio": 1.0,
                "adaptation_a": .003, "adaptation_d": 8.0,
                "motor": 4.0, "motor_inhibition": 2.0,
                "motor_bias": 0.0, "inhibitory_bias": -2.0}
    for candidate in range(candidates):
        if time.perf_counter() - started >= min(240, budget_s):
            break
        params = baseline.copy() if candidate == 0 else {
            "drive": float(rng.uniform(4.7, 7.0)),
            "recurrence": float(rng.uniform(1, 9)),
            "recruitment": float(rng.uniform(1, 12)),
            "cross_inhibition": float(rng.uniform(.1, 1.5)),
            "proprio": float(rng.uniform(.5, 4)),
            "adaptation_a": float(rng.uniform(.001, .012)),
            "adaptation_d": float(rng.uniform(3, 18)),
            "motor": 4.0, "motor_inhibition": 2.0,
            "motor_bias": 0.0,
            "inhibitory_bias": float(rng.uniform(-2, 3)),
        }
        gains = np.array([params["recurrence"], params["recruitment"], params["motor"],
                          params["cross_inhibition"], params["motor_inhibition"],
                          6, params["proprio"]])
        weights = (signs * gains[kinds] / np.sqrt(degrees)).astype(float)
        observations = []
        for validation_seed in (42, 43, 44):
            counts, _, _, _, state = simulate(
                starts, targets, weights, 1, int(seconds * 1000),
                validation_seed, params["drive"], params["proprio"], 2.5,
                return_state=True, inhibitory_bias=params["inhibitory_bias"],
                motor_bias=params["motor_bias"],
                adaptation_a=params["adaptation_a"],
                adaptation_d=params["adaptation_d"], edge_kinds=kinds)
            observations.append(oscillation_score(counts, seconds, state["total"]))
        row = {"candidate": candidate, "params": params,
               "pass_all": all(item["pass"] for item in observations),
               "score": min(item["score"] for item in observations),
               "observations": observations}
        rows.append(row)
    ranked = sorted(rows, key=lambda item: (item["pass_all"], item["score"]), reverse=True)
    return {"wall_seconds": round(time.perf_counter() - started, 3),
            "evaluated": len(rows), "best": ranked[0], "passing": sum(row["pass_all"] for row in rows),
            "top": ranked[:10]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, default=350)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--budget-s", type=float, default=150)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "results" / "fast_tune_v1.json")
    args = parser.parse_args()
    if not 10 <= args.seconds <= 20 or not 0 < args.budget_s <= 240:
        parser.error("episódios 10–20 s; orçamento máximo 240 s reais")
    result = tune(args.candidates, args.seconds, args.budget_s)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("wall_seconds", "evaluated", "passing", "best")}), flush=True)


if __name__ == "__main__":
    main()
