"""Busca curta de andaime motor por física corporal, sem alvo de marcha."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from types import SimpleNamespace

from .body import Body, LEG_NAMES
from .config import Config
from .metrics import Metrics
from .motor import MotorDecoder, POOL_NAMES
from .fast_motor import build_probe
from .world import World

LOCAL_OSCILLATOR = {
    "drive": 5.346911716749979, "recurrence": 8.43198403333492,
    "recruitment": 9.949371367101634,
    "cross_inhibition": 1.4555073679116919, "proprio": .7957043467686156,
    "adaptation_a": .005423741565801697,
    "adaptation_d": 11.211611327719453,
    "motor": 4.0, "motor_inhibition": 2.0, "motor_bias": 0.0,
    "inhibitory_bias": .3711401241159191,
}


class FastEvaluator:
    def __init__(self, seed=42, interleg_opposite_prob=0.5,
                 cross_force=None, normalize_neighbors=False, cross_tau_ms=150.0):
        import pyximport
        pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)
        from .fast_kernel import simulate
        self.simulate = simulate
        self.seed = seed
        self.probe = build_probe(legs=6, seed=seed, backend="cython",
                                 interleg_opposite_prob=interleg_opposite_prob)
        order = np.argsort(self.probe.pre, kind="stable")
        self.starts = np.r_[0, np.cumsum(np.bincount(self.probe.pre, minlength=768))].astype(np.int32)
        self.targets = self.probe.post[order].astype(np.int32)
        self.kinds = self.probe.edge_kinds[order]
        self.signs = self.probe.fixed_sign[order]
        self.degrees = self.probe.edge_degrees[order]
        self.cross_force = cross_force
        self.normalize_neighbors = normalize_neighbors
        self.cross_tau_ms = cross_tau_ms

    def build_weights(self, params):
        gains = np.array([params["recurrence"], params["recruitment"], params["motor"],
                          params["cross_inhibition"], params["motor_inhibition"], params["coupling"],
                          params["proprio"], params.get("mutual_inhibition", 0.0)])
        weights = (self.signs * gains[self.kinds] / np.sqrt(self.degrees)).astype(float)
        if self.cross_force is not None:
            from .tripod_audit import neighbor_normalizers
            cross = self.kinds == 7
            scale = (neighbor_normalizers()[self.targets[cross] // 128]
                     if self.normalize_neighbors else 1.0)
            weights[cross] = -self.cross_force * scale / np.sqrt(self.degrees[cross])
        return weights

    def evaluate(self, params, seconds=15):
        weights = self.build_weights(params)
        world = World()
        body = Body(world)
        metrics = Metrics(body)
        motor_model = SimpleNamespace(
            group=self.probe.neurons,
            motor_pools={leg: {name: self.probe.pools[index].reshape(4, 8)[pool]
                               for pool, name in enumerate(POOL_NAMES)}
                         for index, leg in enumerate(LEG_NAMES)})
        decoder = MotorDecoder(motor_model, Config(motor_gain=.18))
        state = None
        samples = []
        previous = np.zeros(768, dtype=np.int64)
        for _ in range(round(seconds / .02)):
            feet = (body.angles[:, 1] < 0).astype(float)
            _, _, _, _, state = self.simulate(
                self.starts, self.targets, weights, 6, 20, self.seed,
                params["drive"], params["proprio"], 2.5, state=state,
                external_angles=body.angles.ravel(), external_contacts=feet,
                return_state=True, edge_kinds=self.kinds,
                cross_tau_ms=self.cross_tau_ms,
                motor_bias=params["motor_bias"],
                inhibitory_bias=params["inhibitory_bias"],
                adaptation_a=params["adaptation_a"],
                adaptation_d=params["adaptation_d"],
                contact_gain=params.get("contact_gain", 1.0),
                interleg_contact_gain=params.get("interleg_contact_gain", 0.0))
            delta = state["total"] - previous
            samples.append(delta[self.probe.premotor].sum(axis=-1))
            previous = state["total"].copy()
            torques = decoder.update(state["total"], .02)
            body.step(.02, torques, feet > .5, world)
            metrics.update(body, torques, feet)
        observed = metrics.snapshot(body)
        counts = np.asarray(samples).transpose(1, 2, 3, 0)
        rates = counts.sum(axis=-1) / (12 * seconds)
        net_energy = observed["displacement"] / max(observed["energy_spent"], 1e-9)
        stability = (1 + observed["mean_abs_lateral_flow"] / max(observed["mean_speed"], 1e-9)
                     + observed["mean_abs_rotation"] * seconds
                     + observed["wall_collisions"])
        correlations = []
        for leg in range(6):
            for joint in range(2):
                first, second = counts[leg, joint]
                if np.std(first) and np.std(second):
                    correlations.append(float(np.corrcoef(first, second)[0, 1]))
        return {"fitness": round(float(net_energy / stability), 4),
                "net_path": observed["net_path"], "net_energy": round(net_energy, 4),
                "displacement": observed["displacement"],
                "path_length": observed["path_length"],
                "forward_flow": observed["mean_forward_flow"],
                "mean_speed": observed["mean_speed"],
                "mean_abs_torque": observed["mean_abs_torque"],
                "joint_angle_sd": np.sqrt(np.asarray(observed["joint_variance"])).round(4).tolist(),
                "contact_fraction": observed["contact_fraction"],
                "lateral": observed["mean_abs_lateral_flow"],
                "rotation": observed["mean_abs_rotation"],
                "collisions": observed["wall_collisions"],
                "rate_min": round(float(rates.min()), 2),
                "rate_max": round(float(rates.max()), 2),
                "phase": observed["neighbor_phase_correlation"],
                "contact_correlation": observed["contact_correlation"],
                "saturation": round(decoder.saturation_fraction(), 5),
                "antagonist": round(float(np.mean(correlations)), 3) if correlations else 0.0}



BOUNDS = {"drive": (5.0, 5.8), "recurrence": (7.5, 9.0),
          "recruitment": (9.0, 11.0), "cross_inhibition": (1.0, 1.8),
          "coupling": (0.0, 8.0), "proprio": (0.5, 1.2)}


def search(seconds=15, generations=4, population=12, budget_s=180, seed=42):
    started = time.perf_counter()
    evaluator = FastEvaluator(seed=seed)
    rng = np.random.default_rng(seed + 1000)
    incumbent = {**LOCAL_OSCILLATOR, "coupling": 2.0}
    rows = []
    best = None
    for generation in range(generations):
        for candidate in range(population):
            if time.perf_counter() - started >= min(240, budget_s):
                break
            if generation == 0 and candidate == 0:
                params = incumbent.copy()
            elif generation == 0:
                params = {**LOCAL_OSCILLATOR,
                          **{name: float(rng.uniform(*bounds)) for name, bounds in BOUNDS.items()}}
            else:
                params = {**LOCAL_OSCILLATOR,
                          **{name: float(np.clip(incumbent[name] + rng.normal(0, .12 * (hi-lo)), lo, hi))
                             for name, (lo, hi) in BOUNDS.items()}}
            result = evaluator.evaluate(params, seconds)
            valid = result["rate_min"] >= 1 and result["rate_max"] <= 80
            result["valid"] = valid
            row = {"generation": generation, "candidate": candidate,
                   "params": params, **result}
            rows.append(row)
            if valid and (best is None or result["fitness"] > best["fitness"]):
                best = row
                incumbent = params.copy()
        if time.perf_counter() - started >= min(240, budget_s):
            break
    return {"wall_seconds": round(time.perf_counter() - started, 3),
            "evaluated": len(rows), "best": best,
            "physics": "Body.step + MotorDecoder + sensory feedback every 20 ms",
            "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--budget-s", type=float, default=180)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "results" / "fast_search_v1.json")
    args = parser.parse_args()
    if not 10 <= args.seconds <= 20 or not 0 < args.budget_s <= 240:
        parser.error("episódios devem ter 10–20 s e busca no máximo 240 s reais")
    result = search(args.seconds, args.generations, args.population, args.budget_s)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("wall_seconds", "evaluated", "best")}), flush=True)


if __name__ == "__main__":
    main()
