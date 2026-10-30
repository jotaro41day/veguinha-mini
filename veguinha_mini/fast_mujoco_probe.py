"""Short closed-loop test of the established 768-cell motor circuit in MuJoCo."""

import argparse
import csv
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .body import LEG_NAMES
from .config import Config
from .motor import MotorDecoder, POOL_NAMES
from .mujoco_body import MujocoBody
from .search_fast import FastEvaluator, LOCAL_OSCILLATOR


def run(seed, cross_force, motor_gain, seconds=8, load_gain=0,
        stance_load_gain=0, trace_path=None, excitatory_coupling=None,
        normalize_neighbors=True, interleg_contact_gain=0,
        randomize_initial_phase=False, noise_amp=2.5):
    if (seconds <= 2 or motor_gain <= 0 or cross_force < 0
            or load_gain < 0 or stance_load_gain < 0 or interleg_contact_gain < 0):
        raise ValueError("ensaio requer duração >2 s e ganhos válidos")
    evaluator = FastEvaluator(seed=seed, cross_force=cross_force,
                              normalize_neighbors=normalize_neighbors, cross_tau_ms=40)
    if excitatory_coupling is None:
        excitatory_coupling = 0.0 if cross_force == 0 else 2.0
    if excitatory_coupling < 0:
        raise ValueError("acoplamento excitatório deve ser não negativo")
    params = {**LOCAL_OSCILLATOR,
              "coupling": excitatory_coupling,
              "mutual_inhibition": cross_force}
    weights = evaluator.build_weights(params)
    body = MujocoBody()
    model = SimpleNamespace(
        group=evaluator.probe.neurons,
        motor_pools={leg: {name: evaluator.probe.pools[index].reshape(4, 8)[pool]
                           for pool, name in enumerate(POOL_NAMES)}
                     for index, leg in enumerate(LEG_NAMES)})
    decoder = MotorDecoder(model, Config(motor_gain=motor_gain))
    leg_weight = body.model.body_mass.sum() * 9.81 / 6
    state = None
    initial_phase_spread_deg = None
    if randomize_initial_phase:
        # Free run: same initial state for each paired force, but no interleg
        # edges, sensory input or body motion. It randomizes oscillator phases
        # through independent neural noise rather than injecting a target gait.
        free_weights = weights.copy()
        free_weights[np.isin(evaluator.kinds, (5, 7))] = 0.0
        free_counts = []
        zero_angles = np.zeros(12)
        zero_feet = np.zeros(6)
        for _ in range(300):
            counts, _, _, _, state = evaluator.simulate(
                evaluator.starts, evaluator.targets, free_weights, 6, 20, seed,
                params["drive"], 0.0, noise_amp, state=state,
                external_angles=zero_angles, external_contacts=zero_feet,
                return_state=True, edge_kinds=evaluator.kinds,
                cross_tau_ms=evaluator.cross_tau_ms, motor_bias=params["motor_bias"],
                inhibitory_bias=params["inhibitory_bias"],
                adaptation_a=params["adaptation_a"],
                adaptation_d=params["adaptation_d"], contact_gain=0.0)
            free_counts.append(counts[..., 0])
            decoder.update(state["total"], .02)
        # A single Fourier frequency gives comparable circular phases across legs.
        trace = np.asarray(free_counts[-200:])[:, :, 0, 0] - np.asarray(free_counts[-200:])[:, :, 0, 1]
        spectrum = np.fft.rfft(trace - trace.mean(axis=0), axis=0)
        frequencies = np.fft.rfftfreq(len(trace), .02)
        band = (frequencies >= 1.0) & (frequencies <= 3.33)
        peak = np.flatnonzero(band)[np.argmax((abs(spectrum[band]) ** 2).sum(axis=1))]
        phases = np.angle(spectrum[peak])
        initial_phase_spread_deg = round(float(np.rad2deg(np.sqrt(
            -2 * np.log(max(abs(np.exp(1j * phases).mean()), 1e-12))))), 2)
    torque_series, contact_series, forward_series = [], [], []
    neural_series, height_series = [], []
    fell = False
    started = time.perf_counter()
    for _ in range(round(seconds / .02)):
        _, _, _, _, state = evaluator.simulate(
            evaluator.starts, evaluator.targets, weights, 6, 20, seed,
            params["drive"], params["proprio"], noise_amp, state=state,
            external_angles=body.angles.ravel(),
            external_contacts=body.contacts.astype(float), return_state=True,
            external_loads=np.clip(body.normal_load / leg_weight, 0, 2),
            load_reflex_gain=load_gain,
            stance_load_gain=stance_load_gain,
            interleg_contact_gain=interleg_contact_gain,
            edge_kinds=evaluator.kinds, cross_tau_ms=evaluator.cross_tau_ms,
            motor_bias=params["motor_bias"],
            inhibitory_bias=params["inhibitory_bias"],
            adaptation_a=params["adaptation_a"],
            adaptation_d=params["adaptation_d"], contact_gain=1.0)
        if trace_path:
            neural_series.append((state["total"] - decoder.previous)[
                evaluator.probe.premotor].sum(axis=-1).copy())
        torque = decoder.update(state["total"], .02)
        body.step(.02, torque)
        torque_series.append(torque)
        contact_series.append(body.contacts.copy())
        forward_series.append(float(body.velocity @ [math.cos(body.heading),
                                                      math.sin(body.heading)]))
        if trace_path:
            height_series.append(float(body.data.qpos[2]))
        fell |= body.fell
    if trace_path:
        target = Path(trace_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, torque=np.asarray(torque_series),
                            contact=np.asarray(contact_series),
                            forward=np.asarray(forward_series),
                            premotor=np.asarray(neural_series),
                            height=np.asarray(height_series))
    window_start_s = max(2.0, seconds - 10.0)
    late = slice(round(window_start_s / .02), None)
    torque = np.asarray(torque_series)[late]
    contacts = np.asarray(contact_series)[late]
    group_a = contacts[:, [0, 4, 2]].mean(axis=1)
    group_b = contacts[:, [3, 1, 5]].mean(axis=1)
    alternation = (-float(np.corrcoef(group_a, group_b)[0, 1])
                   if np.std(group_a) > 1e-9 and np.std(group_b) > 1e-9 else 0.0)
    return {"seed": seed, "cross_force": cross_force, "motor_gain": motor_gain,
            "load_gain": load_gain, "stance_load_gain": stance_load_gain,
            "excitatory_coupling": excitatory_coupling,
            "normalize_neighbors": normalize_neighbors,
            "interleg_contact_gain": interleg_contact_gain,
            "randomize_initial_phase": randomize_initial_phase,
            "initial_phase_spread_deg": initial_phase_spread_deg,
            "seconds": seconds, "window_start_s": window_start_s,
            "forward_m_s": round(float(np.mean(np.asarray(forward_series)[late])), 5),
            "support": round(float(contacts.mean()), 4),
            "three_feet": round(float((contacts[:, [0, 4, 2]].all(axis=1)
                                         | contacts[:, [3, 1, 5]].all(axis=1)).mean()), 4),
            "alternation": round(alternation, 4),
            "hip_rms": round(float(np.sqrt(np.mean(torque[:, :, 0] ** 2))), 4),
            "knee_rms": round(float(np.sqrt(np.mean(torque[:, :, 1] ** 2))), 4),
            "fall": int(fell), "wall_s": round(time.perf_counter() - started, 3)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--forces", type=float, nargs="+", default=[0, 18])
    parser.add_argument("--gains", type=float, nargs="+", default=[.18, .36, .54])
    parser.add_argument("--load-gains", type=float, nargs="+", default=[0])
    parser.add_argument("--stance-load-gains", type=float, nargs="+", default=[0])
    parser.add_argument("--excitatory-coupling", type=float)
    parser.add_argument("--no-neighbor-normalization", action="store_true")
    parser.add_argument("--interleg-contact-gain", type=float, default=0)
    parser.add_argument("--output", default="veguinha_mini/results/fast_mujoco_gain_v1.csv")
    args = parser.parse_args()
    rows = []
    for seed in args.seeds:
        for gain in args.gains:
            for force in args.forces:
                for load_gain in args.load_gains:
                    for stance_gain in args.stance_load_gains:
                        row = run(seed, force, gain, args.seconds, load_gain,
                                  stance_gain, excitatory_coupling=args.excitatory_coupling,
                                  normalize_neighbors=not args.no_neighbor_normalization,
                                  interleg_contact_gain=args.interleg_contact_gain)
                        rows.append(row)
                        print(row, flush=True)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
