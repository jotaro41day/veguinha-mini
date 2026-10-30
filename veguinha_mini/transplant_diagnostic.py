"""Short, paired open-loop motor-transplant audit on the corrected body.

Recorded neural torques are replayed without modifying the physics.  The
phase-template conditions are diagnostic probes, never inputs to the network.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from .body import Body
from .contact_diagnostic import RESULTS, kernel, summarize
from .phase_lock_audit import _save_atomic
from .tripod_body_diagnostic import PERIOD_S, TRIPOD_PHASE
from .world import World

DT = .002
BLOCK = .02
SEEDS = tuple(range(52, 62))
CONDITIONS = ("replay", "no_coupling", "motor_shuffled", "imposed_matched",
              "imposed_original", "network_hip_imposed_knee",
              "imposed_hip_network_knee", "sine_network_phases",
              "network_template_tripod", "network_template_tripod_rms",
              "sine_random_phases", "sine_reverse",
              "random_mirror", "random_isotropic", "random_friction_swapped",
              "shuffled_mirror")


def _record(seed, name):
    with np.load(RESULTS / f"contact_network_{seed}_{name}.npz") as source:
        return {key: source[key] for key in source.files}


def _phase_templates(record):
    """Return within-leg torque waveforms conditioned on neural hip phase."""
    phases = record["neural_phases"][0]
    torques = record["torques"]
    template = np.empty((64, 6, 2), float)
    indices = np.mod(np.floor((phases[-1000:] + np.pi) * 64 / (2*np.pi)).astype(int), 64)
    for leg in range(6):
        for bin_index in range(64):
            mask = indices[:, leg] == bin_index
            template[bin_index, leg] = torques[-1000:, leg][mask].mean(axis=0) if np.any(mask) else 0
    # Smooth phase bins without altering each joint's phase relationship.
    template = (np.roll(template, 1, axis=0) + 2*template + np.roll(template, -1, axis=0))/4
    return template


def _lookup(template, phase):
    coordinate = np.mod((phase + np.pi) * len(template) / (2*np.pi), len(template))
    low = np.floor(coordinate).astype(int)
    fraction = coordinate - low
    return ((1-fraction)[:, None]*template[low, np.arange(6)]
            + fraction[:, None]*template[(low+1) % len(template), np.arange(6)])


def torque_series(seed, condition):
    if condition in ("random_mirror", "random_isotropic", "random_friction_swapped"):
        condition = "sine_random_phases"
    if condition == "shuffled_mirror": condition = "motor_shuffled"
    network = _record(seed, "rede18")
    raw = network["torques"]
    n = len(raw)
    if condition == "replay": return raw
    if condition == "no_coupling": return _record(seed, "controle")["torques"]
    if condition == "motor_shuffled": return _record(seed, "embaralhado")["torques"]
    rms = np.sqrt(np.mean(raw[-1000:]**2, axis=(0, 1)))
    amp = rms * np.sqrt(2)
    # Same torque RMS in every diagnostic transplant; the original imposed
    # amplitude is shown separately so amplitude does not masquerade as phase.
    calibration = json.loads((RESULTS/'tripod_body_diagnostic_v2.json').read_text())['rows']
    original_amp = np.mean([r['calibration_amp'] for r in calibration
                            if r['condition'] == 'rede18'], axis=0)
    if condition == "imposed_original": amp = original_amp
    time_axis = np.arange(n)*BLOCK
    cycle = 2*np.pi*time_axis/PERIOD_S
    tripod = cycle[:, None] + TRIPOD_PHASE[None, :]
    imposed = np.stack((amp[0]*np.sin(tripod),
                        amp[1]*np.cos(tripod)), axis=-1)
    if condition in ("imposed_matched", "imposed_original"): return imposed
    if condition == "network_hip_imposed_knee":
        return np.stack((raw[:, :, 0], imposed[:, :, 1]), axis=-1)
    if condition == "imposed_hip_network_knee":
        return np.stack((imposed[:, :, 0], raw[:, :, 1]), axis=-1)
    if condition == "sine_network_phases":
        # Instantaneous neural phases preserve the observed interleg drift.
        phase = network['neural_phases'][0]
        return np.stack((amp[0]*np.sin(phase), amp[1]*np.cos(phase)), axis=-1)
    if condition in ("network_template_tripod", "network_template_tripod_rms"):
        template = _phase_templates(network)
        if condition == "network_template_tripod_rms":
            source_rms = np.sqrt(np.mean(raw[-1000:]**2, axis=0))
            template_rms = np.sqrt(np.mean(template**2, axis=0))
            template *= source_rms[None, :, :] / np.maximum(template_rms[None, :, :],1e-6)
        return np.asarray([_lookup(template, p) for p in tripod])
    if condition == "sine_random_phases":
        phase = cycle[:, None] + np.random.default_rng(seed + 47000).uniform(-np.pi,np.pi,6)
        return np.stack((amp[0]*np.sin(phase), amp[1]*np.cos(phase)), axis=-1)
    if condition == "sine_reverse":
        return np.stack((amp[0]*np.sin(tripod), -amp[1]*np.cos(tripod)),axis=-1)
    raise ValueError(condition)


def run_one(seed, condition):
    advance = kernel()
    torques = torque_series(seed, condition)
    world = World(); body = Body(world); body.vertical_support = True
    origin = body.position.copy()
    symmetry = {
        'mirror_fore_aft': condition in ('random_mirror','shuffled_mirror'),
        'friction_long': 2.7 if condition == 'random_isotropic' else
                         4.0 if condition == 'random_friction_swapped' else 1.4,
        'friction_lat': 2.7 if condition == 'random_isotropic' else
                        1.4 if condition == 'random_friction_swapped' else 4.0}
    logs = [advance(body, world, DT, torque, 10, 2, **symmetry) for torque in torques]
    result = summarize(body, origin, logs, DT, len(torques)*BLOCK)
    result['torque_rms'] = np.sqrt(np.mean(torques[-1000:]**2,axis=(0,1))).tolist()
    result['seed'] = seed; result['condition'] = condition
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conditions', nargs='+', default=CONDITIONS)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--budget', type=float, default=260)
    args = parser.parse_args()
    path = RESULTS/'transplant_diagnostic_v1.json'
    data = json.loads(path.read_text()) if path.exists() else {'dt':DT,'seconds':40,'rows':[]}
    done = {(row['seed'],row['condition']) for row in data['rows']}
    begin = time.monotonic()
    for condition in args.conditions:
        for seed in args.seeds:
            if (seed,condition) in done: continue
            if time.monotonic()-begin > args.budget: return
            start = time.monotonic()
            row = run_one(seed,condition)
            row['wall_s'] = time.monotonic()-start
            data['rows'].append(row)
            _save_atomic(path,data)
            print(json.dumps({'seed':seed,'condition':condition,
                              'forward_speed':row['forward_speed'],
                              'contact':row['contact_fraction'],
                              'triple':row['all_three'],
                              'alternation':row['alternation'],
                              'wall_s':row['wall_s']}),flush=True)


if __name__ == '__main__': main()
