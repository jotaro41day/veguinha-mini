"""Short paired phase-lock ladder for the Cython motor circuit.

No learning or target gait enters the neural dynamics. The 4-leg graph is
LF–LM and RF–RM with corresponding contralateral links; the 3-leg graph is
LF–LM–LH. Missing MuJoCo legs remain passive and receive zero torque.
"""

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np

from .config import Config
from .fast_motor import build_probe
from .mujoco_body import MujocoBody
from .search_fast import LOCAL_OSCILLATOR

PAIRS = {
    1: (),
    2: ((0, 1),),
    3: ((0, 1), (1, 2)),
    4: ((0, 1), (2, 3), (0, 2), (1, 3)),
    6: ((0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)),
}
SAME_TRIPOD_PAIRS = ((0, 2), (0, 4), (2, 4), (3, 1), (3, 5), (1, 5))
PHYSICAL = {1: (0,), 2: (0, 1), 3: (0, 1, 2),
            4: (0, 1, 3, 4), 6: (0, 1, 2, 3, 4, 5)}


def _phase_metrics(counts, dt, pairs):
    """Late 20 s edge phase and window-to-window drift; no gait target injected."""
    late = counts[..., -round(20 / dt):]
    traces = late[:, 0, 0] - late[:, 0, 1]
    spectrum = np.fft.fft(traces - traces.mean(axis=1, keepdims=True), axis=1)
    frequencies = np.fft.fftfreq(traces.shape[1], dt)
    band = (frequencies >= 1) & (frequencies <= 3.33)
    peak = np.flatnonzero(band)[np.argmax((abs(spectrum[:, band]) ** 2).sum(axis=0))]
    center = frequencies[peak]
    analytic = np.fft.ifft(spectrum * (2 * ((frequencies >= max(1, center - .8))
                                           & (frequencies <= min(3.5, center + .8)))), axis=1)
    # Remove FFT edge effects before measuring circular dispersion.
    analytic = analytic[:, round(1 / dt):-round(1 / dt)]
    edge_rows = []
    for first, second in pairs:
        product = analytic[first] * analytic[second].conjugate()
        valid = abs(product) >= np.percentile(abs(product), 20)
        halves = np.array_split(product[valid], 2)
        phasor = np.exp(1j * np.angle(product[valid])).mean()
        phase_deg = float(np.angle(phasor, deg=True))
        sd_deg = float(np.rad2deg(np.sqrt(-2 * np.log(max(abs(phasor), 1e-12)))))
        first_phase = np.angle(np.exp(1j * np.angle(halves[0])).mean())
        last_phase = np.angle(np.exp(1j * np.angle(halves[1])).mean())
        drift_deg = float(np.angle(np.exp(1j * (last_phase - first_phase)), deg=True))
        edge_rows.append((phase_deg, sd_deg, drift_deg))
    lock = bool(edge_rows and all(abs(row[0]) >= 135 and row[1] <= 35 for row in edge_rows))
    return (lock, float(np.mean([abs(row[2]) for row in edge_rows])) if edge_rows else 0,
            float(np.mean([row[1] for row in edge_rows])) if edge_rows else 0,
            [round(row[0], 1) for row in edge_rows],
            [round(row[1], 1) for row in edge_rows])


class _PartialDecoder:
    """Same motor filter and baseline as MotorDecoder, for N active modules."""

    def __init__(self, probe, gain=.54):
        self.pools = probe.pools.reshape(len(probe.pools), 4, 8)
        self.previous = np.zeros(len(probe.neurons), dtype=np.int64)
        self.filtered = np.zeros((len(probe.pools), 4))
        self.baseline = np.zeros_like(self.filtered)
        self.config = Config(motor_gain=gain)

    def update(self, total):
        delta = total - self.previous
        self.previous = total.copy()
        hz = delta[self.pools].mean(axis=-1) / .02
        fast = -math.expm1(-20 / self.config.motor_filter_ms)
        slow = -math.expm1(-20 / self.config.motor_baseline_ms)
        self.filtered += fast * (hz - self.filtered)
        self.baseline += slow * (self.filtered - self.baseline)
        deviation = self.filtered - self.baseline
        difference = np.column_stack((deviation[:, 0] - deviation[:, 1],
                                      deviation[:, 2] - deviation[:, 3]))
        activity = np.column_stack((self.filtered[:, 0] + self.filtered[:, 1],
                                    self.filtered[:, 2] + self.filtered[:, 3]))
        return np.clip(self.config.motor_gain * difference * np.clip(activity / 6, 0, 1),
                       -self.config.motor_limit, self.config.motor_limit)


def run(seed, legs, force=18, seconds=40, body_mode="off", noise=2.5,
        reset_gain=2.0, hip_polarity=1, head_contact=True, com_offset_x=0):
    if legs not in PAIRS or body_mode not in (
            "off", "open", "proprio", "contact", "load", "feedback",
            "reset", "reset_no_load"):
        raise ValueError("ladder configuration invalid")
    if hip_polarity not in (-1, 1):
        raise ValueError("hip_polarity must be -1 or 1")
    import pyximport
    pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)
    from .fast_kernel import simulate

    started = time.perf_counter()
    probe = build_probe(legs=legs, seed=seed, backend="cython", coupling=0,
                        mutual_inhibition=0, cross_tau_ms=40)
    order = np.argsort(probe.pre, kind="stable")
    starts = np.r_[0, np.cumsum(np.bincount(probe.pre, minlength=128 * legs))].astype(np.int32)
    targets = probe.post[order].astype(np.int32)
    kinds = probe.edge_kinds[order]
    weights = np.asarray(probe.recurrent.w[:], dtype=float)[order]
    weights[kinds == 5] = 0
    degree = np.zeros(legs)
    for a, b in PAIRS[legs]:
        degree[a] += 1
        degree[b] += 1
    cross = kinds == 7
    weights[cross] = -force / np.sqrt(np.maximum(1, degree[targets[cross] // 128])) / np.sqrt(
        probe.edge_degrees[order][cross])
    free_weights = weights.copy()
    free_weights[cross] = 0
    zeros = np.zeros(2 * legs)
    zero_feet = np.zeros(legs)
    common = dict(return_state=True, edge_kinds=kinds, cross_tau_ms=40,
                  inhibitory_bias=LOCAL_OSCILLATOR["inhibitory_bias"],
                  motor_bias=LOCAL_OSCILLATOR["motor_bias"],
                  adaptation_a=LOCAL_OSCILLATOR["adaptation_a"],
                  adaptation_d=LOCAL_OSCILLATOR["adaptation_d"])
    # Six seconds of independent free oscillation disperse initial phases.
    _, _, _, _, state = simulate(starts, targets, free_weights, legs, 6000, seed,
                                 LOCAL_OSCILLATOR["drive"], 0, noise,
                                 external_angles=zeros, external_contacts=zero_feet,
                                 contact_gain=0, **common)
    body = MujocoBody() if body_mode != "off" else None
    if body is not None and not head_contact:
        body.model.geom('head').contype = 0
        body.model.geom('head').conaffinity = 0
    if body is not None and com_offset_x:
        import mujoco
        body.model.body_ipos[body.body_id, 0] += com_offset_x
        mujoco.mj_forward(body.model, body.data)
    decoder = _PartialDecoder(probe) if body is not None else None
    if decoder is not None:
        decoder.previous = state["total"].copy()
    counts_list, speed_list, contact_list = [], [], []
    toe_z_list, torque_list, knee_angle_list, pitch_list = [], [], [], []
    body_floor_list, body_height_list = [], []
    fall_seen = False
    if body is None:
        counts, _, _, _, state = simulate(starts, targets, weights, legs,
                                         round(seconds * 1000), seed,
                                         LOCAL_OSCILLATOR["drive"], 0, noise,
                                         state=state, external_angles=zeros,
                                         external_contacts=zero_feet, contact_gain=0,
                                         **common)
        dt = .05
    else:
        physical = np.asarray(PHYSICAL[legs], dtype=int)
        leg_weight = body.model.body_mass.sum() * 9.81 / 6
        body_geom_ids = {body.model.geom(name).id for name in ('shell', 'head')}
        for _ in range(round(seconds / .02)):
            use_proprio = body_mode in ("proprio", "feedback", "reset", "reset_no_load")
            use_contact = body_mode in ("contact", "feedback", "reset", "reset_no_load")
            use_load = body_mode in ("load", "feedback", "reset")
            angles = body.angles[physical].ravel() if use_proprio else zeros
            contacts = body.contacts[physical].astype(float) if use_contact else zero_feet
            loads = (np.clip(body.normal_load[physical] / leg_weight, 0, 2)
                     if use_load else zero_feet)
            sample, _, _, _, state = simulate(
                starts, targets, weights, legs, 20, seed,
                LOCAL_OSCILLATOR["drive"], LOCAL_OSCILLATOR["proprio"] if use_proprio else 0,
                noise, state=state, external_angles=angles, external_contacts=contacts,
                external_loads=loads,
                contact_gain=1 if use_contact and body_mode not in ("reset", "reset_no_load") else 0,
                contact_phase_reset_gain=(reset_gain if body_mode in ("reset", "reset_no_load")
                                          else 0),
                stance_load_gain=1 if use_load else 0, **common)
            counts_list.append(sample[..., 0])
            commands = np.zeros((6, 2))
            commands[physical] = decoder.update(state["total"])
            commands[:, 0] *= hip_polarity
            body.step(.02, commands)
            fall_seen |= body.fell
            body_floor_list.append(any(
                (int(body.data.contact[i].geom1) == body.floor_id
                 and int(body.data.contact[i].geom2) in body_geom_ids)
                or (int(body.data.contact[i].geom2) == body.floor_id
                    and int(body.data.contact[i].geom1) in body_geom_ids)
                for i in range(body.data.ncon)))
            body_height_list.append(float(body.data.qpos[2]))
            speed_list.append(float(body.velocity @ [math.cos(body.heading), math.sin(body.heading)]))
            contact_list.append(body.contacts.copy())
            toe_z_list.append(body.data.site_xpos[body.toe_ids, 2].copy())
            torque_list.append(commands.copy())
            knee_angle_list.append(body.angles[:, 1].copy())
            pitch_list.append(math.degrees(math.asin(np.clip(
                body.data.xmat[body.body_id].reshape(3, 3)[2, 0], -1, 1))))
        counts = np.asarray(counts_list).transpose(1, 2, 3, 0)
        dt = .02
    locked, drift, spread, phases, edge_spreads = _phase_metrics(counts, dt, PAIRS[legs])
    if legs == 6:
        _, _, _, same_phases, same_spreads = _phase_metrics(counts, dt, SAME_TRIPOD_PAIRS)
        same_coherent = bool(all(abs(phase) <= 45 and sd <= 35
                                 for phase, sd in zip(same_phases, same_spreads)))
    else:
        same_phases = same_spreads = []
        same_coherent = None
    motor_rate = float(state["total"].reshape(legs, 128)[:, :32].sum() / (32 * legs * (seconds + 6)))
    late_speed = float(np.mean(speed_list[-round(10 / .02):])) if speed_list else None
    late_contacts = np.asarray(contact_list[-round(10 / .02):]) if contact_list else None
    late_toe_z = np.asarray(toe_z_list[-round(10 / .02):]) if toe_z_list else None
    late_torque = np.asarray(torque_list[-round(10 / .02):]) if torque_list else None
    late_knee_angle = np.asarray(knee_angle_list[-round(10 / .02):]) if knee_angle_list else None
    late_pitch = np.asarray(pitch_list[-round(10 / .02):]) if pitch_list else None
    late_body_floor = np.asarray(body_floor_list[-round(10 / .02):]) if body_floor_list else None
    late_body_height = np.asarray(body_height_list[-round(10 / .02):]) if body_height_list else None
    if legs == 6 and late_contacts is not None:
        a = late_contacts[:, [0, 4, 2]].mean(axis=1)
        b = late_contacts[:, [3, 1, 5]].mean(axis=1)
        alt = -float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else 0
        three = float((late_contacts[:, [0, 4, 2]].all(axis=1)
                       | late_contacts[:, [3, 1, 5]].all(axis=1)).mean())
        functional = bool(locked and same_coherent and late_speed > 0 and alt > .3
                          and three > .3)
    else:
        alt = three = None
        functional = None
    return dict(seed=seed, legs=legs, force=force, mode=body_mode, noise=noise,
                reset_gain=reset_gain if body_mode in ("reset", "reset_no_load") else 0,
                hip_polarity=hip_polarity,
                head_contact=head_contact,
                com_offset_x=com_offset_x,
                locked=locked, phase_deg=phases, edge_sd_deg=edge_spreads,
                edge_pass_fraction=(round(sum(abs(phase) >= 135 and sd <= 35
                                              for phase, sd in zip(phases, edge_spreads))
                                          / len(phases), 3) if phases else None),
                phase_sd_deg=round(spread, 2),
                same_phase_deg=same_phases, same_sd_deg=same_spreads,
                same_coherent=same_coherent,
                drift_deg=round(drift, 2), motor_hz=round(motor_rate, 2),
                forward_m_s=round(late_speed, 5) if late_speed is not None else None,
                support=(round(float(late_contacts.mean()), 4)
                         if late_contacts is not None else None),
                support_by_leg=(np.round(late_contacts.mean(axis=0), 3).tolist()
                                if late_contacts is not None else None),
                toe_z_m=(np.round(late_toe_z.mean(axis=0), 3).tolist()
                         if late_toe_z is not None else None),
                torque_rms_by_leg=(np.round(np.sqrt((late_torque ** 2).mean(axis=(0, 2))), 3).tolist()
                                   if late_torque is not None else None),
                knee_mean_by_leg=(np.round(late_knee_angle.mean(axis=0), 3).tolist()
                                  if late_knee_angle is not None else None),
                pitch_deg=(round(float(late_pitch.mean()), 2)
                           if late_pitch is not None else None),
                body_floor_fraction=(round(float(late_body_floor.mean()), 3)
                                     if late_body_floor is not None else None),
                body_height_m=(round(float(late_body_height.mean()), 3)
                               if late_body_height is not None else None),
                fall=fall_seen if body is not None else None,
                alternation=round(alt, 4) if alt is not None else None,
                three_feet=round(three, 4) if three is not None else None,
                functional_tripod=functional,
                wall_s=round(time.perf_counter() - started, 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legs", type=int, nargs="+", default=[2, 3, 4, 6])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    parser.add_argument("--mode", choices=("off", "open", "proprio", "contact",
                                           "load", "feedback", "reset", "reset_no_load"),
                        default="off")
    parser.add_argument("--reset-gain", type=float, default=2.0)
    parser.add_argument("--hip-polarity", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--no-head-contact", action="store_true")
    parser.add_argument("--com-offset-x", type=float, default=0)
    parser.add_argument("--forces", type=float, nargs="+", default=[0, 18])
    parser.add_argument("--seconds", type=float, default=40)
    parser.add_argument("--noise", type=float, default=2.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = None
        for legs in args.legs:
            for seed in args.seeds:
                for force in args.forces:
                    row = run(seed, legs, force, args.seconds, args.mode, args.noise,
                              args.reset_gain, args.hip_polarity,
                              not args.no_head_contact, args.com_offset_x)
                    if writer is None:
                        writer = csv.DictWriter(handle, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    handle.flush()
                    print(row, flush=True)


if __name__ == "__main__":
    main()
