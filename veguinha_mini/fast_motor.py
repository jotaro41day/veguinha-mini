"""Sonda curta de half-centers Izhikevich: 128 células por perna.

As juntas integram torque oriundo apenas de disparos motores. Ângulo e contato
retornam como corrente proprioceptiva; não há sinal periódico nem marcha alvo.
"""

import argparse
from dataclasses import dataclass
import json
import time

import numpy as np
from brian2 import Clock, Network, NeuronGroup, SpikeMonitor, StateMonitor, Synapses
from brian2 import ms, prefs, seed as brian_seed

from .network import EQUATIONS


@dataclass
class Probe:
    network: Network
    neurons: NeuronGroup
    joints: NeuronGroup
    spikes: SpikeMonitor
    states: StateMonitor
    pools: np.ndarray
    premotor: np.ndarray
    inhibitory: np.ndarray
    recurrent: Synapses
    edge_kinds: np.ndarray
    edge_degrees: np.ndarray
    fixed_sign: np.ndarray
    pre: np.ndarray
    post: np.ndarray

    def set_gains(self, recurrence=8.431984, recruitment=9.949371, motor=4.0,
                  cross_inhibition=1.455507, motor_inhibition=2.0,
                  coupling=2.0, proprio=.795704, mutual_inhibition=0.0,
                  drive=5.346912,
                  inhibitory_bias=.371140, motor_bias=0.0):
        gains = np.array([recurrence, recruitment, motor, cross_inhibition,
                          motor_inhibition, coupling, proprio,
                          mutual_inhibition], dtype=float)
        self.recurrent.w = self.fixed_sign * gains[self.edge_kinds] / np.sqrt(self.edge_degrees)
        self.neurons.I_base = self.neurons.I_base[:] * 0 + 5.0
        self.neurons.I_base[self.pools.ravel()] = motor_bias
        self.neurons.I_base[self.premotor.ravel()] = drive
        self.neurons.I_base[self.inhibitory.ravel()] = inhibitory_bias
        self.neurons.I_base[np.arange(len(self.neurons)).reshape(-1, 128)[:, 120:].ravel()] = 1.0


def build_probe(legs=1, seed=42, backend="cython", coupling=2.0,
                drive=5.346912, recurrence=8.431984, recruitment=9.949371,
                cross_inhibition=1.455507, proprio=.795704, noise=2.5,
                motor=4.0, motor_inhibition=2.0,
                inhibitory_bias=.371140, motor_bias=0.0,
                adaptation_a=.005423742, adaptation_d=11.211611,
                pair_kind="neighbor", contact_gain=1.0,
                interleg_contact_gain=0.0,
                interleg_opposite_prob=0.5, mutual_inhibition=0.0,
                cross_tau_ms=150.0):
    if legs not in (1, 2, 3, 4, 6):
        raise ValueError("a sonda aceita 1, 2, 3, 4 ou 6 pernas")
    if pair_kind not in ("neighbor", "contralateral"):
        raise ValueError("tipo de par inválido")
    if not 0 <= interleg_opposite_prob <= 1:
        raise ValueError("probabilidade de acoplamento inválida")
    if cross_tau_ms <= 1:
        raise ValueError("cross_tau_ms deve ser maior que 1 ms")
    prefs.codegen.target = backend
    brian_seed(seed)
    rng = np.random.default_rng(seed)
    n = legs * 128
    # 4 pools motores × 8; 4 half-centers E × 12; 4 half-centers I × 10;
    # 4 pools sensoriais × 2 = 128 células por perna, 68.75% E.
    pools = (np.arange(n).reshape(legs, 128)[:, :32]
             .reshape(legs, 2, 2, 8))
    premotor = (np.arange(n).reshape(legs, 128)[:, 32:80]
                .reshape(legs, 2, 2, 12))
    inhibitory = (np.arange(n).reshape(legs, 128)[:, 80:120]
                  .reshape(legs, 2, 2, 10))
    sensory = (np.arange(n).reshape(legs, 128)[:, 120:]
               .reshape(legs, 2, 2, 2))
    probe_equations = (EQUATIONS.replace(" + I_syn + I_noise", " + I_syn + I_slow + I_cross + I_noise")
                       + "\ndI_slow/dt = -I_slow/(150*ms) : 1\n"
                       + "dI_cross/dt = -I_cross/tau_cross : 1\n"
                       + "tau_cross : second\n")
    neurons = NeuronGroup(n, probe_equations, threshold="v >= 30",
                          reset="v = c; u += d", method="euler",
                          clock=Clock(dt=1 * ms))
    neurons.a = .02
    neurons.b = .2
    neurons.c = -65
    neurons.d = 8
    neurons.a[premotor.ravel()] = adaptation_a
    neurons.d[premotor.ravel()] = adaptation_d
    neurons.a[inhibitory.ravel()] = .1
    neurons.d[inhibitory.ravel()] = 2
    neurons.v = -65 + rng.normal(0, 2, n)
    neurons.u = "b*v"
    neurons.I_base = 5
    neurons.I_base[inhibitory.ravel()] = inhibitory_bias
    neurons.I_base[premotor.ravel()] = drive
    neurons.I_base[sensory.ravel()] = 1
    neurons.I_noise = 0
    neurons.I_slow = 0
    neurons.I_cross = 0
    neurons.tau_cross = cross_tau_ms * ms
    neurons.I_odor = 0
    neurons.I_sense = 0
    neurons.noise = noise
    neurons.noise_gain = 1
    neurons.spike_gate = 1
    neurons.tau_syn = 5 * ms
    neurons.run_regularly("I_noise = noise*noise_gain*randn()")
    neurons.run_regularly("v = clip(v, -90, 100)", when="after_groups")

    joints = NeuronGroup(legs * 2, """
        dtheta/dt = omega/second : 1
        domega/dt = (4*clip(drive, -1, 1) - 2*theta - 1.8*omega)/second : 1
        ddrive/dt = -drive/(120*ms) : 1
        contact = int(theta < 0) : 1
    """, method="euler", clock=neurons.clock)
    joints.theta = rng.normal(0, .03, legs * 2)
    joints.omega = 0
    joints.drive = 0

    pre, post, kinds, signs = [], [], [], []

    def connect(src, dst, kind, sign=1, no_self=False):
        for i in src:
            for j in dst:
                if no_self and i == j:
                    continue
                pre.append(int(i)); post.append(int(j))
                kinds.append(kind); signs.append(sign)

    for leg in range(legs):
        for joint in range(2):
            for action in range(2):
                e = premotor[leg, joint, action]
                i = inhibitory[leg, joint, action]
                other = 1 - action
                connect(e, e, 0, no_self=True)
                connect(e, i, 1)
                connect(e, pools[leg, joint, action], 2)
                connect(i, premotor[leg, joint, other], 3, sign=-1)
                connect(i, pools[leg, joint, other], 4, sign=-1)
                connect(sensory[leg, joint, action], e, 6)
    if legs > 1:
        pairs = {2: [(0, 1)],
                 3: [(0, 1), (1, 2)],
                 4: [(0, 1), (2, 3), (0, 2), (1, 3)],
                 6: [(0, 1), (1, 2), (3, 4), (4, 5),
                     (0, 3), (1, 4), (2, 5)]}[legs]
        # Pares contralaterais têm uma amostra própria da mesma regra esparsa.
        # Nenhum dos dois tipos recebe uma fase ou sinal de marcha prescrito.
        coupling_rng = (np.random.default_rng(seed + 10_000)
                        if legs == 2 and pair_kind == "contralateral" else rng)
        for first, second in pairs:
            for a, b in ((first, second), (second, first)):
                for joint in range(2):
                    for action in range(2):
                        for source in premotor[a, joint, action]:
                            # Small, noisy contralateral/neighbor phase bias can
                            # support coordination without prescribing a gait.
                            target_action = (1 - action if
                                             coupling_rng.random() < interleg_opposite_prob
                                             else action)
                            targets = premotor[b, joint, target_action]
                            connect((source,), coupling_rng.choice(targets, size=3,
                                                                   replace=False), 5)
        for first, second in pairs:
            for a, b in ((first, second), (second, first)):
                for joint in range(2):
                    for action in range(2):
                        for source in inhibitory[a, joint, action]:
                            targets = premotor[b, joint, action]
                            connect((source,), coupling_rng.choice(targets, size=3,
                                                                   replace=False), 7, sign=-1)

    pre = np.asarray(pre, dtype=np.int32)
    post = np.asarray(post, dtype=np.int32)
    kinds = np.asarray(kinds, dtype=np.int8)
    signs = np.asarray(signs, dtype=float)
    degree = np.maximum(1, np.bincount(post, minlength=n))[post]
    recurrent = Synapses(neurons, neurons, model="w : 1\nkind : integer (constant)",
                         on_pre="I_syn_post += w*int(kind != 3 and kind != 7); "
                                "I_slow_post += w*int(kind == 3); "
                                "I_cross_post += w*int(kind == 7)",
                         clock=neurons.clock)
    recurrent.connect(i=pre, j=post)
    recurrent.kind = kinds
    recurrent.delay = 2 * ms

    motor_syn = Synapses(neurons, joints, model="w : 1",
                         on_pre="drive_post += w", clock=neurons.clock)
    m_pre, m_post, m_weight = [], [], []
    for leg in range(legs):
        for joint in range(2):
            for action, sign in ((0, 1), (1, -1)):
                for cell in pools[leg, joint, action]:
                    m_pre.append(int(cell)); m_post.append(leg * 2 + joint)
                    m_weight.append(.028 * sign)
    motor_syn.connect(i=m_pre, j=m_post)
    motor_syn.w = m_weight

    feedback = Synapses(joints, neurons,
                        model="I_sense_post = clip(-gain*theta_pre, 0, 1e9)*int(kind == 0)"
                              " + gain*contact_pre*int(kind == 1) : 1 (summed)\n"
                              "gain : 1\nkind : integer (constant)", clock=neurons.clock)
    f_pre, f_post, f_gain, f_kind = [], [], [], []
    for leg in range(legs):
        for joint in range(2):
            for action, sign in ((0, 1), (1, -1)):
                for cell in sensory[leg, joint, action]:
                    f_pre.append(leg * 2 + joint); f_post.append(int(cell))
                    f_gain.append(sign * proprio)
                    f_kind.append(0)
        # Same sensory encoding as the full model: stretch excites
        # the restoring antagonist; knee contact excites local inhibitory cells.
        for cell in inhibitory[leg].ravel():
            f_pre.append(leg * 2 + 1); f_post.append(int(cell))
            f_gain.append(contact_gain); f_kind.append(1)
    if legs > 1 and interleg_contact_gain:
        contact_pairs = {2: [(0, 1)],
                         3: [(0, 1), (1, 2)],
                         4: [(0, 1), (2, 3), (0, 2), (1, 3)],
                         6: [(0, 1), (1, 2), (3, 4), (4, 5),
                             (0, 3), (1, 4), (2, 5)]}[legs]
        neighbors = [set() for _ in range(legs)]
        for first, second in contact_pairs:
            neighbors[first].add(second)
            neighbors[second].add(first)
        for target_leg, sources in enumerate(neighbors):
            for source_leg in sorted(sources):
                # A planted source foot recruits the target flexor-side I pool.
                # This is a load-sharing reflex; no target trajectory is set.
                for cell in inhibitory[target_leg, :, 0].ravel():
                    f_pre.append(source_leg * 2 + 1); f_post.append(int(cell))
                    f_gain.append(contact_gain * interleg_contact_gain / len(sources))
                    f_kind.append(1)
    feedback.connect(i=f_pre, j=f_post)
    feedback.gain = f_gain
    feedback.kind = f_kind

    spikes = SpikeMonitor(neurons)
    states = StateMonitor(joints, ("theta", "omega", "drive", "contact"),
                          record=True, dt=20 * ms)
    network = Network(neurons, joints, recurrent, motor_syn, feedback, spikes, states)
    probe = Probe(network, neurons, joints, spikes, states, pools, premotor,
                  inhibitory, recurrent, kinds, degree, signs, pre, post)
    probe.set_gains(recurrence=recurrence, recruitment=recruitment,
                    cross_inhibition=cross_inhibition, coupling=coupling,
                    proprio=proprio, drive=drive, motor=motor,
                    motor_inhibition=motor_inhibition,
                    mutual_inhibition=mutual_inhibition,
                    inhibitory_bias=inhibitory_bias, motor_bias=motor_bias)
    return probe


def _correlation(a, b):
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _leg_phase_lags(counts, pairs):
    """Defasagem espectral em graus, medida separadamente nas duas metades."""
    values = []
    samples = counts.shape[-1]
    for first, second in pairs:
        pair_lags = []
        for half in (0, 1):
            lo, hi = half * samples // 2, (half + 1) * samples // 2
            traces = np.asarray([
                counts[first, 0, 0, lo:hi] - counts[first, 0, 1, lo:hi],
                counts[second, 0, 0, lo:hi] - counts[second, 0, 1, lo:hi],
            ], dtype=float)
            traces -= traces.mean(axis=1, keepdims=True)
            spectrum = np.fft.rfft(traces * np.hanning(traces.shape[-1]), axis=1)
            frequency = np.fft.rfftfreq(traces.shape[-1], .05)
            band = (frequency >= 1) & (frequency <= 3.33)
            if not band.any():
                pair_lags.append(None)
                continue
            index = np.flatnonzero(band)[np.argmax((np.abs(spectrum[:, band]) ** 2).sum(axis=0))]
            lag = np.angle(spectrum[0, index] * spectrum[1, index].conjugate(), deg=True)
            pair_lags.append(round(float(lag), 1))
        values.append(pair_lags)
    return values


def measure_probe(legs=1, seconds=20.0, seed=42, backend="cython", **parameters):
    if backend == "kernel":
        return measure_kernel(legs=legs, seconds=seconds, seed=seed, **parameters)
    probe = build_probe(legs=legs, seed=seed, backend=backend, **parameters)
    built = time.perf_counter()
    probe.network.run(seconds * 1000 * ms)
    wall = time.perf_counter() - built
    times = np.asarray(probe.spikes.t / ms)
    indices = np.asarray(probe.spikes.i, dtype=np.int32)
    bins = np.arange(0, seconds * 1000 + 50, 50)
    counts = np.zeros((legs, 2, 2, len(bins) - 1), dtype=float)
    for leg in range(legs):
        for joint in range(2):
            for action in range(2):
                mask = np.isin(indices, probe.premotor[leg, joint, action])
                counts[leg, joint, action] = np.histogram(times[mask], bins)[0]
    correlations = [[_correlation(counts[leg, joint, 0], counts[leg, joint, 1])
                     for joint in range(2)] for leg in range(legs)]
    periods = []
    for leg in range(legs):
        joint_periods = []
        for joint in range(2):
            signal = counts[leg, joint, 0] - counts[leg, joint, 1]
            signal = signal - signal.mean()
            autocorr = np.correlate(signal, signal, mode="full")[len(signal)-1:]
            if autocorr[0] <= 0 or len(autocorr) <= 20:
                joint_periods.append(None)
            else:
                lo, hi = 6, min(21, len(autocorr))
                peak = lo + int(np.argmax(autocorr[lo:hi]))
                joint_periods.append(round(peak * .05, 3)
                                     if autocorr[peak] > .1 * autocorr[0] else None)
        periods.append(joint_periods)
    rates = counts.sum(axis=-1) / (12 * seconds)
    pairs = ([(0, 1)] if legs == 2 else
             [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)] if legs == 6 else [])
    phase = []
    for a, b in pairs:
        phase.append(_correlation(counts[a, 0, 0] - counts[a, 0, 1],
                                  counts[b, 0, 0] - counts[b, 0, 1]))
    return {"legs": legs, "pair_kind": parameters.get("pair_kind", "neighbor"),
            "neurons": len(probe.neurons),
            "synapses": len(probe.recurrent), "simulated_seconds": seconds,
            "wall_seconds": round(wall, 3),
            "realtime_ratio": round(seconds / wall, 2),
            "antagonist_correlation": np.round(correlations, 3).tolist(),
            "period_s": periods, "premotor_rates_hz": np.round(rates, 2).tolist(),
            "interleg_phase": np.round(phase, 3).tolist(),
            "interleg_lag_deg": _leg_phase_lags(counts, pairs),
            "joint_angle_sd": np.round(np.std(probe.states.theta[:], axis=1), 4).tolist(),
            "contact_fraction": np.round(np.mean(probe.states.contact[:], axis=1), 3).tolist()}


def measure_kernel(legs=1, seconds=20.0, seed=42, **parameters):
    """Mesma rede/integração Euler em núcleo Cython, para triagem curta."""
    import pyximport
    pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)
    from .fast_kernel import simulate

    probe = build_probe(legs=legs, seed=seed, backend="cython", **parameters)
    order = np.argsort(probe.pre, kind="stable")
    starts = np.concatenate(([0], np.cumsum(np.bincount(probe.pre,
                                                        minlength=128 * legs))))
    gains = np.array([parameters.get("recurrence", 8.431984),
                      parameters.get("recruitment", 9.949371), parameters.get("motor", 4.0),
                      parameters.get("cross_inhibition", 1.455507),
                      parameters.get("motor_inhibition", 2.0),
                      parameters.get("coupling", 2.0),
                      parameters.get("proprio", .795704),
                      parameters.get("mutual_inhibition", 0.0)])
    weights = probe.fixed_sign[order] * gains[probe.edge_kinds[order]] / np.sqrt(probe.edge_degrees[order])
    started = time.perf_counter()
    counts, angles, contacts, _ = simulate(starts.astype(np.int32),
                                         probe.post[order].astype(np.int32),
                                         weights.astype(np.float64), legs,
                                         round(seconds * 1000), seed,
                                         parameters.get("drive", 5.346912),
                                         parameters.get("proprio", .795704),
                                         parameters.get("noise", 2.5),
                                         inhibitory_bias=parameters.get("inhibitory_bias", .371140),
                                         motor_bias=parameters.get("motor_bias", 0.0),
                                         adaptation_a=parameters.get("adaptation_a", .005423742),
                                         adaptation_d=parameters.get("adaptation_d", 11.211611),
                                         contact_gain=parameters.get("contact_gain", 1.0),
                                         interleg_contact_gain=parameters.get("interleg_contact_gain", 0.0),
                                         edge_kinds=probe.edge_kinds[order])
    wall = time.perf_counter() - started
    correlations = [[_correlation(counts[leg, joint, 0], counts[leg, joint, 1])
                     for joint in range(2)] for leg in range(legs)]
    periods = []
    for leg in range(legs):
        joint_periods = []
        for joint in range(2):
            signal = counts[leg, joint, 0].astype(float) - counts[leg, joint, 1]
            signal -= signal.mean()
            autocorr = np.correlate(signal, signal, mode="full")[len(signal)-1:]
            if autocorr[0] <= 0 or len(autocorr) <= 20:
                joint_periods.append(None)
            else:
                lo, hi = 6, min(21, len(autocorr))
                peak = lo + int(np.argmax(autocorr[lo:hi]))
                joint_periods.append(round(peak * .05, 3)
                                     if autocorr[peak] > .1 * autocorr[0] else None)
        periods.append(joint_periods)
    rates = counts.sum(axis=-1) / (12 * seconds)
    pairs = ([(0, 1)] if legs == 2 else
             [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)] if legs == 6 else [])
    phase = []
    for a, b in pairs:
        phase.append(_correlation(counts[a, 0, 0] - counts[a, 0, 1],
                                  counts[b, 0, 0] - counts[b, 0, 1]))
    return {"legs": legs, "pair_kind": parameters.get("pair_kind", "neighbor"),
            "neurons": 128 * legs, "synapses": len(weights),
            "simulated_seconds": seconds, "wall_seconds": round(wall, 3),
            "realtime_ratio": round(seconds / wall, 2),
            "antagonist_correlation": np.round(correlations, 3).tolist(),
            "period_s": periods, "premotor_rates_hz": np.round(rates, 2).tolist(),
            "interleg_phase": np.round(phase, 3).tolist(),
            "interleg_lag_deg": _leg_phase_lags(counts, pairs),
            "joint_angle_sd": np.round(np.std(angles, axis=1), 4).tolist(),
            "contact_fraction": np.round(np.mean(contacts, axis=1), 3).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legs", type=int, choices=(1, 2, 6), default=1)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=("numpy", "cython", "kernel"), default="kernel")
    parser.add_argument("--pair-kind", choices=("neighbor", "contralateral"), default="neighbor")
    parser.add_argument("--drive", type=float, default=5.346912)
    parser.add_argument("--recurrence", type=float, default=8.431984)
    parser.add_argument("--recruitment", type=float, default=9.949371)
    parser.add_argument("--cross-inhibition", type=float, default=1.455507)
    parser.add_argument("--coupling", type=float, default=2.0)
    parser.add_argument("--proprio", type=float, default=.795704)
    parser.add_argument("--contact-gain", type=float, default=1.0)
    parser.add_argument("--interleg-contact-gain", type=float, default=0.0)
    args = parser.parse_args()
    print(json.dumps(measure_probe(legs=args.legs, seconds=args.seconds,
                                   seed=args.seed, backend=args.backend,
                                   drive=args.drive, recurrence=args.recurrence,
                                   recruitment=args.recruitment,
                                   cross_inhibition=args.cross_inhibition,
                                   coupling=args.coupling, proprio=args.proprio,
                                   contact_gain=args.contact_gain,
                                   interleg_contact_gain=args.interleg_contact_gain,
                                   pair_kind=args.pair_kind),
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
