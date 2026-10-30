"""Intrinsic three-joint LF diagnostic, without body, feedback or plasticity."""
from dataclasses import replace
import time
import numpy as np
from brian2 import start_scope, NetworkOperation, StateMonitor, Clock, ms
from veguinha_mini.tools.mk1_neuralcheck import trial_config
from veguinha_mini.network import build_network
from veguinha_mini.mk1_validation import atomic
from veguinha_mini.tools.mk1_walk_audit import OUT


def run(seed, adaptation_d=None, dt_ms=1):
    start_scope()
    c = trial_config(seed, 'coupled', isolated=True)
    regions = tuple(r for r in c.regions if not r.name.startswith('vnc_') or r.name == 'vnc_lf')
    names = {r.name for r in regions}
    c = replace(c, regions=regions, phase5_enabled=False, dt_ms=dt_ms,
                pathways=tuple(p for p in c.pathways if p.source in names and p.target in names))
    if adaptation_d is not None:
        c = replace(c, half_center_adaptation_d=adaptation_d)
    model = build_network(c, record_spikes=False)
    assert set(model.motor_pools) == {'lf'} and not len(model.plastic_edges)
    keys = [f'{joint}_{action}' for joint in c.joint_names for action in ('flex', 'extend')]
    pools = ([model.motor_pools['lf'][k] for k in keys]
             + [model.half_centers['lf'][k+'_e'] for k in keys]
             + [model.half_centers['lf'][k+'_i'] for k in keys])
    voltage = StateMonitor(model.group, 'v', record=np.concatenate(pools[6:]),
                           when='after_groups', clock=model.group.clock)
    model.network.add(voltage)
    previous = np.zeros(len(model.group), dtype=np.int64)
    series = []

    def record():
        nonlocal previous
        count = np.asarray(model.monitor.count[:], dtype=np.int64)
        delta = count-previous
        previous = count.copy()
        series.append([delta[p].sum()/len(p)/.02 for p in pools])

    model.network.add(NetworkOperation(record, clock=Clock(dt=20*ms), when='end'))
    model.network.run(0*ms)
    start = time.perf_counter()
    model.network.run(20000*ms)
    wall = time.perf_counter()-start
    rates = np.asarray(series)
    smooth = np.array([np.convolve(x, np.ones(6)/6, mode='valid') for x in rates.T]).T[-500:]
    parts = {}
    for label, index in (('motor', 0), ('premotor', 6), ('inhibitory', 12)):
        data = smooth[:, index:index+6].reshape(-1, 3, 2)
        diff = data[:, :, 0]-data[:, :, 1]
        freq = np.fft.rfftfreq(len(diff), .02)
        power = abs(np.fft.rfft(diff-diff.mean(axis=0), axis=0))**2
        band = (freq >= 1) & (freq <= 3.33)
        peak = np.argmax(power[band], axis=0)
        concentration = (np.take_along_axis(power[band], peak[None], axis=0)[0]
                         / np.maximum(power[(freq >= .5) & (freq <= 8)].sum(axis=0), 1e-20))
        correlation = [float(np.corrcoef(data[:, j, 0], data[:, j, 1])[0, 1])
                       if min(data[:, j, 0].std(), data[:, j, 1].std()) > 1e-10 else 0.
                       for j in range(3)]
        parts[label] = dict(mean_hz=data.mean(axis=0).tolist(),
                            antagonist_correlation=correlation,
                            spectral_fraction=concentration.tolist(),
                            candidate_period_s=(1/freq[band][peak]).tolist(),
                            rhythmic=((concentration > .3) & (np.array(correlation) < -.3)).tolist())
    row = dict(seed=seed, cells=len(model.group), seconds=20,
               adaptation_d=c.half_center_adaptation_d, dt_ms=dt_ms,
               voltage_min=float(np.min(voltage.v)), voltage_max=float(np.max(voltage.v)),
               device='controller 25078PC3EG', wall_s=wall, parts=parts,
               conditions='one LF module + tonic descending drive; no physical sensory input, other VNC, plasticity or Simulation homeostasis')
    name = f'single_leg_{seed}'
    if adaptation_d is not None:
        name += f'_d{adaptation_d:g}'
    if dt_ms != 1:
        name += f'_dt{dt_ms:g}'
    atomic(OUT/f'{name}.json', row)
    np.savez_compressed(OUT/f'{name}.npz', rates=rates)
    print(seed, 'motor rhythmic', parts['motor']['rhythmic'],
          'premotor rhythmic', parts['premotor']['rhythmic'], 'wall_s', wall, flush=True)
    return row


if __name__ == '__main__':
    for seed in (42, 43, 44):
        run(seed)
