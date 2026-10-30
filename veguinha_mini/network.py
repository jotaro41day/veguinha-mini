"""Rede Brian2 isolada e diagnóstico de atividade por região."""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
from brian2 import Clock, EventMonitor, Network, NeuronGroup, Synapses, ms, prefs, seed

from .config import Config
from .neurons import make_parameters
from .regions import MOTOR_LEARNING_PATHWAYS, MOTOR_LEARNING_REGIONS


EQUATIONS = """
dv/dt = (0.04*v*v + 5*v + 140 - u + I_base + I_syn + I_noise + I_odor + I_sense)/ms : 1
du/dt = a*(b*v-u)/ms : 1
dI_syn/dt = -I_syn/tau_syn : 1
I_base : 1
I_noise : 1
I_odor : 1
I_sense : 1
spike_gate : 1
noise : 1
noise_gain : 1
a : 1
b : 1
c : 1
d : 1
tau_syn : second (shared)
"""


class EmittedSpikeMonitor(EventMonitor):
    """Monitor dos eventos transmitidos, compatível com o contador antigo."""

    @property
    def num_spikes(self):
        return self.num_events


@dataclass
class IsolatedNetwork:
    network: Network
    group: NeuronGroup
    synapses: Synapses
    monitor: EventMonitor
    regions: dict[str, np.ndarray]
    inhibitory: np.ndarray
    cell_types: np.ndarray
    motor_pools: dict[str, dict[str, np.ndarray]]
    interneurons: dict[str, dict[str, np.ndarray]]
    half_centers: dict[str, dict[str, np.ndarray]]
    subgroups: dict[str, object]
    plastic_edges: np.ndarray | None = None
    ms: object = ms
    cord: object | None = None


@dataclass
class Diagnostic:
    rates_hz: dict[str, float]
    late_rates_hz: dict[str, float]
    counts: dict[str, int]
    warnings: list[str]
    cells: int
    synapses: int
    inhibitory_fraction: float
    kc_active_fraction_50ms: float


def build_network(config: Config = Config(), record_spikes: bool = True) -> IsolatedNetwork:
    modular=config.motor_circuit=='modular'
    prefs.codegen.target = config.backend
    seed(config.seed)
    rng = np.random.default_rng(config.seed)
    regions = {}
    inhibitory_parts = []
    biases = []
    motor_pools = {}
    interneurons = {}
    half_centers = {}
    offset = 0
    learning_regions=tuple(r for r in MOTOR_LEARNING_REGIONS if not modular or r.name!='proprio')
    region_specs = config.regions + (learning_regions if config.phase5_enabled else ())
    if config.body_model == "mk1":
        region_specs = tuple(replace(r, cells=196 if modular else 144, inhibitory=60 if modular else 46) if r.name.startswith("vnc_")
                             else replace(r, cells=144) if r.name == "proprio" else r
                             for r in region_specs)
    joints = config.joint_names
    pool_names = tuple(f"{j}_{a}" for j in joints for a in ("flex", "extend"))
    n_pools = len(pool_names)
    pathway_specs = config.pathways + (MOTOR_LEARNING_PATHWAYS if config.phase5_enabled else ())
    for item in region_specs:
        count = item.cells if item.name in ("proprio", "flow", "oct") else max(1, round(item.cells * config.scale))
        n_inhibitory = item.inhibitory if item.name in ("proprio", "flow", "oct") else round(item.inhibitory * config.scale)
        indices = np.arange(offset, offset + count)
        regions[item.name] = indices
        local_inhibitory = np.zeros(count, dtype=bool)
        if item.name == "flow":
            local_inhibitory.reshape(3, 2, 8)[:, 1, :] = True
        elif item.name.startswith("vnc_"):
            leg = item.name[4:]
            per_pool = max(1, round(8 * config.scale)) if config.body_model == "mk1" else max(1, round(count / 16))
            motor_pools[leg] = {
                name: indices[p * per_pool:(p + 1) * per_pool]
                for p, name in enumerate(pool_names)
            }
            exc_count = max(n_pools * per_pool, count - n_inhibitory)
            local_inhibitory[exc_count:] = True
            if modular:
                local_inhibitory[:]=False;local_inhibitory[120:180]=True
            interneurons[leg] = {"excitatory": indices[n_pools * per_pool:exc_count],
                                  "inhibitory": indices[exc_count:]}
            if config.motor_scaffold:
                exc_pools = np.array_split(interneurons[leg]["excitatory"], n_pools)
                inh_pools = np.array_split(interneurons[leg]["inhibitory"], n_pools)
                labels = pool_names
                if any(len(pool) == 0 for pool in (*exc_pools, *inh_pools)):
                    raise ValueError("andaime requer ao menos quatro interneurônios E e I por perna")
                half_centers[leg] = {f"{label}_{kind}": pool
                                     for kind, pools in (("e", exc_pools), ("i", inh_pools))
                                     for label, pool in zip(labels, pools)}
        else:
            local_inhibitory[:n_inhibitory] = True
            rng.shuffle(local_inhibitory)
        inhibitory_parts.append(local_inhibitory)
        biases.append(np.full(count, item.bias))
        offset += count
    inhibitory = np.concatenate(inhibitory_parts)
    cell_types, parameters = make_parameters(inhibitory, rng)
    if config.motor_scaffold:
        for centers in half_centers.values():
            exc = np.concatenate([pool for name, pool in centers.items() if name.endswith("_e")])
            parameters[exc, 0] = config.half_center_adaptation_a * rng.uniform(.9, 1.1, len(exc))
            parameters[exc, 3] = config.half_center_adaptation_d * rng.uniform(.9, 1.1, len(exc))
    clock = Clock(dt=config.dt_ms * ms, name="veguinha_neural_clock")
    equations=EQUATIONS
    if modular:
        equations=equations.replace(' + I_syn + I_noise',' + I_syn + I_slow + I_cross + I_noise')
        equations+='\ndI_slow/dt=-I_slow/(150*ms):1\ndI_cross/dt=-I_cross/(40*ms):1\nv_threshold:1\n'
    if config.motor_cross_tau_ms is not None:
        equations=equations.replace(' + I_syn + I_noise',' + I_syn + I_motor_slow + I_noise')
        equations+=f'\ndI_motor_slow/dt = -I_motor_slow/({config.motor_cross_tau_ms}*ms) : 1\n'
    group = NeuronGroup(
        offset, equations, threshold="v >= v_threshold" if modular else "v >= 30", reset="v = c; u += d",
        events={"emit": "v >= v_threshold and spike_gate > 0" if modular else "v >= 30 and spike_gate > 0"}, method="euler", clock=clock,
        name="veguinha_cells",
    )
    group.a = parameters[:, 0]
    group.b = parameters[:, 1]
    group.c = parameters[:, 2]
    group.d = parameters[:, 3]
    group.v = -65 + rng.normal(0, 2, offset)
    group.u = "b*v"
    group.I_base = np.concatenate(biases) + rng.normal(0, 0.3, offset)
    if config.motor_pool_bias is not None:
        for pools in motor_pools.values():
            group.I_base[np.concatenate(tuple(pools.values()))] = config.motor_pool_bias
    if config.motor_scaffold:
        for centers in half_centers.values():
            for name, pool in centers.items():
                if name.endswith("_i"):
                    group.I_base[pool] = config.half_center_inhibitory_bias
                elif config.half_center_exc_bias is not None:
                    group.I_base[pool] = config.half_center_exc_bias
    group.noise = config.noise
    group.noise_gain = 1
    group.noise_gain[regions["kc"]] = 0.15
    group.spike_gate = 1
    group.tau_syn = config.synaptic_tau_ms * ms
    if modular:group.v_threshold=30
    if modular:group.run_regularly("I_noise = noise*noise_gain*randn()",clock=Clock(dt=1*ms,name='veguinha_cells_run_regularly_clock'))
    else:group.run_regularly("I_noise = noise*noise_gain*randn()")
    cord=None;local_template=None
    if modular:
        from .cord import build_cord
        local_template=build_cord(config.seed,config.motor_cross_strength,backend='numpy')
        prefs.codegen.target=config.backend;seed(config.seed)
        mapping=np.concatenate([regions['vnc_'+l] for l in ('lf','lm','lh','rf','rm','rh')])
        for field in ('a','b','c','d','v','u','I_base','I_syn','I_slow','I_cross','I_noise','I_sense','I_odor','noise_gain','spike_gate','v_threshold'):
            getattr(group,field)[mapping]=getattr(local_template.group,field)[:]
        group.noise_gain[mapping[local_template.motors.ravel()]]=0
        cord=SimpleNamespace(group=group,motors=mapping[local_template.motors],premotor=mapping[local_template.premotor],inhibitory=mapping[local_template.inhibitory],positions=mapping[local_template.positions],loads=mapping[local_template.loads])
        cell_types[mapping]='RS';cell_types[cord.inhibitory.ravel()]='FS'
        for k,l in enumerate(('lf','lm','lh','rf','rm','rh')):
            interneurons[l]={'excitatory':cord.premotor[k].ravel(),'inhibitory':cord.inhibitory[k].ravel()}
            half_centers[l]={j+'_'+a+'_'+t:ids[k,n,s] for n,j in enumerate(joints) for s,a in enumerate(('flex','extend')) for t,ids in (('e',cord.premotor),('i',cord.inhibitory))}
    subgroups = {name: group[indices[0]:indices[-1] + 1]
                 for name, indices in regions.items()}

    pre_parts = []
    post_parts = []
    strength_parts = []
    plastic_parts = []
    slow_parts = []
    for path in pathway_specs:
        if modular and path.source.startswith('vnc_') and path.target.startswith('vnc_'):
            continue
        if (config.motor_scaffold and config.scaffold_sparse and path.source.startswith("vnc_")
                and path.target.startswith("vnc_")):
            continue
        sources = regions[path.source]
        targets = regions[path.target]
        if modular and path.target.startswith('vnc_'):
            targets=targets[48:180]
        degree = min(path.in_degree, len(sources) - (path.source == path.target))
        if not degree:
            continue
        selected = np.empty((len(targets), degree), dtype=np.int32)
        for row, target in enumerate(targets):
            candidates = sources[sources != target] if path.source == path.target else sources
            selected[row] = rng.choice(candidates, size=degree, replace=False)
        pre_parts.append(selected.ravel())
        post_parts.append(np.repeat(targets, degree).astype(np.int32))
        strength_parts.append(np.full(selected.size, path.weight, dtype=np.float32))
        motor_target = path.target.startswith("vnc_")
        plastic_parts.append(np.full(selected.size, config.phase5_enabled and motor_target
                                     and (path.source == "dn" or path.source.startswith("vnc_")), dtype=bool))
        slow_parts.append(np.zeros(selected.size,dtype=bool))

    def add_projection(sources, targets, strength, plastic=False, slow=False):
        if not len(sources) or not len(targets):
            return
        pre_parts.append(np.repeat(sources, len(targets)).astype(np.int32))
        post_parts.append(np.tile(targets, len(sources)).astype(np.int32))
        size = len(sources) * len(targets)
        strength_parts.append(np.full(size, strength, dtype=np.float32))
        plastic_parts.append(np.full(size, plastic, dtype=bool))
        slow_parts.append(np.full(size,slow,dtype=bool))

    recurrent_legs = {path.source[4:] for path in pathway_specs
                      if path.source.startswith("vnc_") and path.source == path.target}
    if modular:recurrent_legs=set()
    for leg in sorted(recurrent_legs):
        motors = motor_pools[leg]
        if config.motor_scaffold:
            centers = half_centers[leg]
            for joint in joints:
                for action, opposite in (("flex", "extend"), ("extend", "flex")):
                    premotor_e = centers[f"{joint}_{action}_e"]
                    premotor_i = centers[f"{joint}_{action}_i"]
                    add_projection(premotor_e, premotor_e, config.half_center_recurrence)
                    add_projection(premotor_e, premotor_i, config.half_center_recruitment)
                    add_projection(premotor_e, motors[f"{joint}_{action}"],
                                   config.half_center_motor)
                    add_projection(premotor_i, centers[f"{joint}_{opposite}_e"],
                                   config.half_center_cross_inhibition,slow=True)
                    add_projection(premotor_i, motors[f"{joint}_{opposite}"],
                                   config.half_center_motor_inhibition)
            dn_exc = regions["dn"][~inhibitory[regions["dn"]]]
            add_projection(dn_exc,
                           np.concatenate([pool for name, pool in centers.items()
                                           if name.endswith("_e")]),
                           config.half_center_dn_drive)
        else:
            pools = motor_pools[leg]
            inhibitory_pools = np.array_split(interneurons[leg]["inhibitory"], n_pools)
            for joint_i, joint in enumerate(joints):
                flex, extend = pools[f"{joint}_flex"], pools[f"{joint}_extend"]
                flex_i, extend_i = inhibitory_pools[2 * joint_i:2 * joint_i + 2]
                add_projection(flex, flex_i, 0.9)
                add_projection(flex_i, extend, 1.5)
                add_projection(extend, extend_i, 0.9)
                add_projection(extend_i, flex, 1.5)

    if config.motor_scaffold:
        pairs = sorted({tuple(sorted((path.source[4:], path.target[4:])))
                        for path in pathway_specs
                        if path.source.startswith("vnc_") and path.target.startswith("vnc_")
                        and path.source != path.target})
        for first, second in pairs:
            for source_leg, target_leg in ((first, second), (second, first)):
                for joint in joints:
                    for action in ("flex", "extend"):
                        target_action = rng.choice(("flex", "extend"))
                        targets = half_centers[target_leg][f"{joint}_{target_action}_e"]
                        sources = half_centers[source_leg][f"{joint}_{action}_e"]
                        if config.scaffold_sparse:
                            for source in sources:
                                add_projection(np.array([source]),
                                               rng.choice(targets, size=min(3, len(targets)),
                                                          replace=False),
                                               config.half_center_interleg_strength, plastic=True)
                        else:
                            add_projection(sources, targets,
                                           config.half_center_interleg_strength, plastic=True)

    if config.phase5_enabled and not modular:
        channels = np.asarray(regions["proprio"]).reshape(6, len(joints), 2, -1)
        for leg_i, leg in enumerate(("lf", "lm", "lh", "rf", "rm", "rh")):
            for joint_i, joint in enumerate(joints):
                for sign_i, action in enumerate(("flex", "extend")):
                    sources = channels[leg_i, joint_i, sign_i]
                    targets = np.concatenate((motor_pools[leg][f"{joint}_{action}"],
                                              interneurons[leg]["excitatory"]))
                    add_projection(sources, targets, config.proprio_projection_strength, plastic=True)

    synapse_equations='w : 1';on_pre='I_syn_post += w'
    if modular:
        synapse_equations+='\nslow_flag:integer (constant)\ncross_flag:integer (constant)'
        on_pre='I_syn_post+=w*(1-slow_flag)*(1-cross_flag); I_slow_post+=w*slow_flag; I_cross_post+=w*cross_flag'
    if config.motor_cross_tau_ms is not None:
        synapse_equations+='\nslow_flag : integer (constant)'
        on_pre='I_syn_post += w*(1-slow_flag); I_motor_slow_post += w*slow_flag'
    synapses = Synapses(group, group, model=synapse_equations, on_pre=on_pre,
                        on_event="emit", clock=clock, name="veguinha_synapses")
    pre = np.concatenate(pre_parts)
    post = np.concatenate(post_parts)
    in_degree = np.bincount(post, minlength=offset)
    if modular:
        in_degree+=np.bincount(mapping[np.asarray(local_template.synapses.j[:],int)],minlength=offset)
    strength = np.concatenate(strength_parts)
    weights = strength * rng.uniform(0.9, 1.1, len(pre)) / np.sqrt(in_degree[post])
    weights[inhibitory[pre]] *= -1
    slow_edges=np.concatenate(slow_parts)
    plastic_mask=np.concatenate(plastic_parts)
    if modular:
        extra_pre=mapping[np.asarray(local_template.synapses.i[:],int)]
        extra_post=mapping[np.asarray(local_template.synapses.j[:],int)]
        cross_flags=np.r_[np.zeros(len(pre),int),np.asarray(local_template.synapses.cross[:],int)]
        pre=np.r_[pre,extra_pre];post=np.r_[post,extra_post]
        weights=np.r_[weights,np.asarray(local_template.synapses.w[:])]
        slow_edges=np.r_[slow_edges,np.asarray(local_template.synapses.slow[:],bool)]
        plastic_mask=np.r_[plastic_mask,np.zeros(len(extra_pre),bool)]
    if config.motor_cross_tau_ms is not None:
        # Isolate kinetics: conserve the integrated inhibitory charge per spike.
        weights[slow_edges] *= config.synaptic_tau_ms/config.motor_cross_tau_ms
    synapses.connect(i=pre, j=post)
    synapses.w = weights
    if modular:
        synapses.slow_flag=slow_edges.astype(int);synapses.cross_flag=cross_flags
    if config.motor_cross_tau_ms is not None:
        synapses.slow_flag=slow_edges.astype(int)
    synapses.delay = config.delay_min_ms * ms
    monitor = EmittedSpikeMonitor(group, "emit", record=record_spikes, name="veguinha_spikes")
    network = Network(group, synapses, monitor)
    if config.phase5_enabled:
        for leg in motor_pools:
            recurrent = np.isin(pre, regions[f"vnc_{leg}"]) & np.isin(post, regions[f"vnc_{leg}"])
            interneuron_source = np.isin(pre, np.concatenate(tuple(interneurons[leg].values())))
            motor_target = np.isin(post, np.concatenate(tuple(motor_pools[leg].values())))
            plastic_mask[recurrent] &= (interneuron_source & motor_target)[recurrent]
    if config.phase5_enabled and config.motor_plastic_fraction < 1:
        proprio_edges = np.isin(pre, regions.get("proprio", []))
        plastic_mask &= (proprio_edges | (rng.random(len(pre)) < config.motor_plastic_fraction))
    if config.motor_scaffold:
        leg_index = np.full(offset, -1, dtype=np.int8)
        for index, leg in enumerate(motor_pools):
            leg_index[regions[f"vnc_{leg}"]] = index
        proprio_edges = np.isin(pre, regions.get("proprio", []))
        interleg_edges = ((leg_index[pre] >= 0) & (leg_index[post] >= 0)
                          & (leg_index[pre] != leg_index[post]))
        plastic_mask &= proprio_edges | interleg_edges
    if config.phase5_enabled and config.motor_plastic_scope == 'afferent_dn_interleg':
        leg_index = np.full(offset,-1,dtype=np.int8)
        for index,leg in enumerate(motor_pools):
            leg_index[regions[f'vnc_{leg}']]=index
        # Rebuild the allowed set: descending inputs must not be discarded by
        # the legacy scaffold selector; local motor/recurrent edges stay fixed.
        proprio_ids=cord.positions.ravel() if modular else regions['proprio']
        allowed=(np.isin(pre,proprio_ids) & (leg_index[post]>=0))
        allowed |= np.isin(pre,regions['dn']) & (leg_index[post]>=0)
        allowed |= (leg_index[pre]>=0)&(leg_index[post]>=0)&(leg_index[pre]!=leg_index[post])
        plastic_mask = allowed
    plastic_mask &= weights!=0
    plastic_edges = np.flatnonzero(plastic_mask)
    return IsolatedNetwork(network, group, synapses, monitor, regions, inhibitory,
                           cell_types, motor_pools, interneurons, half_centers, subgroups,
                           plastic_edges,cord=cord)


def set_odor_pulse(model: IsolatedNetwork, strength: float, odor: int = 0):
    if odor not in range(4):
        raise ValueError("odor deve estar entre 0 e 3")
    model.group.I_odor = 0
    glomeruli = model.regions["al_glom"]
    channels = np.array_split(glomeruli, 4)
    model.group.I_odor[channels[odor]] = strength


def run_diagnostic(config: Config = Config(), duration_ms: int = 500) -> Diagnostic:
    if duration_ms <= 0:
        raise ValueError("duration_ms deve ser positivo")
    model = build_network(config)
    model.network.run(duration_ms * ms)
    per_cell = np.asarray(model.monitor.count[:], dtype=int)
    counts = {name: int(per_cell[indices].sum()) for name, indices in model.regions.items()}
    rates = {
        name: count / len(model.regions[name]) / (duration_ms / 1000)
        for name, count in counts.items()
    }
    spike_times = np.asarray(model.monitor.t[:] / ms)
    spike_neurons = np.asarray(model.monitor.i[:], dtype=int)
    late_counts = np.bincount(spike_neurons[spike_times >= duration_ms / 2],
                              minlength=len(model.group))
    late_rates = {
        name: float(late_counts[indices].sum() / len(indices) / (duration_ms / 2000))
        for name, indices in model.regions.items()
    }
    window_start = max(0, duration_ms - 50)
    recent = spike_neurons[spike_times >= window_start]
    kc_active = np.isin(model.regions["kc"], recent).mean()
    warnings = []
    for name, rate in late_rates.items():
        if rate > config.hyperactivity_hz:
            warnings.append(f"{name}: hiperatividade ({rate:.1f} Hz)")
        elif rate < config.silence_hz:
            warnings.append(f"{name}: silêncio ({rate:.1f} Hz)")
    if kc_active >= config.kc_sparse_limit:
        warnings.append(f"kc: código pouco esparso ({kc_active:.1%} ativos em 50 ms)")
    return Diagnostic(rates, late_rates, counts, warnings, len(model.group),
                      len(model.synapses), float(model.inhibitory.mean()), float(kc_active))
