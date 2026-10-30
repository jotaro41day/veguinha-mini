"""Sensores físicos convertidos em correntes de neurônios específicos."""

import numpy as np

from .body import LEG_NAMES
from .config import Config


def homeostatic_need(energy, span):
    """Continuous reserve deficit, saturated at the receptor's dynamic range."""
    return float(np.clip((1-np.clip(energy,0,1))/span,0,1))


class ValueFlowFilter:
    """Memória sensorial curta do fluxo usado pela região de valor."""

    def __init__(self, tau_ms):
        if tau_ms <= 0:
            raise ValueError("tau do fluxo de valor deve ser positivo")
        self.tau_ms = float(tau_ms)
        self.state = np.zeros(3, dtype=float)

    def update(self, flow, dt_s):
        alpha = -np.expm1(-1000.0 * dt_s / self.tau_ms)
        self.state += alpha * (np.asarray(flow, dtype=float) - self.state)
        return self.state.copy()

    def reset(self):
        self.state[:] = 0.0


def inject_sensors(model, body_sample, odor_type=0, config=None, efference=None,
                   value_flow=None):
    """Injeta aferências e uma cópia motora; nenhuma saída é escolhida aqui."""
    if odor_type not in range(4):
        raise ValueError("odor_type deve estar entre 0 e 3")
    config = config or Config()
    # Interoceptive receptor gain, not a behavioral state or action threshold.
    energy=np.clip(float(body_sample['energy']),0,1)
    hunger=homeostatic_need(energy,config.intent_hunger_span if getattr(model,'intent',None) is not None else 1.)
    currents = np.zeros(len(model.group), dtype=float)
    angles = np.asarray(body_sample["angles"], dtype=float)
    contacts = np.asarray(body_sample["contacts"], dtype=float)
    modular=getattr(model,'cord',None) is not None
    for index, leg in enumerate(LEG_NAMES):
        if modular:
            continue
        pools = model.motor_pools[leg]
        for joint_index, joint in enumerate(config.joint_names):
            angle = angles[index, joint_index]
            if config.phase5_enabled:
                channels = model.regions["proprio"].reshape(6, len(config.joint_names), 2, -1)
                currents[channels[index, joint_index, 0]] += config.proprio_gain * max(0.0, -angle)
                currents[channels[index, joint_index, 1]] += config.proprio_gain * max(0.0, angle)
            else:
                currents[pools[f"{joint}_flex"]] += config.proprio_gain * max(0.0, -angle)
                currents[pools[f"{joint}_extend"]] += config.proprio_gain * max(0.0, angle)
        currents[model.interneurons[leg]["inhibitory"]] += config.foot_contact_gain * contacts[index]
        currents[model.interneurons[leg]["excitatory"]] += config.touch_vnc_gain * float(body_sample["touch"])
    if modular:
        from .cord import cord_currents
        neutral=getattr(model,'motor_neutral',np.zeros_like(angles))
        signal=np.asarray(body_sample.get('load_signal',np.clip(np.asarray(body_sample['normal_load'])/(.316*9.81/6),0,1)))
        currents+=cord_currents(model.cord,angles-neutral,signal*(.316*9.81/6),config.motor_position_gain,config.motor_load_gain)
        currents[model.cord.loads.ravel()]+=config.touch_vnc_gain*float(body_sample['touch'])
        contacts=np.asarray(body_sample.get('contact_fraction',contacts),float)
    currents[model.regions["dn"]] += (config.foot_descending_gain * contacts.mean()
                                      + config.touch_descending_gain * float(body_sample["touch"]))
    currents[model.regions["seg"]] += config.touch_seg_gain * float(body_sample["touch"])
    if config.feeding_enabled:
        seg=model.regions['seg'];exc=seg[~model.inhibitory[seg]]
        close=np.array_split(exc[:min(16,len(exc))],2)[0]
        currents[close]+=config.mouth_contact_gain*float(body_sample.get('mouth_contact',0))
    currents[model.regions["lh"]] += config.touch_lh_gain * float(body_sample["touch"])
    if not config.phase5_enabled:
        currents[model.regions["value"]] += config.energy_gain * hunger
    if config.phase5_enabled:
        forward, lateral, rotation = np.asarray(body_sample["flow"], dtype=float)
        flow = model.regions["flow"].reshape(3, 2, -1)
        for axis, velocity in enumerate((forward, lateral, rotation)):
            if config.neural_value_channels:
                velocity /= config.flow_reference_m_s if axis<2 else .5
            currents[flow[axis, 0]] += config.flow_gain * np.clip(velocity, 0, 4)
            currents[flow[axis, 1]] += config.flow_gain * np.clip(-velocity, 0, 4)
        value_forward, value_lateral, value_rotation = (
            np.asarray(value_flow, dtype=float) if value_flow is not None
            else (forward, lateral, rotation))
        instability = 0.25 * abs(value_lateral) + config.value_rotation_scale * abs(value_rotation)
        metabolic_cost = config.value_energy_cost_gain * float(body_sample["energy_cost_rate"])
        if config.neural_value_channels:
            # Independent sensor channels excite E or I value neurons. Their
            # spikes and signed synapses determine DAN activity. No scalar
            # reward or dopamine is injected into the DANs or learning rule.
            ids=model.regions['value']
            exc=np.array_split(ids[~model.inhibitory[ids]],2)
            inh=np.array_split(ids[model.inhibitory[ids]],5)
            vf=value_forward/config.flow_reference_m_s
            vl=value_lateral/config.flow_reference_m_s
            vr=value_rotation/.5
            imu_tilt=max(0.,1.-float(body_sample.get('upright',1.)))
            currents[exc[0]]+=config.value_flow_gain*np.clip(vf,0,4)
            currents[exc[1]]+=config.energy_gain*hunger
            currents[exc[1]]+=config.nutrition_gain*float(body_sample.get('food_intake_rate',0))
            signals=(config.value_flow_gain*np.clip(-vf,0,4),
                     config.value_instability_gain*.25*abs(vl),
                     config.value_instability_gain*(abs(vr)+4*imu_tilt),
                     config.energy_gain*metabolic_cost,
                     config.value_collision_gain*float(body_sample['touch'] or body_sample.get('fell',False)))
            for targets,signal in zip(inh,signals):currents[targets]+=signal
            currents[model.regions['oct']]+=config.octopamine_gain*(
                np.clip(.5-vf,0,4)+.25*abs(vl)+abs(vr)+4*imu_tilt+float(body_sample['touch']))
        else:
            value_signal = (config.value_flow_gain * np.clip(value_forward, -4, 4)
                            - config.value_collision_gain * float(body_sample["touch"])
                            - config.value_instability_gain * instability
                            - config.energy_gain * metabolic_cost)
            currents[model.regions["value"]] += config.energy_gain * hunger
            currents[model.regions["value"]] += value_signal
            currents[model.regions["dan"]] += value_signal
            currents[model.regions["oct"]] += config.octopamine_gain * (
                max(0.0, 0.5 - value_forward) + instability + float(body_sample["touch"]))
    if config.vision_enabled:
        retina=np.asarray(body_sample.get('retina',np.zeros((2,8,2))),float).copy()
        if config.optical_flow_enabled:retina[:,:,2:]=0
        pixels=retina.ravel()
        ids=model.regions['cx'];exc=ids[~model.inhibitory[ids]]
        for targets,signal in zip(np.array_split(exc,len(pixels)),pixels):
            currents[targets]+=config.vision_gain*signal
        optical=body_sample.get('optical_flow')
        if config.optical_flow_enabled and optical is not None:
            velocity=np.asarray(optical['angular_rad_s'],float)
            confidence=np.asarray(optical['confidence'],float)
            directional=np.stack([np.maximum(velocity[...,0],0),np.maximum(-velocity[...,0],0),
                                  np.maximum(velocity[...,1],0),np.maximum(-velocity[...,1],0)],axis=-1)
            signal=np.clip(directional*confidence[...,None],0,2).mean(axis=1).ravel()
            for targets,value in zip(np.array_split(exc,len(signal)),signal):
                currents[targets]+=config.optical_flow_gain*value
    if config.nociception_enabled:
        pain=np.clip(np.asarray(body_sample.get('pain',[0,0]),float),0,1)
        ids=model.regions['value'];inh=ids[model.inhibitory[ids]]
        currents[inh]+=config.pain_gain*float(pain.sum())
        ids=model.regions['dn'];inh=ids[model.inhibitory[ids]]
        if getattr(model,'intent',None) is None:
            for targets,signal in zip(np.array_split(inh,2),pain):currents[targets]+=.25*config.pain_gain*signal
        if 'oct' in model.regions:currents[model.regions['oct']]+=.5*config.pain_gain*float(pain.sum())
    intent=getattr(model,'intent',None)
    if intent is not None:
        currents[intent.go]+=config.intent_hunger_gain*hunger+config.intent_curiosity_gain*float(body_sample.get('novelty',0))
        # Weak innate odor valence is an afferent drive, never a turn/target.
        odor_valence=(1.,0.,-1.,0.)[odor_type]
        odor_need=config.intent_odor_gain*hunger*float(np.clip(np.mean(body_sample['antenna_odor']),0,1))
        currents[intent.go]+=odor_need*max(0.,odor_valence)
        if config.ecology_enabled:
            prey_odor=np.clip(np.asarray(body_sample.get('prey_odor',[0,0]),float),0,1)
            currents[intent.go]+=1.5*config.intent_odor_gain*hunger*float(prey_odor.mean())
        currents[intent.hold]+=config.intent_rest_gain*(1-hunger)
        currents[intent.hold]+=odor_need*max(0.,-odor_valence)
        currents[intent.hold]+=config.feeding_contact_gain*float(body_sample.get('mouth_contact',0))*(.25+hunger)
        currents[intent.hold]+=config.intent_pain_gain*np.clip(np.sum(body_sample.get('pain',[0,0])),0,1)
    if efference is not None:
        for targets, value in zip(np.array_split(model.regions["comparator"], np.size(efference)),
                                  np.asarray(efference, dtype=float).ravel()):
            currents[targets] += config.efference_gain * value
    if config.somatic_enabled:
        taste=np.clip(np.asarray(body_sample.get('taste',np.zeros((2,3))),float),0,1).mean(axis=0)
        sweet,protein,bitter=taste
        seg=model.regions['seg'];exc=seg[~model.inhibitory[seg]]
        close,opened=np.array_split(exc[:min(16,len(exc))],2)
        currents[close]+=config.taste_gain*(sweet+.6*protein)
        currents[opened]+=config.taste_gain*bitter
        value=model.regions['value']
        currents[value[~model.inhibitory[value]]]+=1.5*(sweet+protein)
        currents[value[model.inhibitory[value]]]+=config.taste_gain*bitter
        tactile=np.maximum(np.asarray(body_sample.get('tactile',np.zeros((9,3))),float),0)
        signal=np.clip(np.log1p((tactile[:,0]+.5*tactile[:,1]+2*tactile[:,2])/.5),0,2)
        # Tonic foot load already has its own afferent; add shear/onset without duplicating it.
        leg_signal=np.clip(np.log1p((.5*tactile[:6,1]+2*tactile[:6,2])/.5),0,2)
        for k,leg in enumerate(LEG_NAMES):
            ids=model.cord.loads[k].ravel() if modular else model.interneurons[leg]['excitatory']
            currents[ids]+=config.tactile_gain*leg_signal[k]
        currents[seg]+=config.tactile_gain*float(signal[6:].mean())
        if intent is not None:
            currents[intent.hold]+=config.looming_gain*float(np.clip(body_sample.get('looming',0),0,1))
            currents[intent.hold]+=config.tactile_gain*float(signal[6:].max())+config.taste_gain*bitter
    odor_currents = np.zeros(len(model.group), dtype=float)
    glomeruli = np.array_split(model.regions["al_glom"], 4)[odor_type]
    left, right = np.array_split(glomeruli, 2)
    antenna = body_sample["antenna_odor"]
    odor_currents[left] = config.odor_gain * antenna[0]
    odor_currents[right] = config.odor_gain * antenna[1]
    if config.ecology_enabled:
        left,right=np.array_split(np.array_split(model.regions['al_glom'],4)[1],2)
        prey=np.clip(np.asarray(body_sample.get('prey_odor',[0,0]),float),0,1)
        odor_currents[left]=config.odor_gain*prey[0];odor_currents[right]=config.odor_gain*prey[1]
        state=body_sample.get('ecology')
        if state is not None:
            ids=model.regions['value'];currents[ids[model.inhibitory[ids]]]+=2*float(np.mean(state['dust']))+3*float(state['fatigue'])
    model.group.I_sense = currents
    model.group.I_odor = odor_currents
