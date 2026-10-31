from dataclasses import dataclass
from math import isfinite

from .regions import BASE_REGIONS, Pathway, Region, pathways as default_pathways


@dataclass(frozen=True)
class Config:
    body_model: str = "control"
    seed: int = 42
    scale: float = 1.0
    backend: str = "cython"
    dt_ms: float = 1.0
    synaptic_tau_ms: float = 5.0
    noise: float = 2.5
    block_ms: int = 10
    proprio_gain: float = 2.0
    proprio_projection_strength: float = 8.0
    foot_contact_gain: float = 1.0
    foot_descending_gain: float = 0.25
    touch_vnc_gain: float = 0.8
    touch_descending_gain: float = 0.8
    touch_seg_gain: float = 2.0
    touch_lh_gain: float = 1.0
    odor_gain: float = 12.0
    energy_gain: float = 3.0
    efference_gain: float = 1.0
    motor_filter_ms: float = 120.0
    motor_baseline_ms: float = 5000.0
    motor_gain: float = 0.18
    motor_limit: float = 1.0
    manual_test_mode: bool = False
    ecology_enabled: bool = False
    prey_count: int = 2
    prey_radius_m: float = .025
    prey_energy: float = 1.5
    prey_bite_rate: float = .5
    prey_bite_damage: float = 2.
    dust_deposition_rate: float = .003
    dust_rub_gain: float = 8.
    dust_adhesion_accel: float = 20.
    dust_shed_gain: float = .5
    fatigue_gain: float = .012
    fatigue_recovery: float = .004
    feeding_enabled: bool = False
    vision_enabled: bool = False
    binocular_enabled: bool = False
    optical_flow_enabled: bool = False
    optical_flow_gain: float = 3.
    vision_fov_deg: float = 90.
    vision_pitch_deg: float = -15.
    vision_baseline_m: float = .18
    binocular_current_gain: float = 40.
    binocular_projection_gain: float = 6.
    nociception_enabled: bool = False
    feeding_rate: float = .25
    feeding_contact_gain: float = 24.
    intent_enabled: bool = False
    motor_gate_strength: float = 10.
    intent_hunger_gain: float = 18.
    intent_hunger_span: float = .05
    intent_odor_gain: float = 4.
    intent_rest_gain: float = 8.
    intent_curiosity_gain: float = 12.
    intent_pain_gain: float = 40.
    homeostasis_enabled: bool = False
    odor_adaptation_tau_s: float = 8.
    odor_adaptation_strength: float = 1.5
    homeostatic_odor_gain: float = 1.
    intent_fatigue_gain: float = 12.
    fatigue_drive_suppression: float = .75
    vision_width: int = 48
    vision_height: int = 24
    vision_interval_ms: float = 100.
    foveated_enabled: bool = False
    vision_periphery_ms: float = 100.
    vision_workers: int = 1
    somatic_enabled: bool = False
    taste_gain: float = 10.
    tactile_gain: float = 2.
    looming_gain: float = 80.
    ball_taste: tuple[float,...] = (.8,.1,0.)
    prey_taste: tuple[float,...] = (.15,.95,0.)
    pruning_enabled: bool = False
    prune_weak_ratio: float = .3
    prune_after_s: float = 30.
    prune_max_fraction: float = .1
    mouth_contact_gain: float = 6.
    nutrition_gain: float = 8.
    vision_gain: float = 3.
    pain_gain: float = 12.
    pain_force_threshold_n: float = 2.
    pain_force_scale_n: float = 6.
    pain_decay_s: float = .3
    phase5_enabled: bool = False
    neural_value_channels: bool = False
    flow_reference_m_s: float = .05
    motor_plastic_scope: str = "legacy"
    flow_gain: float = 2.0
    value_flow_tau_ms: float = 200.0
    value_flow_gain: float = 4.0
    value_collision_gain: float = 3.0
    value_instability_gain: float = 1.0
    value_rotation_scale: float = 3.0
    value_energy_cost_gain: float = 30.0
    eligibility_tau_s: float = 1.0
    learning_rate: float = 0.00005
    vnc_hebb_rate: float = 0.0000003
    motor_plastic_fraction: float = 0.25
    octopamine_gain: float = 0.8
    motor_scaffold: bool = False
    motor_circuit: str = 'legacy'
    motor_cross_strength: float = 2.0
    motor_position_gain: float = 4.0
    motor_load_gain: float = 12.0
    motor_pool_bias: float | None = None
    motor_cross_tau_ms: float | None = None
    half_center_recurrence: float = 22.0
    half_center_recruitment: float = 28.0
    half_center_motor: float = 24.0
    half_center_cross_inhibition: float = 68.0
    half_center_motor_inhibition: float = 22.0
    half_center_dn_drive: float = 8.0
    half_center_interleg_strength: float = 7.0
    half_center_adaptation_a: float = 0.003
    half_center_adaptation_d: float = 2.0
    half_center_inhibitory_bias: float = -2.0
    half_center_exc_bias: float | None = None
    scaffold_sparse: bool = False
    motor_damage_leg: str = ""
    foot_contact_angle: float | None = None
    regions: tuple[Region, ...] = BASE_REGIONS
    pathways: tuple[Pathway, ...] = default_pathways()
    delay_min_ms: int = 2
    delay_max_ms: int = 2
    silence_hz: float = 0.1
    hyperactivity_hz: float = 80.0
    kc_sparse_limit: float = 0.15

    def __post_init__(self):
        if (not isfinite(self.odor_adaptation_tau_s) or self.odor_adaptation_tau_s<=0
            or any(not isfinite(v) or v<0 for v in (self.odor_adaptation_strength,self.homeostatic_odor_gain,self.intent_fatigue_gain))
            or not 0<=self.fatigue_drive_suppression<=1):
            raise ValueError('parâmetros de regulação homeostática inválidos')
        if type(self.vision_workers) is not int or self.vision_workers not in (1,2,4):raise ValueError('visão aceita 1, 2 ou 4 trabalhadores')
        if any(not isfinite(v) or v<0 for v in (self.taste_gain,self.tactile_gain,self.looming_gain)) or any(len(v)!=3 or any(not isfinite(x) or not 0<=x<=1 for x in v) for v in (self.ball_taste,self.prey_taste)):
            raise ValueError('ganhos/composição sensorial inválidos')
        if not 0<self.prune_weak_ratio<1 or not isfinite(self.prune_after_s) or self.prune_after_s<=0 or not 0<=self.prune_max_fraction<=.1:raise ValueError('limites de poda inválidos')
        if not 0<self.prey_radius_m<=.05:raise ValueError('raio de presa inválido')
        if self.foveated_enabled and (not self.binocular_enabled or self.vision_width%4 or self.vision_height%4 or not 20<=self.vision_periphery_ms<=self.vision_interval_ms or self.vision_periphery_ms%self.block_ms or self.vision_interval_ms%self.vision_periphery_ms):
            raise ValueError('visão foveada requer olhos binoculares, dimensões múltiplas de4 e período periférico válido')
        if self.ecology_enabled and (self.motor_circuit!='modular' or not self.intent_enabled):
            raise ValueError('ecologia requer o circuito mk1 modular')
        if self.prey_count not in (1,2) or any(not isfinite(v) or v<0 for v in (self.prey_energy,self.prey_bite_rate,self.prey_bite_damage,self.dust_deposition_rate,self.dust_rub_gain,self.dust_adhesion_accel,self.dust_shed_gain,self.fatigue_gain,self.fatigue_recovery)) or self.prey_energy<=0 or self.dust_adhesion_accel<=0:
            raise ValueError('parâmetros ecológicos inválidos')
        if self.optical_flow_enabled and not self.vision_enabled:
            raise ValueError('fluxo óptico requer imagens dos olhos')
        if not 0 < self.intent_hunger_span <= 1 or not 0 <= self.optical_flow_gain <= 20:
            raise ValueError('escala homeostática ou ganho óptico inválido')
        if self.binocular_enabled and not self.vision_enabled:
            raise ValueError('visão binocular requer imagens dos olhos')
        if (not 30 <= self.vision_fov_deg <= 120 or not -60 <= self.vision_pitch_deg <= 30
                or not .01 <= self.vision_baseline_m <= .3
                or not 0 < self.binocular_current_gain <= 100
                or not 0 <= self.binocular_projection_gain <= 20):
            raise ValueError('calibração binocular inválida')
        if any(not isinstance(v,bool) for v in (self.feeding_enabled,self.vision_enabled,self.nociception_enabled)):
            raise ValueError('habilitação sensorial deve ser booleana')
        if self.intent_enabled and self.motor_circuit!='modular':
            raise ValueError('competição descendente requer VNC modular')
        if any(v<0 or not isfinite(v) for v in (self.feeding_rate,self.feeding_contact_gain,self.motor_gate_strength,self.intent_hunger_gain,self.intent_odor_gain,self.intent_rest_gain,self.intent_curiosity_gain,self.intent_pain_gain,self.mouth_contact_gain,self.nutrition_gain,self.vision_gain,self.pain_gain,self.pain_force_threshold_n)):
            raise ValueError('ganho sensorial inválido')
        if not (16<=self.vision_width<=640 and 8<=self.vision_height<=360 and self.vision_interval_ms>=20):
            raise ValueError('resolução ou período visual inválido')
        if any(not isfinite(v) or v<=0 for v in (self.pain_force_scale_n,self.pain_decay_s)):
            raise ValueError('escala de nocicepção inválida')
        if self.motor_circuit not in ('legacy','modular'):
            raise ValueError('circuito motor desconhecido')
        if self.motor_circuit=='modular' and (self.body_model!='mk1' or self.scale!=1 or self.dt_ms!=.5 or self.motor_scaffold or self.motor_cross_tau_ms is not None):
            raise ValueError('circuito modular requer mk1, escala1, dt0.5ms e andaime antigo desligado')
        if self.motor_circuit=='modular' and self.phase5_enabled and (
                not self.neural_value_channels or self.motor_plastic_scope!='afferent_dn_interleg'):
            raise ValueError('plasticidade modular requer valor neural e escopo afferent_dn_interleg')
        if not 0<=self.motor_cross_strength<=20 or not 0<=self.motor_position_gain<=100 or not 0<=self.motor_load_gain<=100:
            raise ValueError('ganho do módulo motor inválido')
        if self.body_model not in ("control", "mk1"):
            raise ValueError("body_model deve ser control ou mk1")
        if self.motor_plastic_scope not in ('legacy','afferent_dn_interleg'):
            raise ValueError('escopo de plasticidade inválido')
        if not self.flow_reference_m_s > 0:
            raise ValueError('referência de fluxo deve ser positiva')
        if self.motor_pool_bias is not None and not -10 <= self.motor_pool_bias <= 10:
            raise ValueError('corrente basal motora inválida')
        if self.motor_cross_tau_ms is not None and not self.dt_ms < self.motor_cross_tau_ms <= 500:
            raise ValueError('tau da inibição motora inválido')
        if not 0.05 <= self.scale <= 10.0:
            raise ValueError("scale deve ficar entre 0.05 e 10.0")
        if self.backend not in ("numpy", "cython"):
            raise ValueError("backend deve ser numpy ou cython")
        if not 0 < self.dt_ms <= 1.0:
            raise ValueError("dt_ms deve ficar entre 0 e 1")
        if self.noise < 0 or self.synaptic_tau_ms <= 0:
            raise ValueError("ruído e tau sináptica devem ser válidos")
        if self.block_ms not in (10, 20) or self.block_ms % self.dt_ms:
            raise ValueError("bloco deve ser 10 ou 20 ms e múltiplo de dt")
        if (not 0.5 <= self.eligibility_tau_s <= 2.0 or self.learning_rate < 0
                or self.vnc_hebb_rate < 0):
            raise ValueError("plasticidade motora inválida")
        if not 0 < self.motor_plastic_fraction <= 1:
            raise ValueError("fração de sinapses plásticas inválida")
        scaffold_gains = (self.half_center_recurrence, self.half_center_recruitment,
                          self.half_center_motor, self.half_center_cross_inhibition,
                          self.half_center_motor_inhibition, self.half_center_dn_drive,
                          self.half_center_interleg_strength)
        if (not isinstance(self.motor_scaffold, bool) or any(gain <= 0 for gain in scaffold_gains)
                or self.half_center_adaptation_a <= 0 or self.half_center_adaptation_d <= 0
                or not -10 <= self.half_center_inhibitory_bias <= 10
                or (self.half_center_exc_bias is not None
                    and not -10 <= self.half_center_exc_bias <= 10)
                or not isinstance(self.scaffold_sparse, bool)):
            raise ValueError("parâmetros do andaime motor inválidos")
        if self.motor_damage_leg not in ("", "lf", "lm", "lh", "rf", "rm", "rh"):
            raise ValueError("perna danificada inválida")
        if self.foot_contact_angle is not None and not -1.5 <= self.foot_contact_angle <= 1.5:
            raise ValueError("limiar de contato inválido")

        gains = (self.proprio_gain, self.proprio_projection_strength,
                 self.foot_contact_gain, self.foot_descending_gain,
                 self.touch_vnc_gain, self.touch_descending_gain, self.touch_seg_gain,
                 self.touch_lh_gain, self.odor_gain, self.energy_gain,
                 self.efference_gain, self.motor_gain, self.flow_gain,
                 self.value_flow_gain, self.value_collision_gain,
                 self.value_instability_gain, self.value_rotation_scale,
                 self.value_energy_cost_gain,
                 self.octopamine_gain)
        if (any(gain < 0 for gain in gains) or self.motor_filter_ms <= 0
                or self.motor_baseline_ms <= 0 or self.motor_limit <= 0
                or self.value_flow_tau_ms <= 0):
            raise ValueError("ganhos e constantes de tempo devem ser positivos")
        if self.delay_min_ms < 1 or self.delay_max_ms < self.delay_min_ms:
            raise ValueError("intervalo de atrasos inválido")
        names = {region.name for region in self.regions}
        if len(names) != len(self.regions) or not names or "kc" not in names:
            raise ValueError("regiões devem ter nomes únicos e incluir kc")
        if any(region.cells < 1 or not 0 <= region.inhibitory <= region.cells
               for region in self.regions):
            raise ValueError("tamanho ou fração inibitória inválidos")
        if any(path.source not in names or path.target not in names
               or path.in_degree < 1 or path.weight <= 0 for path in self.pathways):
            raise ValueError("projeção inválida")

    @property
    def joint_names(self):
        return ("coxa", "femur", "tibia") if self.body_model == "mk1" else ("hip", "knee")

    @property
    def contact_threshold(self):
        return (0.0 if self.phase5_enabled else 0.35) if self.foot_contact_angle is None else self.foot_contact_angle
