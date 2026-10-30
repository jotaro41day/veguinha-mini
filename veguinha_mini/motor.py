"""Decodificador de spikes em torque; sem pesos treináveis ou seletor de marcha."""

import math

import numpy as np

from .body import LEG_NAMES


POOL_NAMES = ("hip_flex", "hip_extend", "knee_flex", "knee_extend")


class NeuralMouth:
    """Antagonistic oral muscle activation decoded only from SEG spikes."""
    def __init__(self,model):
        cells=model.regions['seg'];exc=cells[~model.inhibitory[cells]]
        self.close,self.open=np.array_split(exc[:min(16,len(exc))],2)
        self.previous=np.zeros(len(model.group),dtype=np.int64)
        self.rates=np.zeros(2)

    def update(self,counts,dt):
        counts=np.asarray(counts,dtype=np.int64);delta=counts-self.previous;self.previous=counts.copy()
        rates=np.array([delta[p].mean()/dt if len(p) else 0 for p in (self.close,self.open)])
        self.rates+=-math.expm1(-dt/.12)*(rates-self.rates)
        difference=(self.rates[0]-self.rates[1])*.05
        return float(np.clip(difference,0,1)),float(np.clip(-difference,0,1))



class MotorDecoder:
    def __init__(self, model, config, legs=LEG_NAMES):
        self.config = config
        self.legs = tuple(legs)
        self.pool_names = tuple(f"{joint}_{action}" for joint in config.joint_names
                                for action in ("flex", "extend"))
        self.pools = np.array([[model.motor_pools[leg][name]
                                for name in self.pool_names] for leg in self.legs])
        self.pool_size = self.pools.shape[-1]
        self.previous = np.zeros(len(model.group), dtype=np.int64)
        self.filtered_hz = np.zeros((len(self.legs), len(self.pool_names)), dtype=float)
        self.baseline_hz = np.zeros_like(self.filtered_hz)
        self.torques = np.zeros((len(self.legs), len(config.joint_names)), dtype=float)
        self.saturation_samples = 0
        self.samples = 0

    def update(self, counts, dt_s):
        delta = np.asarray(counts, dtype=np.int64) - self.previous
        self.previous = np.asarray(counts, dtype=np.int64).copy()
        instantaneous = delta[self.pools].sum(axis=-1) / self.pool_size / dt_s
        fast = -math.expm1(-dt_s * 1000 / self.config.motor_filter_ms)
        slow = -math.expm1(-dt_s * 1000 / self.config.motor_baseline_ms)
        self.filtered_hz += fast * (instantaneous - self.filtered_hz)
        self.baseline_hz += slow * (self.filtered_hz - self.baseline_hz)
        deviation = self.filtered_hz - self.baseline_hz
        difference = deviation[:, 0::2] - deviation[:, 1::2]
        activity = self.filtered_hz[:, 0::2] + self.filtered_hz[:, 1::2]
        drive = self.config.motor_gain * difference * np.clip(activity / 6.0, 0, 1)
        self.saturation_samples += int(np.count_nonzero(np.abs(drive) >= self.config.motor_limit))
        self.samples += drive.size
        self.torques = np.clip(drive, -self.config.motor_limit, self.config.motor_limit)
        return self.torques.copy()

    def pool_rates(self):
        return {leg: dict(zip(self.pool_names, np.round(self.filtered_hz[index], 2).tolist()))
                for index, leg in enumerate(self.legs)}

    def saturation_fraction(self):
        return self.saturation_samples / max(1, self.samples)
