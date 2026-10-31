"""Malha fechada Brian2/corpo, independente dos clientes web."""

import csv
import math
import os
import pickle
import queue
import threading
import time
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
from brian2 import Clock, NetworkOperation, ms

from .config import Config
from .body import Body, LEG_NAMES
from .mujoco_body import MujocoBody
from .learning import MotorPlasticity, kenyon_homeostasis
from .metrics import Metrics
from .motor import MotorDecoder,NeuralMouth
from .network import build_network, set_odor_pulse
from .sensory import ValueFlowFilter, inject_sensors
from .world import World
from .intent import status as intent_status


class Simulation:
    def __init__(self, config=Config(), headless=False, metrics_csv=None,
                 reward_mode="coupled", body_substep_ms=20, physics="legacy"):
        if not 0 < body_substep_ms <= 20:
            raise ValueError("subpasso físico inválido")
        self.config = config
        self.body_substep_ms = body_substep_ms
        if physics not in ("legacy", "mujoco"):
            raise ValueError("física desconhecida")
        self.physics = physics
        if config.body_model == "mk1" and physics != "mujoco":
            raise ValueError("mk1 requer física MuJoCo")
        if (config.feeding_enabled or config.vision_enabled or config.nociception_enabled) and physics!='mujoco':
            raise ValueError('percepção e alimentação 3D requerem MuJoCo')
        self.headless = headless
        self.metrics_csv = metrics_csv
        self.commands = queue.Queue(maxsize=100)
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.paused = False
        self.speed = 1
        self.plasticity = config.motor_circuit!='modular'
        self.noise = config.noise
        self.reward_mode = reward_mode
        self.damaged_leg = config.motor_damage_leg
        self.learning_curve = []
        self.world = World(shade_enabled=config.ecology_enabled)
        self.viewer_port = None
        self.seq = 0
        self.epoch = -1
        self.reset_requested = False
        self.last_tick_wall = 0.0
        self._build()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="veguinha-brian2")

    def _build(self):
        self.epoch += 1
        self.body = self._new_body()
        self.manual_torques = np.zeros((6, len(self.config.joint_names)), dtype=float)
        self.contacts = (self.body.contacts.copy() if self.physics == "mujoco"
                         else np.ones(6, dtype=bool))
        self.model = build_network(self.config, record_spikes=not self.headless)
        if self.config.motor_circuit=='modular':
            self.model.motor_neutral=self.body.angles.copy()
            lateral=self.body.data.xmat[self.body.body_id].reshape(3,3)[:,1]
            sides=(self.body.data.xpos[self.body.hip_ids]-self.body.data.xpos[self.body.body_id])@lateral
            self.model.motor_side=dict(zip(LEG_NAMES,np.sign(sides)))
        self.model.group.noise = self.noise
        self.motor = MotorDecoder(self.model, self.config)
        self.mouth=NeuralMouth(self.model) if self.config.feeding_enabled else None
        from .intent import build_intent
        self.intent=build_intent(self.model,self.config) if self.config.intent_enabled else None
        self.maintenance=None;self.prey=None
        if self.config.ecology_enabled:
            from .maintenance import MaintenanceCircuit
            from .ecology import PreyCircuit
            self.maintenance=MaintenanceCircuit(self.config,self.model)
            self.prey=PreyCircuit(self.config,Clock(dt=1*ms,name='prey_clock'),self.model.network,self.config.prey_count)
        from .binocular import BinocularCircuit
        self.binocular=BinocularCircuit(self.config,Clock(dt=1*ms,name='binocular_clock'),self.model.network,self.model) if self.config.binocular_enabled else None
        if self.binocular is not None:self._binocular_input()
        self.value_filter = ValueFlowFilter(self.config.value_flow_tau_ms)
        self.motor_learning = MotorPlasticity(self.model, self.config, self.reward_mode) if self.config.phase5_enabled else None
        self.metrics = Metrics(self.body)
        self.falls_seen = False
        self.window_metrics = None
        self.metrics_writer = None
        self.metrics_output = None
        self.next_csv_ms = 1000
        names = list(self.model.regions)
        self.region_index = np.empty(len(self.model.group), dtype=np.int16)
        for index, name in enumerate(names):
            self.region_index[self.model.regions[name]] = index
        self.names = names
        pre = np.asarray(self.model.synapses.i[:], dtype=np.int32)
        post = np.asarray(self.model.synapses.j[:], dtype=np.int32)
        weights = np.asarray(self.model.synapses.w[:], dtype=np.float32)
        if self.intent is not None:
            recurrent=self.intent.synapses[0]
            pre=np.r_[pre,recurrent.i[:]];post=np.r_[post,recurrent.j[:]]
            weights=np.r_[weights,recurrent.w[:]]
        pair = self.region_index[pre] * len(names) + self.region_index[post]
        sums = np.bincount(pair, weights=weights, minlength=len(names) ** 2)
        counts = np.bincount(pair, minlength=len(names) ** 2)
        matrix = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        self.matrix = np.round(matrix.reshape(len(names), len(names)), 3).tolist()
        self.spike_cursor = 0
        self.rate_count = np.zeros(len(self.model.group), dtype=np.int64)
        self.homeo_count = np.zeros(len(self.model.group), dtype=np.int64)
        self.rate_time_ms = 0.0
        self.homeo_time_ms = 0.0
        self.rates = [0.0] * len(names)
        self.odor_end_ms = 0.0
        self.odor_pulse = 0.0
        self.simulated_ms = 0.0
        self.episode_start_ms=0.
        self.next_learning_point_ms = 10000
        if not self.config.manual_test_mode:
            inject_sensors(self.model, self.body.snapshot(self.world, self.contacts), config=self.config)
        self._ecology_input()
        body_clock = Clock(dt=self.config.block_ms * ms, name="veguinha_body_clock")
        self.model.network.add(NetworkOperation(self._tick, clock=body_clock,
                                                when="end", name="veguinha_body_tick"))
        with self.lock:
            self.seq += 1
            self.frame = {"type": "frame", "seq": self.seq, "epoch": self.epoch,
                          "t": 0, "episode_t_ms":0, "rates": self.rates,
                          "spikes": [], "paused": self.paused, "speed": self.speed,
                          "actual_speed": 0, "plasticity": bool(self.motor_learning and self.plasticity),
                          "ablated": [], "noise": self.noise,
                          "body": self.body.snapshot(self.world, self.contacts),
                          "world": self.world.snapshot(),
                          "manual": self._manual_state() if self.config.manual_test_mode else None,
                          "metrics": self.metrics.snapshot(self.body),
                          "learning_curve": self.learning_curve[-120:],
                          "neuromodulation": self.motor_learning.summary() if self.motor_learning else None,
                          "intent": intent_status(self.intent,self.body.energy,self.config),
                          "homeostasis": self._homeostatic_state(),
                          "damaged_leg": self.damaged_leg,
                          "matrix": self.matrix,
                          "motor_rates": self.motor.pool_rates(), "motor_saturation": 0}

    def _new_body(self):
        body=(MujocoBody(self.world, dt=self.body_substep_ms / 1000, variant=self.config.body_model,
                         sample_contacts=self.config.motor_circuit=='modular',prey_count=self.config.prey_count if self.config.ecology_enabled else 0,prey_radius_m=self.config.prey_radius_m)
                if self.physics == "mujoco" else Body(self.world))
        if self.config.motor_circuit=='modular':
            for _ in range(100):body.step(.02,np.zeros((6,3)))
            body.energy=1.
        if self.config.ecology_enabled:
            from .ecology import EcologyState
            body.ecology_state=EcologyState(self.config,body)
        if self.physics=='mujoco' and (self.config.vision_enabled or self.config.nociception_enabled or self.config.somatic_enabled):
            from .perception import ExternalSenses
            body.senses=ExternalSenses(self.config);body.senses.bind_model(body.model)
            if self.config.somatic_enabled:body.senses.initialize_tactile(body)
            body.senses.scan(body,self.world)
        return body

    def _ecology_input(self):
        if self.maintenance is not None:
            state=self.body.ecology_state
            self.maintenance.sense(self.body,state,self.body.snapshot(self.world))
            self.prey.sense(self.body,state)

    def _binocular_input(self):
        sense=self.body.senses
        image_key=(self.epoch,getattr(sense,'detail_sequence',sense.sequence))
        if self.binocular.last_image_seq!=image_key:
            self.binocular.set_images(sense.rgb,valid_rectangle=getattr(sense,'fovea',None))
            self.binocular.last_image_seq=image_key

    def _manual_state(self):
        return {"torques": np.round(self.manual_torques, 2).tolist(),
                "contacts": self.contacts.tolist()}

    def initial(self):
        return {"type": "init", "regions": self.names, "cells": len(self.model.group)+(len(self.intent.gates) if self.intent is not None else 0)+(len(self.binocular.group) if self.binocular is not None else 0)+(len(self.maintenance.group)+len(self.prey.group) if self.maintenance is not None else 0),
                "synapses": len(self.model.synapses)+(sum(len(s) for s in self.intent.synapses) if self.intent is not None else 0)+(sum(len(s) for s in self.binocular.synapses) if self.binocular is not None else 0)+(sum(len(s) for s in self.maintenance.synapses)+len(self.prey.synapses) if self.maintenance is not None else 0), "matrix": self.matrix,
                "world": self.world.snapshot(), "legs": LEG_NAMES,
                "joint_names": self.config.joint_names, "body_model": self.config.body_model,
                "viewer_port": self.viewer_port,
                "manual_mode": self.config.manual_test_mode,
                "motor_circuit":self.config.motor_circuit,
                "motor_learning_enabled":self.motor_learning is not None,
                "feeding_enabled":self.config.feeding_enabled,
                "vision_enabled":self.config.vision_enabled,
                "binocular_enabled":self.binocular is not None,
                "optical_flow_enabled":self.config.optical_flow_enabled,
                "ecology_enabled":self.config.ecology_enabled,
                "vision_workers":self.config.vision_workers,
                "somatic_enabled":self.config.somatic_enabled,
                "prey_count":self.config.prey_count if self.config.ecology_enabled else 0,
                "nociception_enabled":self.config.nociception_enabled,
                "plasticity_note": ("Fase 5: R-STDP motor disponível; ligue ou desligue no painel."
                                    if self.config.phase5_enabled else "Fase 4: só homeostase Kenyon.")}

    def latest(self):
        with self.lock:
            return self.frame.copy()

    def _homeostatic_state(self):
        fatigue=getattr(getattr(self.body,'ecology_state',None),'fatigue',0.)
        return dict(enabled=self.config.homeostasis_enabled,
                    reserve_deficit=round(1-self.body.energy,4),fatigue_pressure=round(fatigue**2,4),
                    odor_familiarity=np.round(getattr(self.value_filter,'odor_baseline',np.zeros((4,2))),4).tolist(),
                    odor_change=round(float(getattr(self.value_filter,'odor_change',0.)),4))

    def submit(self, command):
        if not isinstance(command, dict):
            raise ValueError("comando deve ser um objeto")
        name = command.get("cmd")
        value = command.get("value")
        if name == "speed" and value not in (1, 5, 20, "turbo"):
            raise ValueError("velocidade inválida")
        if name == "noise" and (not isinstance(value, (int, float)) or not 0 <= value <= 12):
            raise ValueError("ruído fora do intervalo")
        if name == "ablate" and (command.get("region") not in self.names+(["binocular"] if self.binocular is not None else [])+(['maintenance','prey'] if self.maintenance is not None else []) or not isinstance(value, bool)):
            raise ValueError("ablação inválida")
        if name == "pulse" and (not isinstance(value, (int, float)) or not 0 <= value <= 20):
            raise ValueError("pulso inválido")
        if name in ("pause", "plasticity") and not isinstance(value, bool):
            raise ValueError("valor booleano obrigatório")
        if name=='plasticity' and value and self.motor_learning is None:
            raise ValueError('esta execução não possui plasticidade motora; reinicie sem --phase4 no mk1 modular')
        if name == "motor" and (command.get("leg") not in LEG_NAMES
                                or command.get("joint") not in range(len(self.config.joint_names))
                                or not isinstance(value, (int, float)) or not -1 <= value <= 1):
            raise ValueError("torque manual inválido")
        if name == "contact" and (command.get("leg") not in LEG_NAMES
                                  or not isinstance(value, bool)):
            raise ValueError("contato manual inválido")
        if name == "damage" and (command.get("leg") not in LEG_NAMES
                                 or not isinstance(value, bool)):
            raise ValueError("perna inválida")
        if name == "jaw" and (not isinstance(value, (int, float)) or not 0 <= value <= 1):
            raise ValueError("mandíbula inválida")
        if name == "food" and (value is not None and
                              (not isinstance(value, list) or len(value) != 2 or
                               any(not isinstance(x, (int, float)) or not np.isfinite(x) for x in value) or
                               not 0 <= value[0] <= self.world.width or
                               not 0 <= value[1] <= self.world.height)):
            raise ValueError("posição de comida inválida")
        if name in ("motor", "contact", "jaw") and not self.config.manual_test_mode:
            raise ValueError("controle manual disponível apenas no modo de teste da física")
        if name not in ("pause", "speed", "reset", "plasticity", "ablate", "noise", "pulse",
                        "motor", "contact", "jaw", "damage", "food"):
            raise ValueError("comando desconhecido")
        self.commands.put_nowait(command)

    def start(self):
        if self.metrics_csv:
            self._open_live_metrics()
        self.thread.start()

    def stop(self):
        self.stopped.set()
        self.model.network.stop()
        self.thread.join(timeout=2)

    def _commands(self):
        while True:
            try:
                command = self.commands.get_nowait()
            except queue.Empty:
                return
            name = command["cmd"]
            value = command.get("value")
            if name == "pause":
                self.paused = value
            elif name == "speed":
                self.speed = value
            elif name == "plasticity":
                self.plasticity = value
            elif name == "noise":
                self.noise = float(value)
                self.model.group.noise = self.noise
            elif name == "damage":
                self.damaged_leg = command["leg"] if value else ""
            elif name == "motor":
                self.manual_torques[LEG_NAMES.index(command["leg"]), command["joint"]] = float(value)
            elif name == "contact":
                self.contacts[LEG_NAMES.index(command["leg"])] = value
            elif name == "jaw":
                self.body.jaw_open = float(value)
            elif name == "food":
                self.world = (replace(self.world, food_enabled=False) if value is None else
                              replace(self.world, odor_x=float(value[0]), odor_y=float(value[1]),
                                      food_enabled=True,food_energy=1.))
                if getattr(self.body,'senses',None) is not None and self.config.vision_enabled:
                    self.body.senses.scan(self.body,self.world);self.body.senses.binocular=None
                    if self.binocular is not None:self._binocular_input()
            elif name == "ablate":
                if command['region']=='binocular':self.binocular.group.spike_gate=0 if value else 1
                elif command['region']=='maintenance':self.maintenance.group.spike_gate=0 if value else 1
                elif command['region']=='prey':self.prey.enabled=not value;self.prey.group.spike_gate=0 if value else 1
                else:self.model.group.spike_gate[self.model.regions[command["region"]]] = 0 if value else 1
            elif name == "pulse":
                self.odor_pulse = float(value)
                set_odor_pulse(self.model, self.odor_pulse)
                self.odor_end_ms = self.simulated_ms + 1000 if value else 0
            elif name == "reset":
                if self.config.phase5_enabled:
                    self.body = self._new_body()
                    self.value_filter.reset()
                    self.metrics = Metrics(self.body)
                    if self.window_metrics is not None:
                        self.window_metrics = Metrics(self.body)
                    self.contacts[:] = (self.body.contacts if self.physics == "mujoco" else True)
                    self.epoch += 1
                    self.episode_start_ms=self.simulated_ms
                    count=np.asarray(self.model.monitor.count[:],dtype=np.int64)
                    self.motor=MotorDecoder(self.model,self.config);self.motor.previous=count.copy()
                    if self.mouth is not None:
                        self.mouth=NeuralMouth(self.model);self.mouth.previous=count.copy()
                    if not self.config.manual_test_mode:
                        inject_sensors(self.model,self.body.snapshot(self.world,self.contacts),config=self.config)
                    if self.binocular is not None:
                        self.binocular.reset_readout();self._binocular_input()
                    if self.maintenance is not None:
                        for circuit in (self.maintenance,self.prey):
                            circuit.previous=np.asarray(circuit.monitor.count[:]).copy();circuit.rates[:]=0
                        self._ecology_input()
                    self.paused=True
                    self.model.network.stop()
                    self.spike_cursor = int(self.model.monitor.num_events)
                    self._close_metrics()
                    if self.metrics_csv and not self.headless:
                        self._open_live_metrics()
                else:
                    self.reset_requested = True
                    self.model.network.stop()

    def _publish(self, actual_speed):
        model = self.model
        t_ms = self.simulated_ms
        count = np.asarray(model.monitor.count[:], dtype=np.int64)
        end = int(model.monitor.num_events)
        selection=(self.spike_cursor+np.linspace(0,end-self.spike_cursor-1,8,dtype=np.int32)
                   if end-self.spike_cursor>8 else slice(self.spike_cursor,end))
        times = np.asarray(model.monitor.t[selection] / ms, dtype=np.float32)
        neurons = np.asarray(model.monitor.i[selection], dtype=np.int32)
        spikes = [[round(float(t), 1), int(r)] for t, r in zip(times, self.region_index[neurons])]
        self.spike_cursor = end
        active = np.asarray(model.group.spike_gate[:])
        ablated = [name for name in self.names if not active[model.regions[name]].any()]
        if self.binocular is not None and not np.any(self.binocular.group.spike_gate[:]):ablated.append('binocular')
        if self.maintenance is not None:
            if not np.any(self.maintenance.group.spike_gate[:]):ablated.append('maintenance')
            if not self.prey.enabled:ablated.append('prey')
        with self.lock:
            self.seq += 1
            self.frame = {"type": "frame", "seq": self.seq, "epoch": self.epoch,
                          "t": round(t_ms),
                          "episode_t_ms":round(max(0.,t_ms-self.episode_start_ms)),
                          "rates": self.rates, "spikes": spikes, "paused": self.paused,
                          "speed": self.speed, "actual_speed": round(actual_speed, 2),
                          "plasticity": bool(self.motor_learning and self.plasticity), "ablated": ablated,
                          "noise": self.noise,
                          "body": self.body.snapshot(self.world, self.contacts),
                          "world": self.world.snapshot(),
                          "manual": self._manual_state() if self.config.manual_test_mode else None,
                          "metrics": self.metrics.snapshot(self.body),
                          "learning_curve": self.learning_curve[-120:],
                          "neuromodulation": self.motor_learning.summary() if self.motor_learning else None,
                          "intent": intent_status(self.intent,self.body.energy,self.config),
                          "homeostasis": self._homeostatic_state(),
                          "damaged_leg": self.damaged_leg,
                          "matrix": self.matrix,
                          "motor_rates": self.motor.pool_rates(),
                          "motor_saturation": round(self.motor.saturation_fraction(), 4)}

    def _record_learning_point(self):
        if self.simulated_ms + 1e-6 < self.next_learning_point_ms:
            return
        m = self.metrics.snapshot(self.body)
        point = {"t_s": round(self.simulated_ms / 1000, 2),
                 "net_path": round(m["net_path"], 5),
                 "net_energy": round(m["displacement"] / max(m["energy_spent"], 1e-9), 4),
                 "phase": round(float(np.mean(m["neighbor_phase_correlation"])), 4),
                 "motor_saturation": round(self.motor.saturation_fraction(), 5)}
        self.learning_curve.append(point)
        self.next_learning_point_ms += 10000
        if self.motor_learning and not self.headless:
            weights = np.asarray(self.model.synapses.w[:], dtype=np.float32)
            pre = np.asarray(self.model.synapses.i[:], dtype=np.int32)
            post = np.asarray(self.model.synapses.j[:], dtype=np.int32)
            pair = self.region_index[pre] * len(self.names) + self.region_index[post]
            sums = np.bincount(pair, weights=weights, minlength=len(self.names) ** 2)
            counts = np.bincount(pair, minlength=len(self.names) ** 2)
            self.matrix = np.round(np.divide(sums, counts, out=np.zeros_like(sums),
                                             where=counts > 0).reshape(len(self.names), len(self.names)), 3).tolist()

    def _sample_rates(self, count, t_ms):
        if t_ms - self.rate_time_ms >= 500:
            delta = count - self.rate_count
            seconds = (t_ms - self.rate_time_ms) / 1000
            self.rates = [round(float(delta[self.model.regions[name]].sum() /
                                      len(self.model.regions[name]) / seconds), 1)
                          for name in self.names]
            self.rate_count = count.copy()
            self.rate_time_ms = t_ms
        if t_ms - self.homeo_time_ms >= 5000:
            kc = self.model.regions["kc"]
            rate = float((count[kc] - self.homeo_count[kc]).sum() / len(kc) /
                         ((t_ms - self.homeo_time_ms) / 1000))
            kenyon_homeostasis(self.model, rate)
            self.homeo_count = count.copy()
            self.homeo_time_ms = t_ms

    def _write_metrics(self, t_ms):
        if self.metrics_writer and t_ms >= self.next_csv_ms:
            self.metrics_writer.writerow(Metrics.csv_row(
                self.config.seed, t_ms / 1000, self.metrics.snapshot(self.body),
                self.motor.saturation_fraction(), self.rates))
            self.next_csv_ms += 1000
            if self.metrics_output and int(t_ms) % 5000 == 0:
                self.metrics_output.flush()

    def _open_live_metrics(self):
        destination = Path(self.metrics_csv)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination = destination.with_name(f"{destination.stem}_epoch{self.epoch}{destination.suffix}")
        self.metrics_output = destination.open("w", newline="", encoding="utf-8")
        self.metrics_writer = csv.writer(self.metrics_output)
        self.metrics_writer.writerow(Metrics.csv_columns(self.names, self.config.joint_names))
        self.next_csv_ms = 1000 * (int(self.simulated_ms // 1000) + 1)

    def _close_metrics(self):
        if self.metrics_output:
            self.metrics_output.close()
        self.metrics_output = None
        self.metrics_writer = None

    def _tick(self):
        self._commands()
        now = time.perf_counter()
        elapsed = now - self.last_tick_wall if self.last_tick_wall else 0
        self.last_tick_wall = now
        if self.paused or self.reset_requested or self.stopped.is_set():
            self.model.network.stop()
            return
        self.simulated_ms += self.config.block_ms
        t_ms = self.simulated_ms
        if self.headless and self.metrics_csv and int(t_ms) % 30000 == 0:
            print(f"seed {self.config.seed}: {t_ms / 1000:.0f} s simulados", flush=True)
        if self.odor_end_ms and t_ms >= self.odor_end_ms:
            set_odor_pulse(self.model, 0)
            self.odor_end_ms = 0
            self.odor_pulse = 0.0
        count = np.asarray(self.model.monitor.count[:], dtype=np.int64)
        block_s = self.config.block_ms / 1000
        if self.prey is not None:
            self.body.data.ctrl[self.body.prey_motor_ids.ravel()]=self.prey.decode(block_s).ravel()
        torques = self.manual_torques if self.config.manual_test_mode else self.motor.update(count, block_s)
        if self.damaged_leg:
            torques[LEG_NAMES.index(self.damaged_leg)] = 0
        if not self.config.manual_test_mode and self.physics == "legacy":
            self.contacts = self.body.angles[:, 1] < self.config.contact_threshold
            if self.damaged_leg:
                self.contacts[LEG_NAMES.index(self.damaged_leg)] = False
        # Viewer runs hold decoded torque across 2 ms steps; legacy headless
        # experiments retain their configured integration for comparability.
        # MuJoCoBody already integrates its configured 2 ms substeps with the
        # held command. The modular circuit averages load samples at each physical
        # substep, then updates sensory currents once per neural block.
        substeps = (1 if self.physics == 'mujoco' else
                    max(1, math.ceil(self.config.block_ms / self.body_substep_ms)))
        touched = False
        for _ in range(substeps):
            self.body.step(block_s / substeps, torques, self.contacts, self.world)
            touched |= self.body.touch
        self.body.touch = touched
        if self.mouth is not None:
            from .perception import feed
            closing,opening=self.mouth.update(count,block_s)
            if self.prey is not None:
                self.world,taken=self.body.ecology_state.feed_all(self.body,self.world,closing,block_s)
            else:self.world,taken=feed(self.body,self.world,closing,block_s,self.config.feeding_rate)
            self.body.food_intake_rate=taken/block_s
            self.body.food_consumed+=taken
            self.body.jaw_open=float(np.clip(self.body.jaw_open+2*block_s*(opening-closing),0,1))
        if self.physics == "mujoco":
            self.contacts = self.body.contacts.copy()
            self.falls_seen |= self.body.fell
            if self.damaged_leg:
                self.contacts[LEG_NAMES.index(self.damaged_leg)] = False
        contact_samples=(self.body.foot_load_samples>.02) if self.config.motor_circuit=='modular' else None
        self.metrics.update(self.body, torques, self.contacts,contact_samples=contact_samples)
        if self.window_metrics is not None:
            self.window_metrics.update(self.body, torques, self.contacts,contact_samples=contact_samples)
        if self.body.energy <= 0:
            old_body = self.body
            self.body = self._new_body()
            self.metrics.episode_reset(old_body, self.body)
            self.episode_start_ms=self.simulated_ms
            self.epoch+=1
            if self.window_metrics is not None:
                self.window_metrics.episode_reset(old_body, self.body)
            self.contacts[:] = (self.body.contacts if self.physics == "mujoco" else True)
            self.value_filter.reset()
        if self.window_metrics is not None and int(t_ms) in (900000, 2400000):
            self.window_metrics = Metrics(self.body)
        if self.binocular is not None:
            self.body.senses.binocular=self.binocular.read(block_s)
            self._binocular_input()
        if self.maintenance is not None:
            self.body.ecology_state.maintenance_rates=self.maintenance.read(block_s)
            self.body.ecology_state.prey_rates=self.prey.mean_hz.copy()
            self._ecology_input()
        if not self.config.manual_test_mode:
            sample = self.body.snapshot(self.world, self.contacts)
            filtered_flow = (self.value_filter.update(sample["flow"], block_s)
                             if self.config.phase5_enabled else None)
            odor_modulation=None
            if self.config.homeostasis_enabled:
                odor_modulation=self.value_filter.adapt_odor(sample,block_s,self.config)
                sample['odor_novelty']=self.value_filter.odor_change
            inject_sensors(self.model, sample, config=self.config,
                           efference=torques, value_flow=filtered_flow,
                           odor_modulation=odor_modulation)
            if self.odor_end_ms:
                channels = np.array_split(self.model.regions["al_glom"], 4)[0]
                self.model.group.I_odor[channels] += self.odor_pulse
        if self.motor_learning:
            self.motor_learning.update(count, block_s, enabled=self.plasticity)
            fatigue=getattr(getattr(self.body,'ecology_state',None),'fatigue',0.)
            arousal=1-self.config.fatigue_drive_suppression*fatigue**2 if self.config.homeostasis_enabled else 1.
            self.model.group.noise = self.noise * float(np.clip(
                arousal*(0.5 + self.config.octopamine_gain * self.motor_learning.oct_fast / 15.0),
                .1 if self.config.homeostasis_enabled else .4, 2.5))
        if self.intent is not None:
            from .intent import update_rates
            update_rates(self.intent,count,block_s)
        self._sample_rates(count, t_ms)
        self._record_learning_point()
        self._write_metrics(t_ms)
        if not self.headless and (self.speed != "turbo" or now - getattr(self, "last_publish", 0) >= 1):
            self._publish(block_s / elapsed if elapsed else 0)
            self.last_publish = now
        if not self.paused and self.speed != "turbo" and elapsed:
            self.stopped.wait(max(0.0, block_s / self.speed - elapsed))

    def run_headless(self, duration_s):
        if not self.headless:
            raise ValueError("run_headless requer headless=True")
        self.speed = "turbo"
        self.metrics_writer = None
        self.next_csv_ms = 1000
        output = None
        if self.metrics_csv:
            output = open(self.metrics_csv, "w", newline="", encoding="utf-8")
            self.metrics_output = output
            self.metrics_writer = csv.writer(output)
            self.metrics_writer.writerow(Metrics.csv_columns(self.names, self.config.joint_names))
        try:
            self.model.network.run(duration_s * 1000 * ms)
        finally:
            if output:
                output.close()
        return {**self.metrics.snapshot(self.body), "rates": dict(zip(self.names, self.rates)),
                "motor_saturation": round(self.motor.saturation_fraction(), 5),
                "motor_rates": self.motor.pool_rates()}

    def save_checkpoint(self, path):
        """Salva pesos, estado neural/físico e relógios em um arquivo local."""
        # Brian can leave t_ as a clock VariableView after Network.stop().
        self.model.network.t_=float(np.asarray(self.model.network.t_))
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        neural_names = ("v", "u", "I_syn", "I_base", "I_noise", "I_odor", "I_sense",
                        "noise", "noise_gain", "spike_gate")
        neural = {name: np.array(self.model.group.variables[name].get_value(), copy=True)
                  for name in neural_names}
        clocks = {name: (np.array(clock.variables["t"].get_value(), copy=True),
                         np.array(clock.variables["timestep"].get_value(), copy=True))
                  for name, clock in (("neural", self.model.group.clock),
                                      ("body", next(obj.clock for obj in self.model.network.objects
                                                    if isinstance(obj, NetworkOperation))))}
        state = {
            "world":self.world,
            "prey_enabled":self.prey.enabled if self.prey is not None else None,
            "ecology_circuit":{label:{key:getattr(circuit,key).copy() for key in keys}
                for label,circuit,keys in [('maintenance',self.maintenance,('previous','rates')),('prey',self.prey,('previous','rates','mean_hz'))]} if self.maintenance is not None else None,
            "binocular_state":{k:getattr(self.binocular,k).copy() if isinstance(getattr(self.binocular,k),np.ndarray) else getattr(self.binocular,k) for k in ('previous','rates','last_image_seq','sequence')} if self.binocular is not None else None,
            "intent_state":{k:getattr(self.intent,k).copy() for k in ('last_core','last_gate','rates')} if self.intent is not None else None,
            "mouth":self.mouth.__dict__.copy() if self.mouth is not None else None,
            "config": self.config,
            "seed": self.config.seed, "scale": self.config.scale,
            "phase5_enabled": self.config.phase5_enabled,
            "reward_mode": self.reward_mode, "body": self.body.__dict__,
            "neural": neural, "weights": np.array(self.model.synapses.variables["w"].get_value(), copy=True),
            "monitor_count": np.array(self.model.monitor.variables["count"].get_value(), copy=True),
            "clocks": clocks, "network_t": self.model.network.t_,
            "numpy_random": np.random.get_state(),
            "motor": {key: value for key, value in self.motor.__dict__.items()
                      if key not in ("model", "config") and not callable(value)},
            "metrics": self.metrics.__dict__,
            "window_metrics": (self.window_metrics.__dict__
                               if self.window_metrics is not None else None),
            "value_flow_state": self.value_filter.state.copy(),
            "value_flow_extra": {key:value for key,value in self.value_filter.__dict__.items()
                                 if key != 'state'},
            "learning": ({key: value for key, value in self.motor_learning.__dict__.items()
                          if key not in ("model", "config")} if self.motor_learning else None),
            "sim": {key: getattr(self, key) for key in (
                "contacts", "simulated_ms", "episode_start_ms", "epoch", "rate_count", "homeo_count",
                "rate_time_ms", "homeo_time_ms", "rates", "plasticity",
                "noise", "damaged_leg", "learning_curve")},
        }
        from .checkpoint_rng import capture
        state['runtime_rng_values']=capture()
        # Brian's public store includes pending delayed spikes, all additional
        # membrane currents and the Cython RNG buffers. The final checkpoint
        # still becomes visible only through the atomic replace below.
        with tempfile.TemporaryDirectory(prefix='veguinha-network-state-') as directory:
            brian_path=Path(directory)/'network.pkl'
            self.model.network.store(name='resume',filename=str(brian_path))
            state['brian_network']=brian_path.read_bytes()
        temporary = destination.with_suffix(".pkl.tmp")
        with temporary.open("wb") as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination.with_suffix(".pkl"))

    def load_checkpoint(self, path):
        """Carrega checkpoint local confiável construído com a mesma configuração."""
        source = Path(path)
        with source.with_suffix(".pkl").open("rb") as handle:
            state = pickle.load(handle)
        if (state["seed"], state["scale"], state["phase5_enabled"], state["reward_mode"]) != (
                self.config.seed, self.config.scale, self.config.phase5_enabled, self.reward_mode):
            raise ValueError("configuração incompatível com checkpoint")
        if "config" in state and replace(state["config"],homeostasis_enabled=self.config.homeostasis_enabled) != self.config:
            raise ValueError("parâmetros incompatíveis com checkpoint")
        if (len(state["weights"]) != len(self.model.synapses)
                or len(state["neural"]["v"]) != len(self.model.group)
                or (self.motor_learning and len(state["learning"]["weight"])
                    != len(self.motor_learning.weight))):
            raise ValueError("topologia incompatível com checkpoint")
        for name, values in state["neural"].items():
            self.model.group.variables[name].get_value()[:] = values
        self.model.synapses.variables["w"].get_value()[:] = state["weights"]
        self.model.monitor.variables["count"].get_value()[:] = state["monitor_count"]
        for name, clock in (("neural", self.model.group.clock),
                            ("body", next(obj.clock for obj in self.model.network.objects
                                          if isinstance(obj, NetworkOperation)))):
            t, step = state["clocks"][name]
            clock.variables["t"].get_value()[:] = t
            clock.variables["timestep"].get_value()[:] = step
        self.model.network.t_ = state["network_t"]
        np.random.set_state(state["numpy_random"])
        loaded_body=state['body']
        scene_changed=False
        if self.config.ecology_enabled:
            from .ecology import migrate_body
            scene_changed=loaded_body.get('prey_count',0)!=self.body.prey_count
            if loaded_body.get('prey_count',0):
                old_model=loaded_body['model']
                scene_changed|=float(old_model.geom_size[old_model.geom('prey0_collision0').id,0])!=self.config.prey_radius_m
            loaded_body=migrate_body(loaded_body,self.body,self.config)
        self.body.__dict__.update(loaded_body)
        self.world=state.get('world',self.world)
        if getattr(self.body,'senses',None) is not None:
            old=self.body.senses
            calibration=('vision_enabled','binocular_enabled','foveated_enabled','vision_width','vision_height','vision_fov_deg','vision_pitch_deg','vision_baseline_m')
            rebuild=(old.__dict__.get('format_version')!=(5 if self.config.foveated_enabled else 4) or old.__dict__.get('projection')!=('rectilinear' if self.config.binocular_enabled else 'panoramic') or any(getattr(old.config,k)!=getattr(self.config,k) for k in calibration))
            if rebuild:
                from .perception import ExternalSenses
                self.body.senses=ExternalSenses(self.config);self.body.senses.pain=old.pain.copy()
            self.body.senses.config=self.config
            self.body.senses.bind_model(self.body.model)
            if self.config.somatic_enabled and not hasattr(self.body.senses,'tactile'):
                self.body.senses.initialize_tactile(self.body)
            if rebuild or scene_changed or (self.config.foveated_enabled and not hasattr(self.body.senses,'detail_patch_rgb')):
                self.body.senses.scan(self.body,self.world)
            if self.config.optical_flow_enabled and getattr(self.body.senses,'optical',None) is None:
                from .optical_flow import OpticalFlow
                self.body.senses.optical=OpticalFlow()
                self.body.senses.optical.update(self.body.senses.rgb.astype(float).mean(3)/255,None,self.body.senses.focal_px)
        if self.mouth is not None:
            if state.get('mouth') is not None:self.mouth.__dict__.update(state['mouth'])
            else:self.mouth.previous=np.asarray(state['monitor_count'],dtype=np.int64).copy()
        self.motor.__dict__.update(state["motor"])
        self.metrics.__dict__.update(state["metrics"])
        if state.get("window_metrics") is not None:
            self.window_metrics = Metrics(self.body)
            self.window_metrics.__dict__.update(state["window_metrics"])
        elif self.window_metrics is not None:
            # Checkpoints antigos podem terminar exatamente no início da janela.
            self.window_metrics = Metrics(self.body)
        self.value_filter.state[:] = state.get("value_flow_state", np.zeros(3))
        self.value_filter.__dict__.update(state.get('value_flow_extra',{}))
        for key in ("forward_flow_sum", "lateral_flow_abs_sum", "rotation_abs_sum",
                    "wall_collisions"):
            self.metrics.__dict__.setdefault(key, 0.0 if key != "wall_collisions" else 0)
        self.metrics.__dict__.setdefault("contact_sum", np.zeros(6))
        self.metrics.__dict__.setdefault("contact_pair_sum", np.zeros(7))
        if self.motor_learning:
            self.motor_learning.__dict__.update(state["learning"])
        for key, value in state["sim"].items():
            setattr(self, key, value)
        if 'brian_network' in state:
            with tempfile.TemporaryDirectory(prefix='veguinha-network-state-') as directory:
                brian_path=Path(directory)/'network.pkl'
                brian_path.write_bytes(state['brian_network'])
                # Brian2 2.10 serializes Cython buffer pointers, invalid in another process.
                stored_names=set(pickle.loads(state['brian_network'])['resume'])
                new_intent=self.intent is not None and 'permission_cells' not in stored_names
                new_binocular=self.binocular is not None and 'binocular_cells' not in stored_names
                new_ecology=self.maintenance is not None and 'ecology_cells' not in stored_names
                missing=(list(self.intent.objects) if new_intent else [])+(list(self.binocular.objects) if new_binocular else [])
                brake=getattr(self.intent,'somatic_brake',None)
                if not new_intent and brake is not None and brake.name not in stored_names:missing.append(brake)
                if new_ecology:missing+=list(self.maintenance.objects)+list(self.prey.objects)
                if missing:self.model.network.remove(*missing)
                try:self.model.network.restore(name='resume',filename=str(brian_path),restore_random_state=False)
                finally:
                    if missing:self.model.network.add(*missing)
                if new_intent:
                    from .intent import prepare_intent
                    prepare_intent(self.model,self.intent)
        if self.intent is not None:
            if state.get('intent_state'):
                for k,v in state['intent_state'].items():setattr(self.intent,k,v.copy())
            else:self.intent.last_core=np.asarray(state['monitor_count'],dtype=np.int64).copy()
        if self.binocular is not None:
            if state.get('binocular_state'):
                for k,v in state['binocular_state'].items():setattr(self.binocular,k,v.copy() if isinstance(v,np.ndarray) else v)
            else:self.binocular.reset_readout()
            self.binocular.last_image_seq=None;self._binocular_input()
        if self.maintenance is not None:
            self.body.ecology_state.config=self.config
            self.prey.enabled=state.get('prey_enabled') is not False
            for label,circuit in [('maintenance',self.maintenance),('prey',self.prey)]:
                for key,value in (state.get('ecology_circuit') or {}).get(label,{}).items():setattr(circuit,key,value.copy())
            self._ecology_input()
        from .checkpoint_rng import restore
        restore(state.get('runtime_rng_values'))
        self.next_learning_point_ms = 10000 * (int(self.simulated_ms // 10000) + 1)
        if 'episode_start_ms' not in state['sim']:
            elapsed=max(0.,getattr(getattr(self.body,'data',None),'time',0.)-(2. if self.config.motor_circuit=='modular' else 0.))*1000
            self.episode_start_ms=max(0.,self.simulated_ms-elapsed)

    def _loop(self):
        try:
            while not self.stopped.is_set():
                self._commands()
                if self.reset_requested:
                    self._close_metrics()
                    self._build()
                    if self.metrics_csv:
                        self._open_live_metrics()
                    self.reset_requested = False
                    self.last_tick_wall = 0
                if self.paused:
                    self._publish(0)
                    self.stopped.wait(0.1)
                    continue
                self.model.network.run(1000 * ms)
        finally:
            self._close_metrics()
