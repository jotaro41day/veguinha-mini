"""Free-body 3D hexapod driven only by joint torques and MuJoCo contacts."""

import math
from pathlib import Path

import mujoco
import numpy as np

from .body import LEG_NAMES


MJCF_PATH = Path(__file__).with_name("assets") / "hexapod.xml"
JOINT_NAMES = tuple(f"{leg}_{joint}" for leg in LEG_NAMES for joint in ("hip", "knee"))


class MujocoBody:
    """MuJoCo owns the root pose, velocities, joint state, gravity and contact."""

    def __init__(self, world=None, dt=0.002, variant="control", sample_contacts=False,prey_count=0,prey_radius_m=.025):
        if dt not in (0.001, 0.002, 0.004):
            raise ValueError("passo MuJoCo deve ser 1, 2 ou 4 ms")
        if variant not in ("control", "mk1"):
            raise ValueError("corpo desconhecido")
        self.variant = variant
        self.mjcf_path = (MJCF_PATH.with_name("hexapod_mk1.xml") if variant == "mk1" else MJCF_PATH)
        self.joint_names = ("coxa", "femur", "tibia") if variant == "mk1" else ("hip", "knee")
        self.njoints = len(self.joint_names)
        from .ecology import load_model
        self.prey_count=prey_count
        self.prey_radius_m=prey_radius_m
        self.ecology_state=None
        self.model = load_model(self.mjcf_path,prey_count,prey_radius_m)
        self.model.opt.timestep = dt
        self.data = mujoco.MjData(self.model)
        self.motor_ids=np.arange(6*self.njoints)
        self.prey_motor_ids=np.array([[self.model.actuator(f'prey{k}_motor{j}').id for j in range(3)] for k in range(prey_count)],int).reshape(prey_count,3)
        self.prey_geom_ids={i for i in range(self.model.ngeom) if self.model.geom(i).name.startswith('prey') and self.model.geom_contype[i]}
        self.body_id = self.model.body("thorax").id
        self.floor_id = self.model.geom("floor").id
        self.body_contact_ids = {self.model.geom(name).id for name in ("shell", "head")}
        if variant == 'mk1':
            self.body_contact_ids.add(self.model.geom('tail').id)
        self.foot_ids = np.array([self.model.geom(f"{leg}_foot").id for leg in LEG_NAMES])
        self.foot_index = {int(geom): index for index, geom in enumerate(self.foot_ids)}
        self.wall_ids = {self.model.geom(f"wall_{side}").id
                         for side in ("front", "back", "left", "right")}
        names = [f"{leg}_{j}" for leg in LEG_NAMES for j in self.joint_names]
        self.joint_qpos = np.array([self.model.joint(name).qposadr for name in names])
        self.joint_qvel = np.array([self.model.joint(name).dofadr for name in names])
        self.hip_ids = np.array([self.model.body(f"{leg}_{'coxa' if variant == 'mk1' else 'femur'}").id for leg in LEG_NAMES])
        self.knee_ids = np.array([self.model.body(f"{leg}_tibia").id for leg in LEG_NAMES])
        self.toe_ids = np.array([self.model.site(f"{leg}_toe").id for leg in LEG_NAMES])
        self.energy = 1.0
        self.energy_cost_rate = 0.0
        self.jaw_open = 0.0
        self.food_intake_rate=0.
        self.food_consumed=0.
        self.senses=None
        self.touch = False
        self.contacts = np.zeros(6, dtype=bool)
        self.normal_load = np.zeros(6)
        self.sample_contacts=sample_contacts
        hex_mass=sum(self.model.body_mass[i] for i in range(self.model.nbody) if not self.model.body(i).name.startswith('prey'))
        self.foot_weight=float(hex_mass*abs(self.model.opt.gravity[2])/6)
        self.mean_normal_load=np.zeros(6)
        self.contact_fraction=np.zeros(6)
        self.load_signal=np.zeros(6)
        self.foot_load_samples=np.zeros((1,6))
        self.fell = False
        mujoco.mj_forward(self.model, self.data)
        self._update_contacts()

    @property
    def position(self):
        return self.data.qpos[:2].copy()

    @property
    def velocity(self):
        return self.data.qvel[:2].copy()

    @property
    def heading(self):
        axis = self.data.xmat[self.body_id].reshape(3, 3)[:, 0]
        return float(math.atan2(axis[1], axis[0]))

    @property
    def angular_velocity(self):
        return float(self.data.qvel[5])

    @property
    def angles(self):
        return self.data.qpos[self.joint_qpos].reshape(6, self.njoints).copy()

    @property
    def joint_velocity(self):
        return self.data.qvel[self.joint_qvel].reshape(6, self.njoints).copy()

    @property
    def upright(self):
        return float(self.data.xmat[self.body_id].reshape(3, 3)[2, 2])

    def _update_contacts(self):
        self.contacts[:] = False
        self.normal_load[:] = 0
        self.touch = False
        force = np.zeros(6)
        body_on_floor = False
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 in self.wall_ids or geom2 in self.wall_ids:
                self.touch = True
            if ((geom1 == self.floor_id and geom2 in self.body_contact_ids)
                    or (geom2 == self.floor_id and geom1 in self.body_contact_ids)):
                body_on_floor = True
            if geom1 == self.floor_id:
                foot = self.foot_index.get(geom2)
            elif geom2 == self.floor_id:
                foot = self.foot_index.get(geom1)
            elif geom1 in self.prey_geom_ids:
                foot=self.foot_index.get(geom2)
            elif geom2 in self.prey_geom_ids:
                foot=self.foot_index.get(geom1)
            else:
                continue
            if foot is not None:
                mujoco.mj_contactForce(self.model, self.data, index, force)
                self.contacts[foot] = True
                self.normal_load[foot] += max(0.0, force[0])
        self.fell = body_on_floor or self.data.qpos[2] < .12 or self.upright < .5

    def step(self, dt, torques, contacts=None, world=None, observe=None):
        """Apply constant normalized torques for an integer number of physics steps."""
        n = round(dt / self.model.opt.timestep)
        if n < 1 or abs(n * self.model.opt.timestep - dt) > 1e-10:
            raise ValueError("bloco não é múltiplo do passo MuJoCo")
        command = np.asarray(torques, dtype=float)
        if command.shape != (6, self.njoints):
            raise ValueError(f"torques devem ter forma (6, {self.njoints})")
        self.data.ctrl[self.motor_ids] = np.clip(command, -1, 1).ravel()
        touched = False
        if self.senses is not None:self.senses.begin_block()
        if self.sample_contacts:
            load_sum=np.zeros(6);contact_sum=np.zeros(6);signal_sum=np.zeros(6)
            load_samples=[]
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            if self.ecology_state is not None:self.ecology_state.observe(self,self.model.opt.timestep)
            if self.senses is not None:self.senses.observe(self,self.model.opt.timestep)
            if self.sample_contacts:
                loads=self._instant_foot_loads()
                load_samples.append(loads)
                load_sum+=loads;contact_sum+=loads>.02
                signal_sum+=np.clip(loads/self.foot_weight,0,1)
            if observe is not None:
                observe(self)
            if n > 1:
                touched |= any(int(c.geom1) in self.wall_ids or int(c.geom2) in self.wall_ids
                               for c in self.data.contact)
        # mj_step's cached Cartesian poses describe the state before its final
        # integration. Refresh them before contacts, sensors, and the viewer read.
        mujoco.mj_forward(self.model, self.data)
        self._update_contacts()
        if self.senses is not None:self.senses.scan(self,world,dt=dt)
        if self.sample_contacts:
            self.mean_normal_load=load_sum/n
            self.contact_fraction=contact_sum/n
            self.load_signal=signal_sum/n
            self.foot_load_samples=np.asarray(load_samples)
        else:
            self.mean_normal_load=self.normal_load.copy()
            self.contact_fraction=self.contacts.astype(float)
            self.load_signal=np.clip(self.normal_load/self.foot_weight,0,1)
            self.foot_load_samples=self.normal_load[None].copy()
        self.touch |= touched
        effort = float(np.sum(self.data.ctrl[self.motor_ids] ** 2))
        self.energy_cost_rate = .0004 + .002 * effort
        self.energy = max(0.0, self.energy - dt * self.energy_cost_rate)
        if self.ecology_state is not None:self.ecology_state.update(self,dt)

    def _instant_foot_loads(self):
        loads=np.zeros(6);force=np.zeros(6)
        for i,c in enumerate(self.data.contact):
            a,b=int(c.geom1),int(c.geom2)
            foot=self.foot_index.get(b if a==self.floor_id or a in self.prey_geom_ids else a if b==self.floor_id or b in self.prey_geom_ids else -1)
            if foot is not None:
                mujoco.mj_contactForce(self.model,self.data,i,force)
                loads[foot]+=max(0.,force[0])
        return loads

    @staticmethod
    def _world_pixels(points):
        points = np.asarray(points)
        return np.column_stack(((points[:, 0] + 3) * 100,
                                (2 - points[:, 1]) * 100))

    def snapshot(self, world=None, contacts=None):
        """Sensory/dashboard data; mj_state is copied for the independent viewer."""
        from .perception import mouth_position,mouth_contact
        matrix = self.data.xmat[self.body_id].reshape(3, 3)
        center = self._world_pixels(self.data.qpos[None, :2])[0]
        hips = self._world_pixels(self.data.xpos[self.hip_ids])
        knees = self._world_pixels(self.data.xpos[self.knee_ids])
        feet_3d = self.data.site_xpos[self.toe_ids]
        feet = self._world_pixels(feet_3d)
        antenna_3d = self.data.xpos[self.body_id] + np.array([matrix @ [0.50, .18, .08],
                                                               matrix @ [0.50, -.18, .08]])
        antennae = self._world_pixels(antenna_3d)
        forward = matrix[:, 0]
        lateral = matrix[:, 1]
        speed = self.data.qvel[:3]
        odor = world.odor_at(antennae) if world is not None else np.zeros(2)
        ecology=getattr(self,'ecology_state',None)
        prey_odor=ecology.odor_at(self,antenna_3d[:,:2]) if ecology is not None else np.zeros(2)
        oral=mouth_contact(self,world)
        if ecology is not None:oral=max(oral,float(ecology.mouth_contact(self).max()))
        from .somatic import taste_sample
        taste=taste_sample(self,world,self.senses.config) if self.senses is not None and self.senses.config.somatic_enabled else np.zeros((2,3))
        return {
            **(self.senses.snapshot() if self.senses is not None else {}),
            "center": np.round(center, 2).tolist(), "heading": round(self.heading, 4),
            "hips": np.round(hips, 2).tolist(), "knees": np.round(knees, 2).tolist(),
            "feet": np.round(feet, 2).tolist(),
            "antennae": np.round(antennae, 2).tolist(),
            "mouth": np.round(self._world_pixels(mouth_position(self)[None])[0],2).tolist(),
            "mouth_contact":oral,
            "taste":taste.tolist(),
            "ecology":ecology.snapshot(self) if ecology is not None else None,
            "prey_odor":prey_odor.tolist(),
            "food_intake_rate":self.food_intake_rate,"food_consumed":self.food_consumed,
            "angles": np.round(self.angles, 4).tolist(),
            "contacts": self.contacts.tolist(), "antenna_odor": np.round(odor, 3).tolist(),
            "contact_fraction":self.contact_fraction.tolist(),
            "load_signal":self.load_signal.tolist(),
            "mean_normal_load":self.mean_normal_load.tolist(),
            "touch": self.touch, "energy": round(self.energy, 4),
            "energy_cost_rate": self.energy_cost_rate, "jaw": round(self.jaw_open, 2),
            "speed": round(float(np.linalg.norm(speed[:2])), 4),
            "flow": [float(np.dot(speed, forward)), float(np.dot(speed, lateral)),
                     self.angular_velocity],
            "height_m": round(float(self.data.qpos[2]), 4),
            "upright": round(self.upright, 4), "fell": self.fell,
            "normal_load": np.round(self.normal_load, 3).tolist(),
            "mj_state": {"qpos": np.round(self.data.qpos, 8).tolist(),
                         "qvel": np.round(self.data.qvel, 8).tolist()},
        }
