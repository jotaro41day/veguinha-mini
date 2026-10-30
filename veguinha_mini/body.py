"""Hexápode plano passivo, acionado por torques externos aos cálculos físicos."""

import math

import numpy as np


LEG_NAMES = ("lf", "lm", "lh", "rf", "rm", "rh")
ANCHORS = np.array([[17, -12], [0, -14], [-17, -12],
                    [17, 12], [0, 14], [-17, 12]], dtype=float)
BASE_ANGLES = np.array([-0.85, -math.pi / 2, -2.2,
                         0.85, math.pi / 2, 2.2], dtype=float)
SEGMENTS = (29.0, 24.0)
# Joint angles are anatomical: flexion has the same local sign on both sides.
# Global rotation is mirrored on the right side of the body.
JOINT_SIDE = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
JOINT_INERTIA = np.array([1200.0, 600.0])


def rotate(points, heading):
    cosine, sine = math.cos(heading), math.sin(heading)
    matrix = np.array([[cosine, -sine], [sine, cosine]])
    return np.asarray(points) @ matrix.T


def perpendicular(points):
    points = np.asarray(points)
    return np.stack((-points[..., 1], points[..., 0]), axis=-1)


class Body:
    radius = 22.0
    # Reduced 2.5D support: knee flexion lifts a compliant foot from a fixed
    # body-height plane. This is not a vertical balance / falling model.
    foot_lift_radius = 24.0
    normal_stiffness = 50.0
    normal_damping = 1.5
    friction_coefficient = 2.0
    mass = 3.0
    gravity = 9.81
    nominal_foot_drop = 24.0

    def __init__(self, world):
        self.position = np.array([world.width * 0.30, world.height * 0.50])
        self.velocity = np.zeros(2)
        self.heading = 0.0
        self.angular_velocity = 0.0
        self.rest_angles = np.zeros((6, 2))
        self.angles = self.rest_angles.copy()
        self.joint_velocity = np.zeros((6, 2))
        self.energy = 1.0
        self.energy_cost_rate = 0.0
        self.jaw_open = 0.0
        self.touch = False
        self.ground_forces = np.zeros((6, 2))
        self.foot_height = np.zeros(6)
        self.normal_load = np.zeros(6)
        self.contacts = np.zeros(6, dtype=bool)
        self.vertical_support = False
        self.height = self.nominal_foot_drop - self.mass*self.gravity/(6*self.normal_stiffness)
        self.vertical_velocity = 0.0

    def support_state(self):
        """Unilateral spring/damper; never adhesion above the ground plane."""
        height = self.foot_lift_radius * np.sin(self.angles[:, 1])
        vertical_velocity = (self.foot_lift_radius * np.cos(self.angles[:, 1])
                             * self.joint_velocity[:, 1])
        if self.vertical_support:
            height += self.height - self.nominal_foot_drop
            vertical_velocity += self.vertical_velocity
        load = np.where(height <= 0, np.maximum(0., -self.normal_stiffness*height
                        - self.normal_damping*vertical_velocity), 0.)
        return height, load

    def geometry(self):
        first = BASE_ANGLES + JOINT_SIDE * self.angles[:, 0]
        second = first + JOINT_SIDE * self.angles[:, 1]
        first_vector = SEGMENTS[0] * np.column_stack((np.cos(first), np.sin(first)))
        second_vector = SEGMENTS[1] * np.column_stack((np.cos(second), np.sin(second)))
        hips_local = ANCHORS
        knees_local = hips_local + first_vector
        feet_local = knees_local + second_vector
        hips = self.position + rotate(hips_local, self.heading)
        knees = self.position + rotate(knees_local, self.heading)
        feet = self.position + rotate(feet_local, self.heading)
        axis = rotate(second_vector / SEGMENTS[1], self.heading)
        hip_jac = rotate(JOINT_SIDE[:, None] * perpendicular(first_vector + second_vector),
                         self.heading)
        knee_jac = rotate(JOINT_SIDE[:, None] * perpendicular(second_vector), self.heading)
        return hips, knees, feet, axis, hip_jac, knee_jac

    def step(self, dt, torques, contacts, world):
        if not 0 < dt <= 0.05:
            raise ValueError("passo físico fora do intervalo")
        if dt > .02:
            substeps = math.ceil(dt / .02)
            touched = False
            for _ in range(substeps):
                self.step(dt / substeps, torques, contacts, world)
                touched |= self.touch
            self.touch = touched
            return
        torques = np.clip(np.asarray(torques, dtype=float), -1, 1)
        # None: physical height/load contact. Explicit masks are retained only
        # for legacy reproduction and the manual body test harness.
        automatic_contact = contacts is None
        if not automatic_contact:
            contacts = np.asarray(contacts, dtype=bool)
        if torques.shape != (6, 2) or (not automatic_contact and contacts.shape != (6,)):
            raise ValueError("torques ou contatos inválidos")
        acceleration = 4.0 * torques - 2.0 * (self.angles - self.rest_angles) - 1.8 * self.joint_velocity
        self.joint_velocity += acceleration * dt
        self.angles += self.joint_velocity * dt
        self.angles[:, 0] = np.clip(self.angles[:, 0], -1.2, 1.2)
        self.angles[:, 1] = np.clip(self.angles[:, 1], -1.5, 1.5)
        at_joint_limit = np.abs(self.angles) >= np.array([1.2, 1.5])
        # An inelastic stop removes outward velocity before computing foot slip.
        self.joint_velocity[at_joint_limit & (self.angles * self.joint_velocity > 0)] = 0
        if automatic_contact:
            self.foot_height[:], self.normal_load[:] = self.support_state()
            contacts = self.normal_load > 0
        self.contacts[:] = contacts
        hips, knees, feet, axis, hip_jac, knee_jac = self.geometry()
        relative = feet - self.position
        foot_velocity = (self.velocity + self.angular_velocity * perpendicular(relative)
                         + hip_jac * self.joint_velocity[:, :1]
                         + knee_jac * self.joint_velocity[:, 1:])
        lateral = perpendicular(axis)
        longitudinal_speed = np.sum(foot_velocity * axis, axis=1)
        lateral_speed = np.sum(foot_velocity * lateral, axis=1)
        forces = -(1.4 * longitudinal_speed[:, None] * axis
                   + 4.0 * lateral_speed[:, None] * lateral)
        magnitude = np.linalg.norm(forces, axis=1)
        force_limit = self.friction_coefficient*self.normal_load if automatic_contact else 65.0
        forces *= np.minimum(1.0, force_limit / np.maximum(magnitude, 1e-9))[:, None]
        forces[~contacts] = 0
        self.ground_forces[:] = forces
        joint_load = np.column_stack((np.sum(forces * hip_jac, axis=1),
                                      np.sum(forces * knee_jac, axis=1)))
        if automatic_contact:
            joint_load[:, 1] += (self.normal_load * self.foot_lift_radius
                                * np.cos(self.angles[:, 1]))
        self.joint_velocity += dt * joint_load / JOINT_INERTIA
        self.joint_velocity[at_joint_limit & (self.angles * self.joint_velocity > 0)] = 0
        total_force = forces.sum(axis=0)
        torque_body = np.sum(relative[:, 0] * forces[:, 1] - relative[:, 1] * forces[:, 0])
        self.velocity += dt * (total_force / 3.0 - 2.0 * self.velocity)
        self.angular_velocity += dt * (torque_body / 4500.0 - 2.5 * self.angular_velocity)
        if automatic_contact and self.vertical_support:
            self.vertical_velocity += dt * (self.normal_load.sum()/self.mass - self.gravity
                                             - 1.8*self.vertical_velocity)
            self.height += dt*self.vertical_velocity
        self.position += self.velocity * dt
        self.heading = (self.heading + self.angular_velocity * dt + math.pi) % (2 * math.pi) - math.pi
        self.touch = False
        for axis_index, limit in enumerate((world.width, world.height)):
            low, high = self.radius, limit - self.radius
            if self.position[axis_index] < low:
                self.position[axis_index] = low
                self.velocity[axis_index] = max(0.0, -0.2 * self.velocity[axis_index])
                self.touch = True
            elif self.position[axis_index] > high:
                self.position[axis_index] = high
                self.velocity[axis_index] = min(0.0, -0.2 * self.velocity[axis_index])
                self.touch = True
        effort = np.sum(torques ** 2) + 0.2 * np.sum(np.abs(torques * self.joint_velocity))
        self.energy_cost_rate = 0.0004 + 0.002 * effort
        self.energy = max(0.0, self.energy - dt * self.energy_cost_rate)

    def snapshot(self, world, contacts):
        if contacts is None:
            contacts = self.contacts
        hips, knees, feet, _, _, _ = self.geometry()
        front = self.position + rotate([21, 0], self.heading)
        antennae = np.array([front + rotate([28, -13], self.heading),
                             front + rotate([28, 13], self.heading)])
        mouth = self.position + rotate([25, 0], self.heading)
        forward = float(self.velocity[0] * math.cos(self.heading) + self.velocity[1] * math.sin(self.heading))
        lateral = float(-self.velocity[0] * math.sin(self.heading) + self.velocity[1] * math.cos(self.heading))
        return {"center": np.round(self.position, 2).tolist(),
                "heading": round(self.heading, 4),
                "hips": np.round(hips, 2).tolist(),
                "knees": np.round(knees, 2).tolist(),
                "feet": np.round(feet, 2).tolist(),
                "antennae": np.round(antennae, 2).tolist(),
                "mouth": np.round(mouth, 2).tolist(),
                "angles": np.round(self.angles, 3).tolist(),
                "contacts": np.asarray(contacts, dtype=bool).tolist(),
                "antenna_odor": np.round(world.odor_at(antennae), 3).tolist(),
                "touch": self.touch, "energy": round(self.energy, 4),
                "energy_cost_rate": self.energy_cost_rate,
                "jaw": round(self.jaw_open, 2),
                "speed": round(float(np.linalg.norm(self.velocity)), 2),
                "flow": [forward, lateral, float(self.angular_velocity)]}
