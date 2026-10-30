"""Métricas observacionais do corpo, sem controle comportamental."""

import numpy as np

from .body import LEG_NAMES


NEIGHBORS = ((0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5))


def net_path_ratio(displacement, path_length):
    """Razão oficial: soma dos deslocamentos líquidos por episódio / caminho total."""
    return float(displacement) / max(float(path_length), 1e-9)


class Metrics:
    def __init__(self, body):
        self.origin = body.position.copy()
        self.previous = body.position.copy()
        self.completed_displacement = 0.0
        self.episodes = 1
        self.previous_energy = body.energy
        self.path_length = 0.0
        self.energy_spent = 0.0
        self.steps = 0
        self.angle_mean = np.zeros_like(body.angles)
        self.angle_m2 = np.zeros_like(body.angles)
        self.phase_sum = np.zeros(len(NEIGHBORS))
        self.phase_count = np.zeros(len(NEIGHBORS))
        self.speed_sum = 0.0
        self.torque_sum = 0.0
        self.forward_flow_sum = 0.0
        self.lateral_flow_abs_sum = 0.0
        self.rotation_abs_sum = 0.0
        self.wall_collisions = 0
        self.contact_sum = np.zeros(6)
        self.contact_pair_sum = np.zeros(len(NEIGHBORS))

    def episode_reset(self, old_body, new_body):
        self.completed_displacement += float(np.linalg.norm(old_body.position - self.origin))
        self.origin = new_body.position.copy()
        self.previous = new_body.position.copy()
        self.previous_energy = new_body.energy
        self.episodes += 1

    def update(self, body, torque, contacts=None, contact_samples=None):
        self.steps += 1
        self.path_length += float(np.linalg.norm(body.position - self.previous))
        self.previous = body.position.copy()
        self.energy_spent += max(0.0, self.previous_energy - body.energy)
        self.previous_energy = body.energy
        delta = body.angles - self.angle_mean
        self.angle_mean += delta / self.steps
        self.angle_m2 += delta * (body.angles - self.angle_mean)
        phase = np.arctan2(body.joint_velocity[:, 0], body.angles[:, 0])
        moving = np.abs(body.joint_velocity[:, 0]) + np.abs(body.angles[:, 0]) > .01
        for index, (a, b) in enumerate(NEIGHBORS):
            if moving[a] and moving[b]:
                self.phase_sum[index] += np.cos(phase[a] - phase[b])
                self.phase_count[index] += 1
        self.speed_sum += float(np.linalg.norm(body.velocity))
        self.torque_sum += float(np.mean(np.abs(torque)))
        self.forward_flow_sum += float(body.velocity[0] * np.cos(body.heading)
                                       + body.velocity[1] * np.sin(body.heading))
        self.lateral_flow_abs_sum += float(abs(-body.velocity[0] * np.sin(body.heading)
                                           + body.velocity[1] * np.cos(body.heading)))
        self.rotation_abs_sum += abs(float(body.angular_velocity))
        self.wall_collisions += int(body.touch)
        if contact_samples is not None:
            samples=np.asarray(contact_samples,dtype=float)
            if samples.ndim!=2 or samples.shape[1]!=6 or len(samples)==0:
                raise ValueError('amostras de contato devem ter forma N x 6')
            self.contact_sum+=samples.mean(0)
            for index,(a,b) in enumerate(NEIGHBORS):
                self.contact_pair_sum[index]+=(samples[:,a]*samples[:,b]).mean()
        elif contacts is not None:
            contact = np.asarray(contacts, dtype=float)
            self.contact_sum += contact
            for index, (a, b) in enumerate(NEIGHBORS):
                self.contact_pair_sum[index] += contact[a] * contact[b]

    def snapshot(self, body):
        variance = self.angle_m2 / max(1, self.steps)
        displacement = self.completed_displacement + float(np.linalg.norm(body.position - self.origin))
        contact_fraction = self.contact_sum / max(1, self.steps)
        contact_correlation = []
        for index, (a, b) in enumerate(NEIGHBORS):
            mean_product = self.contact_pair_sum[index] / max(1, self.steps)
            denom = np.sqrt(contact_fraction[a] * (1 - contact_fraction[a])
                            * contact_fraction[b] * (1 - contact_fraction[b]))
            contact_correlation.append((mean_product - contact_fraction[a] * contact_fraction[b])
                                       / denom if denom > 1e-9 else 0.0)
        return {
            "displacement": round(displacement, 4),
            "episodes": self.episodes,
            "path_length": round(self.path_length, 4),
            "net_path": round(net_path_ratio(displacement, self.path_length), 6),
            "speed": round(float(np.linalg.norm(body.velocity)), 4),
            "mean_speed": round(self.speed_sum / max(1, self.steps), 4),
            "joint_variance": np.round(variance, 7).tolist(),
            "neighbor_phase_correlation": np.round(self.phase_sum / np.maximum(1, self.phase_count), 4).tolist(),
            "contact_fraction": np.round(contact_fraction, 4).tolist(),
            "contact_correlation": np.round(contact_correlation, 4).tolist(),
            "energy_spent": round(self.energy_spent, 6),
            "mean_abs_torque": round(self.torque_sum / max(1, self.steps), 4),
            "mean_forward_flow": round(self.forward_flow_sum / max(1, self.steps), 4),
            "mean_abs_lateral_flow": round(self.lateral_flow_abs_sum / max(1, self.steps), 4),
            "mean_abs_rotation": round(self.rotation_abs_sum / max(1, self.steps), 4),
            "wall_collisions": self.wall_collisions,
        }

    @staticmethod
    def csv_columns(region_names, joint_names=("hip", "knee")):
        return (["seed", "t_s", "episodes", "displacement", "path_length", "net_path", "speed", "mean_speed",
                 "energy_spent", "mean_abs_torque", "motor_saturation"]
                + [f"variance_{leg}_{joint}" for leg in LEG_NAMES for joint in joint_names]
                + [f"phase_{LEG_NAMES[a]}_{LEG_NAMES[b]}" for a, b in NEIGHBORS]
                + [f"rate_{name}" for name in region_names])

    @staticmethod
    def csv_row(seed, t_s, snapshot, saturation, rates):
        return ([seed, round(t_s, 3), snapshot["episodes"]]
                + [snapshot[name] for name in ("displacement", "path_length", "net_path", "speed",
                                               "mean_speed", "energy_spent", "mean_abs_torque")]
                + [round(saturation, 5)]
                + np.asarray(snapshot["joint_variance"]).ravel().tolist()
                + snapshot["neighbor_phase_correlation"] + list(rates))
