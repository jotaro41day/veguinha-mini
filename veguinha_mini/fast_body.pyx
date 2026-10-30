# cython: boundscheck=False, wraparound=False, cdivision=True
"""Native implementation of Body.step for controlled substep diagnostics.

Mode 0 is the legacy Body equations, including the pre-step knee contact mask.
The parity test guards this independent implementation against drift.
"""
import numpy as np
from libc.math cimport sin, cos, sqrt, fabs

cdef inline double clip(double x, double lo, double hi) noexcept:
    return lo if x < lo else hi if x > hi else x

def advance(body, world, double dt, torque, int steps, int mode=0,
            bint mirror_fore_aft=False, double friction_long=1.4,
            double friction_lat=4.0):
    if dt <= 0 or dt > .02 or steps < 1:
        raise ValueError('invalid substeps')
    if np.shape(torque) != (6, 2):
        raise ValueError('invalid torque shape')
    cdef double[:, :] q = body.angles, omega = body.joint_velocity
    cdef double[:, :] rest = body.rest_angles
    cdef double[:, :] forces = body.ground_forces
    cdef double[:] height_state = body.foot_height, load_state = body.normal_load
    cdef double[:, :] tor = np.ascontiguousarray(np.clip(torque, -1, 1), dtype=float)
    cdef double[:] pos = body.position, vel = body.velocity
    cdef double[:] base = np.array([-.85, -1.5707963267948966, -2.2, .85, 1.5707963267948966, 2.2])
    cdef double[:, :] anchors = np.array([[17., -12.], [0., -14.], [-17., -12.],
                                         [17., 12.], [0., 14.], [-17., 12.]])
    cdef double[:, :] contact = np.zeros((steps, 6))
    cdef double[:, :] impulse = np.zeros((steps, 6))
    cdef double[:] forward = np.zeros(steps), speed = np.zeros(steps)
    cdef double[:] lateral_flow = np.zeros(steps), rotation = np.zeros(steps)
    cdef double heading = body.heading, angular = body.angular_velocity
    cdef double energy = body.energy, rate = 0
    cdef double side, first, second, x1, y1, x2, y2, rx, ry, ax, ay
    cdef double jhx, jhy, jkx, jky, fvx, fvy, longitudinal, lateral
    cdef double fx, fy, mag, scale, tx, ty, moment, effort, co, si, limit, oldh
    cdef double height, load, zvel, force_limit
    cdef double lift = body.foot_lift_radius, stiffness = body.normal_stiffness
    cdef double damping = body.normal_damping, mu = body.friction_coefficient
    cdef double body_height = body.height, vz = body.vertical_velocity, normal_sum
    cdef double mass = body.mass, gravity = body.gravity, drop = body.nominal_foot_drop
    cdef int step, leg, joint
    cdef bint touched = False
    for step in range(steps):
        tx = 0; ty = 0; moment = 0; effort = 0; normal_sum = 0
        oldh = heading; co = cos(heading); si = sin(heading)
        for leg in range(6):
            contact[step, leg] = q[leg, 1] < 0
            for joint in range(2):
                omega[leg, joint] += dt * (4 * tor[leg, joint] -
                    2 * (q[leg, joint] - rest[leg, joint]) - 1.8 * omega[leg, joint])
                q[leg, joint] += dt * omega[leg, joint]
                limit = 1.2 if joint == 0 else 1.5
                q[leg, joint] = clip(q[leg, joint], -limit, limit)
                if fabs(q[leg, joint]) >= limit and q[leg, joint] * omega[leg, joint] > 0:
                    omega[leg, joint] = 0
            side = 1 if leg < 3 else -1
            if mirror_fore_aft:
                side = -side
            load = 0
            if mode >= 1:
                height = lift*sin(q[leg,1])
                zvel = lift*cos(q[leg,1])*omega[leg,1]
                if mode == 2:
                    height += body_height - drop
                    zvel += vz
                load = max(0., -stiffness*height - damping*zvel) if height <= 0 else 0
                normal_sum += load
                height_state[leg] = height; load_state[leg] = load
                contact[step,leg] = load > 0
            first = (3.141592653589793 - base[leg] if mirror_fore_aft else base[leg]) + side * q[leg, 0] + heading
            second = first + side * q[leg, 1]
            x1 = 29 * cos(first); y1 = 29 * sin(first)
            x2 = 24 * cos(second); y2 = 24 * sin(second)
            rx = (anchors[leg, 0] * (-1 if mirror_fore_aft else 1)) * co - anchors[leg, 1] * si + x1 + x2
            ry = (anchors[leg, 0] * (-1 if mirror_fore_aft else 1)) * si + anchors[leg, 1] * co + y1 + y2
            ax = x2 / 24; ay = y2 / 24
            jhx = -side * (y1 + y2); jhy = side * (x1 + x2)
            jkx = -side * y2; jky = side * x2
            fvx = vel[0] - angular * ry + jhx * omega[leg, 0] + jkx * omega[leg, 1]
            fvy = vel[1] + angular * rx + jhy * omega[leg, 0] + jky * omega[leg, 1]
            longitudinal = fvx * ax + fvy * ay
            lateral = -fvx * ay + fvy * ax
            fx = -friction_long * longitudinal * ax + friction_lat * lateral * ay
            fy = -friction_long * longitudinal * ay - friction_lat * lateral * ax
            mag = sqrt(fx*fx + fy*fy)
            force_limit = mu*load if mode >= 1 else 65
            scale = force_limit / mag if mag > force_limit else 1
            if contact[step, leg] == 0:
                scale = 0
            fx *= scale; fy *= scale
            forces[leg, 0] = fx; forces[leg, 1] = fy
            impulse[step, leg] = (fx*co + fy*si) * dt
            omega[leg, 0] += dt * (fx*jhx + fy*jhy) / 1200
            omega[leg, 1] += dt * (fx*jkx + fy*jky) / 600
            if mode >= 1:
                omega[leg,1] += dt*load*lift*cos(q[leg,1])/600
            for joint in range(2):
                limit = 1.2 if joint == 0 else 1.5
                if fabs(q[leg, joint]) >= limit and q[leg, joint] * omega[leg, joint] > 0:
                    omega[leg, joint] = 0
                effort += tor[leg, joint]**2 + .2 * fabs(tor[leg, joint]*omega[leg, joint])
            tx += fx; ty += fy; moment += rx*fy - ry*fx
        vel[0] += dt * (tx / 3 - 2*vel[0])
        vel[1] += dt * (ty / 3 - 2*vel[1])
        angular += dt * (moment / 4500 - 2.5*angular)
        if mode == 2:
            vz += dt*(normal_sum/mass - gravity - 1.8*vz)
            body_height += dt*vz
        pos[0] += vel[0]*dt; pos[1] += vel[1]*dt
        heading += angular*dt
        while heading >= 3.141592653589793: heading -= 6.283185307179586
        while heading < -3.141592653589793: heading += 6.283185307179586
        for joint in range(2):
            limit = (world.width if joint == 0 else world.height) - 22
            if pos[joint] < 22:
                pos[joint] = 22; vel[joint] = max(0., -.2*vel[joint]); touched = True
            elif pos[joint] > limit:
                pos[joint] = limit; vel[joint] = min(0., -.2*vel[joint]); touched = True
        forward[step] = vel[0]*cos(heading) + vel[1]*sin(heading)
        lateral_flow[step] = -vel[0]*sin(heading) + vel[1]*cos(heading)
        rotation[step] = angular
        speed[step] = sqrt(vel[0]**2 + vel[1]**2)
        rate = .0004 + .002*effort
        energy = max(0., energy - dt*rate)
    body.heading = heading; body.angular_velocity = angular
    body.energy = energy; body.energy_cost_rate = rate; body.touch = bool(touched)
    body.height = body_height; body.vertical_velocity = vz
    body.contacts[:] = np.asarray(contact)[steps - 1].astype(bool)
    return {'contacts': np.asarray(contact), 'impulses': np.asarray(impulse),
            'forward': np.asarray(forward), 'speed': np.asarray(speed),
            'lateral': np.asarray(lateral_flow), 'rotation': np.asarray(rotation)}
