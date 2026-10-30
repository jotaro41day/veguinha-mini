# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Núcleo Cython para ensaios curtos; reproduz o Euler Izhikevich de network.py."""

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, log, cos, sin
from libc.stdint cimport uint32_t


cdef inline uint32_t random_u32(uint32_t *state) noexcept nogil:
    cdef uint32_t x = state[0]
    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5
    state[0] = x
    return x


cdef inline double normal_pair(uint32_t *state, double *spare, int *has_spare) noexcept nogil:
    cdef double scale = 1.0 / 4294967296.0
    cdef double radius, angle
    if has_spare[0]:
        has_spare[0] = 0
        return spare[0]
    radius = sqrt(-2.0 * log((random_u32(state) + 0.5) * scale))
    angle = 6.283185307179586 * (random_u32(state) * scale)
    spare[0] = radius * sin(angle)
    has_spare[0] = 1
    return radius * cos(angle)


def simulate(cnp.ndarray[cnp.int32_t, ndim=1] starts,
             cnp.ndarray[cnp.int32_t, ndim=1] targets,
             cnp.ndarray[cnp.float64_t, ndim=1] weights,
             int legs, int steps, int seed, double drive_base, double proprio,
             double noise_amp=2.5, state=None, external_angles=None,
             bint return_state=False, external_contacts=None,
             double inhibitory_bias=-2.0, double motor_bias=5.0,
             double adaptation_a=.003, double adaptation_d=2.0,
             edge_kinds=None, double contact_gain=1.0,
             double interleg_contact_gain=0.0,
             double cross_tau_ms=150.0, external_loads=None,
             double load_reflex_gain=0.0, double sensory_bias=1.0,
             int load_reflex_knee_action=0, double stance_load_gain=0.0,
             double contact_phase_reset_gain=0.0):
    if cross_tau_ms <= 1.0:
        raise ValueError("cross_tau_ms deve ser maior que 1 ms")
    if load_reflex_knee_action < 0 or load_reflex_knee_action > 1:
        raise ValueError("load_reflex_knee_action must be 0 or 1")
    if stance_load_gain < 0:
        raise ValueError("stance_load_gain must be nonnegative")
    if contact_phase_reset_gain < 0:
        raise ValueError("contact_phase_reset_gain must be nonnegative")
    cdef int n = 128 * legs
    cdef int nbins = (steps + 49) // 50
    rng = np.random.default_rng(seed)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] v_np = -65.0 + rng.normal(0, 2, n)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] u_np = .2 * v_np.copy()
    cdef cnp.ndarray[cnp.float64_t, ndim=1] syn_np = np.zeros(n)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] slow_np = np.zeros(n)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] cross_np = np.zeros(n)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] pending_np = np.zeros((3, n))
    cdef cnp.ndarray[cnp.float64_t, ndim=2] pending_slow_np = np.zeros((3, n))
    cdef cnp.ndarray[cnp.float64_t, ndim=2] pending_cross_np = np.zeros((3, n))
    cdef cnp.ndarray[cnp.float64_t, ndim=1] theta_np = rng.normal(0, .03, legs * 2)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] sensory_theta_np = np.zeros(legs * 2)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] omega_np = np.zeros(legs * 2)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] motor_np = np.zeros(legs * 2)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] total_np = np.zeros(n, dtype=np.int64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] contact_age_np = np.zeros(legs)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] stance_end_np = np.zeros(legs)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] unload_np = np.zeros(legs)
    if state is not None:
        v_np = state['v']; u_np = state['u']; syn_np = state['syn']
        slow_np = state['slow']; pending_slow_np = state['pending_slow']
        cross_np = state['cross']; pending_cross_np = state['pending_cross']
        pending_np = state['pending']; theta_np = state['theta']
        omega_np = state['omega']; motor_np = state['motor']; total_np = state['total']
        contact_age_np = state.get('contact_age', contact_age_np)
        stance_end_np = state.get('stance_end', stance_end_np)
        unload_np = state.get('unload', unload_np)
    cdef cnp.ndarray[cnp.int32_t, ndim=4] counts_np = np.zeros((legs, 2, 2, nbins), dtype=np.int32)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] angles_np = np.zeros((legs * 2, nbins))
    cdef cnp.ndarray[cnp.float64_t, ndim=2] contacts_np = np.zeros((legs * 2, nbins))
    cdef cnp.ndarray[cnp.float64_t, ndim=2] drives_np = np.zeros((legs * 2, nbins))
    cdef double[:] v = v_np, u = u_np, syn = syn_np
    cdef double[:] slow = slow_np
    cdef double[:] cross = cross_np
    cdef double[:, :] pending = pending_np
    cdef double[:, :] pending_slow = pending_slow_np
    cdef double[:, :] pending_cross = pending_cross_np
    cdef double[:] theta = theta_np, omega = omega_np, motor = motor_np
    cdef double[:] contact_age = contact_age_np, stance_end = stance_end_np, unload = unload_np
    cdef double[:] sensory_theta = sensory_theta_np
    cdef double[:] foot_contact = np.zeros(legs)
    cdef double[:] contact_current = np.zeros(legs)
    cdef double[:, :] neighbor = np.zeros((legs, legs))
    cdef cnp.int64_t[:] total = total_np
    cdef int[:, :, :, :] counts = counts_np
    cdef double[:, :] angles = angles_np, contacts = contacts_np
    cdef double[:, :] drives = drives_np
    cdef int[:] first = starts, target = targets
    cdef double[:] weight = weights
    cdef cnp.int8_t[:] kind = np.zeros(len(weights), dtype=np.int8) if edge_kinds is None else edge_kinds
    cdef int step, i, j, edge, leg, local, action, joint, bin_index, slot
    cdef int first_leg, second_leg
    cdef int offset = 0 if state is None else state['step']
    cdef bint external = external_angles is not None
    cdef double[:] sensed = np.zeros(legs * 2) if not external else external_angles
    cdef double[:] feet = np.zeros(legs) if external_contacts is None else external_contacts
    cdef double[:] loads = np.zeros(legs) if external_loads is None else external_loads
    cdef double a, b, c, d, baseline, sense, dv, old_v, x, drive_value
    cdef uint32_t random_state = <uint32_t>(seed + 1)
    cdef double spare_noise = 0.0
    cdef int has_spare_noise = 0
    if state is not None:
        random_state = state['rng']; spare_noise = state['spare']
        has_spare_noise = state['has_spare']
    if legs == 2:
        neighbor[0, 1] = 1.0
        neighbor[1, 0] = 1.0
    elif legs == 6:
        for first_leg, second_leg in ((0, 1), (1, 2), (3, 4), (4, 5),
                                      (0, 3), (1, 4), (2, 5)):
            neighbor[first_leg, second_leg] = 1.0
            neighbor[second_leg, first_leg] = 1.0
    for leg in range(legs):
        x = 0.0
        for j in range(legs):
            x += neighbor[leg, j]
        if x > 0:
            for j in range(legs):
                neighbor[leg, j] *= interleg_contact_gain / x

    for step in range(steps):
        slot = (step + offset) % 3
        bin_index = step // 50
        # Brian2: summed sensory current, Euler state updates, thresholds,
        # synaptic delivery (2 ms), then resets. No new spike affects this
        # timestep's state update.
        for j in range(legs * 2):
            sensory_theta[j] = sensed[j] if external else theta[j]
            drive_value = motor[j]
            if drive_value > 1.0:
                drive_value = 1.0
            elif drive_value < -1.0:
                drive_value = -1.0
            if not external:
                x = omega[j]
                omega[j] += .001 * (4 * drive_value - 2 * theta[j] - 1.8 * omega[j])
                theta[j] += .001 * x
            motor[j] *= 1.0 - 1.0 / 120.0
        for leg in range(legs):
            foot_contact[leg] = (feet[leg] if external_contacts is not None
                                 else sensory_theta[leg * 2 + 1] < 0)
            stance_end[leg] *= .95
            unload[leg] *= .95
            if foot_contact[leg] > .5:
                contact_age[leg] += .001
                # A sustained local contact triggers one short pulse to end
                # stance; its age resets on unloading, not on a global clock.
                if contact_age[leg] >= .06 and contact_age[leg] < .061:
                    stance_end[leg] = 1.0
            else:
                if contact_age[leg] > 0:
                    unload[leg] = 1.0
                contact_age[leg] = 0.0
        for leg in range(legs):
            contact_current[leg] = 0.0
            for j in range(legs):
                contact_current[leg] += neighbor[leg, j] * foot_contact[j]
        for i in range(n):
            leg = i // 128
            local = i - leg * 128
            a = .02; b = .2; c = -65.0; d = 8.0
            baseline = 5.0; sense = 0.0
            if local < 32:
                baseline = motor_bias
            elif local < 80:
                a = adaptation_a; d = adaptation_d; baseline = drive_base
                # Measured foot load excites only the local hip retractor
                # premotor half-center; the oscillator still selects spikes.
                if 44 <= local < 56:
                    sense = stance_load_gain * loads[leg]
                if 32 <= local < 44:
                    sense += contact_phase_reset_gain * (stance_end[leg] + unload[leg])
            elif 80 <= local < 120:
                a = .1; d = 2.0; baseline = inhibitory_bias
                action = ((local - 80) // 10) % 2
                sense = contact_gain * (foot_contact[leg] +
                                        contact_current[leg] * (action == 0))
                if 80 <= local < 90:
                    sense += contact_phase_reset_gain * stance_end[leg]
                # Measured load recruits hip extension and a local knee
                # reflex through reciprocal inhibition. Action 0 is the
                # original unloading path; action 1 is an experimental
                # extension path retained for reproducible comparison.
                # Input is measured normal load; no time, phase, or gait signal.
                joint = (local - 80) // 20
                if (joint == 0 and action == 1) or (joint == 1 and action == load_reflex_knee_action):
                    sense += load_reflex_gain * loads[leg]
            elif local >= 120:
                baseline = sensory_bias
                action = (local - 120) // 2
                joint = action // 2
                j = leg * 2 + joint
                sense = -proprio * sensory_theta[j]
                if action % 2:
                    sense = -sense
                if sense < 0:
                    sense = 0.0
            old_v = v[i]
            dv = .04 * old_v * old_v + 5 * old_v + 140 - u[i] + baseline + syn[i] + slow[i] + cross[i] + sense + noise_amp * normal_pair(&random_state, &spare_noise, &has_spare_noise)
            v[i] = old_v + dv
            u[i] += a * (b * old_v - u[i])
            syn[i] *= .8
            slow[i] *= 1.0 - 1.0 / 150.0
            cross[i] *= 1.0 - 1.0 / cross_tau_ms
            if v[i] < -90:
                v[i] = -90
            if v[i] >= 30:
                total[i] += 1
                v[i] = c
                u[i] += d
                for edge in range(first[i], first[i+1]):
                    if kind[edge] == 3:
                        pending_slow[(step + offset + 2) % 3, target[edge]] += weight[edge]
                    elif kind[edge] == 7:
                        pending_cross[(step + offset + 2) % 3, target[edge]] += weight[edge]
                    else:
                        pending[(step + offset + 2) % 3, target[edge]] += weight[edge]
                if local < 32:
                    action = local // 8
                    joint = action // 2
                    j = leg * 2 + joint
                    if action % 2:
                        motor[j] -= .028
                    else:
                        motor[j] += .028
                elif local < 80:
                    action = (local - 32) // 12
                    counts[leg, action // 2, action % 2, bin_index] += 1
        for i in range(n):
            syn[i] += pending[slot, i]
            pending[slot, i] = 0.0
            slow[i] += pending_slow[slot, i]
            pending_slow[slot, i] = 0.0
            cross[i] += pending_cross[slot, i]
            pending_cross[slot, i] = 0.0
        for j in range(legs * 2):
            if step % 50 == 49:
                angles[j, bin_index] = theta[j]
                contacts[j, bin_index] = theta[j] < 0
                drives[j, bin_index] = motor[j]
    if return_state:
        state = {'v': v_np, 'u': u_np, 'syn': syn_np, 'slow': slow_np,
                 'cross': cross_np, 'pending': pending_np,
                 'pending_slow': pending_slow_np,
                 'pending_cross': pending_cross_np,
                 'theta': theta_np, 'omega': omega_np, 'motor': motor_np,
                 'contact_age': contact_age_np, 'stance_end': stance_end_np,
                 'unload': unload_np,
                 'total': total_np, 'step': offset + steps, 'rng': int(random_state),
                 'spare': spare_noise, 'has_spare': has_spare_noise}
        return counts_np, angles_np, contacts_np, drives_np, state
    return counts_np, angles_np, contacts_np, drives_np
