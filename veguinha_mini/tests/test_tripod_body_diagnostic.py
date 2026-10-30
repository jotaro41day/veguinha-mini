import math
import unittest

import numpy as np

from veguinha_mini.body import Body
from veguinha_mini.tripod_body_diagnostic import (_phase_contact_r,
                                                     _phase_from_counts)
from veguinha_mini.world import World


class TripodBodyDiagnosticTests(unittest.TestCase):
    def test_ground_force_probe_is_passive_and_zero_for_airborne_feet(self):
        world = World()
        measured, reference = Body(world), Body(world)
        torque = np.zeros((6, 2)); torque[0, 0] = .8
        air = np.zeros(6, dtype=bool)
        for _ in range(50):
            measured.step(.02, torque, air, world)
            reference.step(.02, torque, air, world)
        np.testing.assert_array_equal(measured.ground_forces, 0)
        np.testing.assert_array_equal(measured.position, reference.position)
        ground = air.copy(); ground[0] = True
        measured.step(.02, torque, ground, world)
        self.assertGreater(np.linalg.norm(measured.ground_forces[0]), 0)
        np.testing.assert_array_equal(measured.ground_forces[1:], 0)

    def test_phase_probe_recovers_alternating_signal_and_contact(self):
        t = np.arange(1000) * .02
        counts = np.zeros((1000, 6, 2, 2))
        phase = 2 * math.pi * t / .38
        for leg in range(6):
            wave = np.sin(phase + (math.pi if leg % 2 else 0))
            counts[:, leg, 0, 0] = np.maximum(wave, 0)
            counts[:, leg, 0, 1] = np.maximum(-wave, 0)
        observed = _phase_from_counts(counts)
        self.assertIsNotNone(observed)
        opposite = np.angle(np.exp(1j * (observed[:, 0] - observed[:, 1])))
        self.assertLess(abs(abs(np.angle(np.exp(1j * opposite).mean())) - math.pi), .2)
        self.assertGreater(_phase_contact_r(observed[:, 0], np.sin(phase) < 0), .7)


if __name__ == "__main__":
    unittest.main()
