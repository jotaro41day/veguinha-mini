"""Contratos do andaime motor: oscilação local sem mapa de marcha pronto."""

import unittest

import numpy as np
from brian2 import Clock, NetworkOperation, ms

from veguinha_mini.isolated import isolated_config, isolated_simulation
from veguinha_mini.network import build_network


class HalfCenterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.model = build_network(isolated_config(seed=42, backend="numpy",
                                                       motor_scaffold=True), record_spikes=False)
        except TypeError as exc:
            cls.model = None
            cls.build_error = str(exc)
            return
        cls.pre = np.asarray(cls.model.synapses.i[:], dtype=int)
        cls.post = np.asarray(cls.model.synapses.j[:], dtype=int)
        cls.weights = np.asarray(cls.model.synapses.w[:], dtype=float)

    def _edges(self, sources, targets):
        return np.isin(self.pre, sources) & np.isin(self.post, targets)

    def test_each_joint_has_excitation_and_reciprocal_inhibition(self):
        self.assertIsNotNone(self.model, getattr(self, "build_error", ""))
        for leg, centers in self.model.half_centers.items():
            for joint in ("hip", "knee"):
                flex_e = centers[f"{joint}_flex_e"]
                flex_i = centers[f"{joint}_flex_i"]
                extend_e = centers[f"{joint}_extend_e"]
                extend_i = centers[f"{joint}_extend_i"]
                flex_motor = self.model.motor_pools[leg][f"{joint}_flex"]
                extend_motor = self.model.motor_pools[leg][f"{joint}_extend"]
                for source, target in ((flex_e, flex_motor), (extend_e, extend_motor),
                                       (flex_e, flex_i), (extend_e, extend_i)):
                    edge = self._edges(source, target)
                    self.assertTrue(edge.any())
                    self.assertTrue((self.weights[edge] > 0).all())
                for source, target in ((flex_i, extend_e), (extend_i, flex_e),
                                       (flex_i, extend_motor), (extend_i, flex_motor)):
                    edge = self._edges(source, target)
                    self.assertTrue(edge.any())
                    self.assertTrue((self.weights[edge] < 0).all())

    def test_plastic_edges_are_only_proprioception_or_interleg(self):
        self.assertIsNotNone(self.model, getattr(self, "build_error", ""))
        regions = self.model.regions
        pre = self.pre[self.model.plastic_edges]
        post = self.post[self.model.plastic_edges]
        proprio = np.isin(pre, regions["proprio"])
        target_leg = np.full(len(self.model.group), -1, dtype=int)
        for index, leg in enumerate(("lf", "lm", "lh", "rf", "rm", "rh")):
            target_leg[regions[f"vnc_{leg}"]] = index
        valid = ((proprio & (target_leg[post] >= 0))
                 | ((target_leg[pre] >= 0) & (target_leg[post] >= 0)
                    & (target_leg[pre] != target_leg[post])))
        self.assertGreater(len(pre), 0)
        self.assertTrue(valid.all())
        self.assertTrue(proprio.any())
        self.assertTrue((~proprio).any())

    def test_antagonists_alternate_without_tonic_inhibitory_saturation(self):
        sim = isolated_simulation(seed=42, backend="cython", tonic_drive=2,
                                  motor_scaffold=True)
        model = sim.model
        centers = model.half_centers["lf"]
        previous = np.zeros(len(model.group), dtype=np.int64)
        samples = []

        def record():
            nonlocal previous
            count = np.asarray(model.monitor.count[:], dtype=np.int64)
            new_spikes = count - previous
            previous = count.copy()
            samples.append([new_spikes[centers[f"hip_{action}_{kind}"]].sum()
                            for kind in ("e", "i") for action in ("flex", "extend")])

        model.network.add(NetworkOperation(record, clock=Clock(dt=sim.config.block_ms * ms),
                                           when="end", order=1))
        result = sim.run_headless(8)
        events = np.asarray(samples, dtype=float)
        averaged = [np.convolve(events[:, column], np.ones(5) / 5, mode="valid")
                    for column in range(4)]
        flex_extend = float(np.corrcoef(averaged[0], averaged[1])[0, 1])
        e_rate = events[:, :2].sum() / sum(len(centers[f"hip_{action}_e"])
                                          for action in ("flex", "extend")) / 8
        i_rate = events[:, 2:].sum() / sum(len(centers[f"hip_{action}_i"])
                                          for action in ("flex", "extend")) / 8
        self.assertLess(flex_extend, 0)
        self.assertLess(i_rate, 2 * e_rate)
        self.assertLess(result["motor_saturation"], .15)


if __name__ == "__main__":
    unittest.main()
