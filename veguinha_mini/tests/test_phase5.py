import math
import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from veguinha_mini.body import Body
from veguinha_mini.config import Config
from veguinha_mini.learning import MotorPlasticity
from veguinha_mini.network import build_network
from veguinha_mini.perturbation import measure
from veguinha_mini.sensory import inject_sensors
from veguinha_mini.sim import Simulation
from veguinha_mini.training import summarize
from veguinha_mini.world import World


def small_config(**changes):
    return Config(scale=.1, backend="numpy", phase5_enabled=True, **changes)


class Phase5Tests(unittest.TestCase):
    def test_summary_requires_paired_controls_and_all_five_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.csv"
            baseline.write_text("seed,displacement_per_energy\nmean,1000\n42,1000\n")
            for condition, ratio, efficiency in (("learning", .24, 50),
                                                 ("off", .09, 15),
                                                 ("shuffled", .11, 17)):
                with (root / f"{condition}_seed42.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, ["t_s", "net_path", "net_energy",
                                                     "contralateral_phase",
                                                     "contralateral_contact_correlation"])
                    writer.writeheader()
                    writer.writerow({"t_s": 300, "net_path": ratio,
                                     "net_energy": efficiency,
                                     "contralateral_phase": -.2,
                                     "contralateral_contact_correlation": -.2})
            with redirect_stdout(io.StringIO()):
                reports = summarize(root, [42, 43], 300, baseline)
            assert reports[0]["pass"] is True
            assert reports[0]["off_net_energy"] == 15
            assert math.isclose(reports[0]["energy_gain_vs_off"], 50 / 15)
            assert reports[1]["pass"] is None

    def test_phase5_contact_threshold_is_passive_and_configurable(self):
        self.assertEqual(Config().contact_threshold, .35)
        self.assertEqual(small_config().contact_threshold, 0)
        self.assertEqual(small_config(foot_contact_angle=.1).contact_threshold, .1)

    def test_flow_is_body_relative_and_reaches_sensory_value_dan(self):
        world = World()
        body = Body(world)
        body.heading = math.pi / 2
        body.velocity[:] = [0, 2]
        sample = body.snapshot(world, np.ones(6, dtype=bool))
        assert np.allclose(sample["flow"], [2, 0, 0], atol=1e-8)
        model = build_network(small_config())
        inject_sensors(model, sample, config=small_config())
        first = np.asarray(model.group.I_sense[:]).copy()
        assert first[model.regions["flow"][:8]].mean() > 0
        assert first[model.regions["dan"]].mean() > 0
        sample["flow"] = [-2, 0, 0]
        inject_sensors(model, sample, config=small_config())
        second = np.asarray(model.group.I_sense[:]).copy()
        assert second[model.regions["dan"]].mean() < first[model.regions["dan"]].mean()
        sample["flow"] = [2, 0, .2]
        inject_sensors(model, sample, config=small_config())
        turning = np.asarray(model.group.I_sense[:]).copy()
        assert turning[model.regions["dan"]].mean() < first[model.regions["dan"]].mean()


    def test_proprio_plastic_edges_only_reach_matching_leg(self):
        model = build_network(small_config())
        pre = np.asarray(model.synapses.i[:], dtype=int)[model.plastic_edges]
        post = np.asarray(model.synapses.j[:], dtype=int)[model.plastic_edges]
        channels = model.regions["proprio"].reshape(6, 2, 2, -1)
        for index, leg in enumerate(("lf", "lm", "lh", "rf", "rm", "rh")):
            mask = np.isin(pre, channels[index].ravel())
            assert mask.any()
            assert np.isin(post[mask], model.regions[f"vnc_{leg}"]).all()

    def test_flow_channels_project_to_orientation_and_descending_neurons(self):
        model = build_network(small_config())
        flow = model.regions["flow"].reshape(3, 2, 8)
        assert not model.inhibitory[flow[:, 0]].any()
        assert model.inhibitory[flow[:, 1]].all()
        pre = np.asarray(model.synapses.i[:], dtype=int)
        post = np.asarray(model.synapses.j[:], dtype=int)
        for target in ("cx", "dn", "value", "comparator"):
            assert (np.isin(pre, flow.ravel()) & np.isin(post, model.regions[target])).any()


    def test_motor_learning_requires_dan_spikes_and_respects_disable(self):
        config = small_config(learning_rate=.02)
        model = build_network(config)
        learning = MotorPlasticity(model, config)
        start = learning.weight.copy()
        count = np.zeros(len(model.group), dtype=np.int64)
        pre_cell, post_cell = int(learning.pre[0]), int(learning.post[0])
        for step in range(40):
            count[pre_cell] += 1
            learning.update(count, .01, enabled=True)
            count[post_cell] += 1
            count[model.regions["dan"]] += 1
            learning.update(count, .01, enabled=True)
        assert not np.array_equal(learning.weight, start)
        before = learning.weight.copy()
        for _ in range(20):
            count[pre_cell] += 1
            count[post_cell] += 1
            count[model.regions["dan"]] += 1
            learning.update(count, .01, enabled=False)
        assert np.array_equal(learning.weight, before)
        assert np.all(learning.weight >= learning.initial * .2)
        assert np.all(learning.weight <= learning.initial * 3)

    def test_weak_interleg_hebbian_rule_uses_local_spike_order(self):
        config = small_config(learning_rate=0, vnc_hebb_rate=.01)
        model = build_network(config)
        learning = MotorPlasticity(model, config)
        edge = int(np.flatnonzero(learning.interleg)[0])
        count = np.zeros(len(model.group), dtype=np.int64)
        before = float(learning.weight[edge])
        count[learning.pre[edge]] += 1
        learning.update(count, .02)
        count[learning.post[edge]] += 1
        for _ in range(4):
            learning.update(count, .02)
        assert learning.weight[edge] > before

    def test_motor_normalization_preserves_excitation_and_inhibition_separately(self):
        model = build_network(small_config())
        learning = MotorPlasticity(model, small_config())
        inhibitory = learning.sign < 0
        assert inhibitory.any() and (~inhibitory).any()
        learning.weight[inhibitory] *= .95
        learning.weight[~inhibitory] *= 1.05
        learning._normalize()
        for sign in (-1, 1):
            mask = learning.sign == sign
            initial = np.bincount(learning.post[mask], weights=learning.initial[mask],
                                  minlength=len(model.group))
            current = np.bincount(learning.post[mask], weights=learning.weight[mask],
                                  minlength=len(model.group))
            assert np.allclose(current[initial > 0], initial[initial > 0], rtol=.01)


    def test_checkpoint_restores_body_clock_and_weights(self):
        config = small_config()
        first = Simulation(config, headless=True)
        first.run_headless(.1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state"
            first.save_checkpoint(path)
            restored = Simulation(config, headless=True)
            restored.load_checkpoint(path)
            assert restored.simulated_ms == first.simulated_ms
            assert np.allclose(restored.body.position, first.body.position)
            assert np.array_equal(restored.motor_learning.weight, first.motor_learning.weight)
            assert restored.run_headless(.1)["path_length"] > 0


    def test_damage_command_zeroes_contact_and_torque(self):
        sim = Simulation(small_config(), headless=True)
        sim.submit({"cmd": "damage", "leg": "lf", "value": True})
        sim.run_headless(.05)
        assert sim.damaged_leg == "lf"
        assert not sim.contacts[0]

    def test_episode_reset_preserves_learned_network(self):
        sim = Simulation(small_config(), headless=True)
        sim.run_headless(.1)
        sim.plasticity = False
        before = np.asarray(sim.model.synapses.w[:]).copy()
        model = sim.model
        old_epoch = sim.epoch
        sim.submit({"cmd": "reset"})
        sim.run_headless(.02)
        assert sim.model is model
        assert sim.epoch == old_epoch + 1
        assert np.array_equal(np.asarray(sim.model.synapses.w[:]), before)

    def test_perturbation_reports_windows_before_and_after_damage(self):
        sim = Simulation(small_config(block_ms=20), headless=True)
        sim.run_headless(.1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state"
            sim.save_checkpoint(path)
            rows = measure(path, seconds=.1, interval_s=.1, baseline_s=.1,
                           seed=42, scale=.1, backend="numpy")
        assert [row["damaged"] for row in rows] == [False, True]
        assert rows[1]["path"] >= 0
