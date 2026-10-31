import unittest,tempfile
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from brian2 import start_scope
from veguinha_mini.config import Config
from veguinha_mini.sensory import ValueFlowFilter,inject_sensors


class AdaptiveHomeostasisTests(unittest.TestCase):
    def test_reserve_receptor_remains_sensitive_beyond_five_percent_deficit(self):
        from veguinha_mini.sensory import homeostatic_need
        energy=(1.,.95,.8,.5,.1,0.)
        need=[homeostatic_need(e,.05,True) for e in energy]
        self.assertEqual(need[0],0);self.assertEqual(need[-1],1)
        self.assertTrue(np.all(np.diff(need)>0))
        self.assertEqual(homeostatic_need(.95,.05),1)

    def test_receptor_parameters_and_time_are_validated(self):
        for fields in ({'odor_adaptation_tau_s':0},{'odor_adaptation_strength':float('nan')},{'fatigue_drive_suppression':1.1}):
            with self.assertRaises(ValueError):self.config(**fields)
        f=ValueFlowFilter(200);c=self.config();f.adapt_odor(self.sample(),.02,c);before=f.odor_baseline.copy()
        f.adapt_odor(self.sample(),0,c);np.testing.assert_array_equal(before,f.odor_baseline)
        with self.assertRaises(ValueError):f.adapt_odor(self.sample(),float('nan'),c)

    def afferent_model(self,c):
        from veguinha_mini.network import build_network
        model=build_network(c,record_spikes=False)
        dn=model.regions['dn'];go,hold=np.array_split(dn[~model.inhibitory[dn]],2)
        model.intent=SimpleNamespace(go=go,hold=hold)
        return model

    def config(self,**kwargs):
        return Config(scale=.1,backend='numpy',phase5_enabled=True,neural_value_channels=True,
                      homeostasis_enabled=True,**kwargs)

    def sample(self,energy=.95,fatigue=0):
        return dict(angles=np.zeros((6,2)),contacts=[False]*6,touch=False,energy=energy,
                    flow=[0,0,0],energy_cost_rate=0,antenna_odor=[.7,.2],novelty=0,
                    ecology=dict(fatigue=fatigue,dust=np.zeros(8)))

    def test_familiar_odor_habituates_and_new_channel_recovers_without_timer_actions(self):
        c=self.config();f=ValueFlowFilter(200);sample=self.sample()
        first=f.adapt_odor(sample,.02,c)
        for _ in range(2000):last=f.adapt_odor(sample,.02,c)
        self.assertLess(last[0,0],first[0,0]*.7)
        self.assertGreater(last[0,0],.2)
        self.assertEqual(last[2,0],1)
        # Receptor recovers after the stimulus disappears; no action is selected.
        absent=self.sample();absent['antenna_odor']=[0,0]
        for _ in range(2000):f.adapt_odor(absent,.02,c)
        restored=f.adapt_odor(sample,.02,c)
        self.assertGreater(restored[0,0],last[0,0]*1.5)
        f.reset();np.testing.assert_array_equal(f.odor_baseline,0)

    def test_fatigue_and_reserve_have_independent_neural_inputs(self):
        start_scope();c=self.config();m=self.afferent_model(c)
        go=m.intent.go;hold=m.intent.hold;motors=np.concatenate([np.concatenate(list(p.values())) for p in m.motor_pools.values()])
        def current(sample,**kwargs):inject_sensors(m,sample,config=c,**kwargs);return np.asarray(m.group.I_sense[:]).copy(),np.asarray(m.group.I_odor[:]).copy()
        fresh,_=current(self.sample());tired,_=current(self.sample(fatigue=1))
        self.assertLess(tired[go].mean(),fresh[go].mean()*.5)
        self.assertGreater(tired[hold].mean(),fresh[hold].mean()+5)
        deep,_=current(self.sample(energy=.3));moderate,_=current(self.sample(energy=.94))
        ids=m.regions['value'][~m.inhibitory[m.regions['value']]]
        self.assertGreater(deep[ids].mean(),moderate[ids].mean())
        np.testing.assert_array_equal(tired[motors],fresh[motors]);np.testing.assert_array_equal(deep[m.regions['dan']],0)
        # Greater need increases sensory salience even when short-range hunger is saturated.
        _,hungry=current(self.sample(energy=.3));_,fed=current(self.sample(energy=1))
        self.assertGreater(hungry.max(),fed.max())

    def test_habituation_does_not_attenuate_pain_or_contact_currents(self):
        start_scope();c=self.config(nociception_enabled=True);m=self.afferent_model(c)
        sample=self.sample();sample['pain']=[1,0]
        inject_sensors(m,sample,config=c,odor_modulation=np.ones((4,2)))
        before=np.asarray(m.group.I_sense[:]).copy()
        inject_sensors(m,sample,config=c,odor_modulation=np.full((4,2),.4))
        after=np.asarray(m.group.I_sense[:]).copy()
        self.assertEqual(after[m.intent.hold].mean(),before[m.intent.hold].mean())
        value=m.regions['value'][m.inhibitory[m.regions['value']]]
        np.testing.assert_array_equal(after[value],before[value])

    def test_filter_checkpoint_and_old_configuration_migration(self):
        from veguinha_mini.sim import Simulation
        start_scope();c=replace(self.config(),scale=.05)
        s=Simulation(c,headless=True);s.paused=True
        for _ in range(50):s.value_filter.adapt_odor(self.sample(),.02,c)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state';s.save_checkpoint(p);s.value_filter.reset();s.load_checkpoint(p)
            self.assertGreater(s.value_filter.odor_baseline[0,0],0)
            expected=s.value_filter.odor_baseline.copy()
            q=Simulation(replace(c,homeostasis_enabled=False),headless=True);q.load_checkpoint(p)
            np.testing.assert_array_equal(q.value_filter.odor_baseline,expected)
            with self.assertRaises(ValueError):Simulation(replace(c,noise=c.noise+1),headless=True).load_checkpoint(p)

if __name__=='__main__':unittest.main()
