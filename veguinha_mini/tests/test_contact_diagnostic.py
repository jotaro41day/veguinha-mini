import unittest
import math
import numpy as np
import pyximport
from types import SimpleNamespace

from veguinha_mini.body import Body
from veguinha_mini.world import World


class ContactDiagnosticTests(unittest.TestCase):
    def test_sensory_bias_recruits_sensory_neurons(self):
        pyximport.install(setup_args={"include_dirs":np.get_include()},language_level=3)
        from veguinha_mini.fast_kernel import simulate
        args=(np.zeros(129,dtype=np.int32),np.zeros(0,dtype=np.int32),np.zeros(0),
              1,200,42,5.,.8,0.)
        kw=dict(return_state=True,external_angles=np.zeros(2),
                external_contacts=np.zeros(1))
        silent=simulate(*args,**kw,sensory_bias=1)[-1]['total'][120:128].sum()
        active=simulate(*args,**kw,sensory_bias=5)[-1]['total'][120:128].sum()
        self.assertEqual(silent,0)
        self.assertGreater(active,0)

    def test_fore_aft_mirror_reverses_signed_speed(self):
        pyximport.install(setup_args={"include_dirs": np.get_include()},language_level=3)
        from veguinha_mini.fast_body import advance
        world=World(); original, mirrored=Body(world),Body(world)
        original.vertical_support=mirrored.vertical_support=True
        first=[]; second=[]
        for block in range(100):
            phase=2*math.pi*block*.02/.38+np.array([.2,1.4,-1.,2.,-2.2,.6])
            torque=np.column_stack((.25*np.sin(phase),.25*np.cos(phase)))
            first.extend(advance(original,world,.002,torque,10,2)['forward'])
            second.extend(advance(mirrored,world,.002,torque,10,2,
                                  mirror_fore_aft=True)['forward'])
        np.testing.assert_allclose(first,-np.asarray(second),atol=1e-10)

    def test_full_tripod_cycles_match_reference_body(self):
        pyximport.install(setup_args={"include_dirs":np.get_include()},language_level=3)
        from veguinha_mini.fast_body import advance
        world=World();fast,slow=Body(world),Body(world)
        fast.vertical_support=slow.vertical_support=True
        shift=np.array([0,math.pi,0,math.pi,0,math.pi])
        for block in range(150):
            phase=2*math.pi*block*.02/.38+shift
            torque=np.column_stack((.42*np.sin(phase),.42*np.cos(phase)))
            advance(fast,world,.002,torque,10,2)
            for _ in range(10):
                slow.step(.002,torque,None,world)
        for field in ('position','velocity','heading','angles','joint_velocity',
                      'height','vertical_velocity','normal_load','ground_forces'):
            np.testing.assert_allclose(getattr(fast,field),getattr(slow,field),
                                       atol=1e-7,rtol=1e-7,err_msg=field)

    def test_load_reflex_targets_only_local_reciprocal_interneurons(self):
        pyximport.install(setup_args={"include_dirs":np.get_include()},language_level=3)
        from veguinha_mini.fast_kernel import simulate
        args=(np.zeros(129,dtype=np.int32),np.zeros(0,dtype=np.int32),np.zeros(0),
              1,1,42,5.,.8,0.)
        kw=dict(return_state=True,external_angles=np.zeros(2),external_contacts=np.zeros(1),
                external_loads=np.ones(1))
        base=simulate(*args,**kw,load_reflex_gain=0)[-1]['v']
        loaded=simulate(*args,**kw,load_reflex_gain=2)[-1]['v']
        expected=np.zeros(128); expected[90:100]=2; expected[100:110]=2
        np.testing.assert_allclose(loaded-base,expected,atol=1e-12)
        extension=simulate(*args,**kw,load_reflex_gain=2,
                           load_reflex_knee_action=1)[-1]['v']
        expected_extension=np.zeros(128)
        expected_extension[90:100]=2; expected_extension[110:120]=2
        np.testing.assert_allclose(extension-base,expected_extension,atol=1e-12)

    def test_joint_links_are_local_inhibitory_and_preserve_old_weights(self):
        from veguinha_mini.contact_diagnostic import joint_links
        cells=np.arange(768).reshape(6,128)
        probe=SimpleNamespace(inhibitory=cells[:,80:120].reshape(6,2,2,10),
                              premotor=cells[:,32:80].reshape(6,2,2,12))
        evaluator=SimpleNamespace(seed=42,probe=probe,
            starts=np.r_[0,np.cumsum(np.bincount([0,100],minlength=768))].astype(np.int32),
            targets=np.array([32,32],dtype=np.int32),kinds=np.array([0,3],dtype=np.int8))
        weights=joint_links(evaluator,np.array([1.,-2.]),3.)
        source=np.repeat(np.arange(768),np.diff(evaluator.starts))
        new=evaluator.kinds==7
        self.assertEqual(new.sum(),720)
        self.assertTrue(np.all(weights[new]<0))
        self.assertTrue(np.all(source[new]//128==evaluator.targets[new]//128))
        self.assertTrue(np.all((source[new]%128>=80)&(source[new]%128<120)))
        np.testing.assert_array_equal(weights[~new],[1.,-2.])

    def test_height_load_gate_overrides_requested_contact(self):
        body = Body(World())
        body.angles[:, 1] = .1
        body.joint_velocity[:, 0] = .2
        body.step(.002, np.zeros((6, 2)), None, World())
        np.testing.assert_array_equal(body.ground_forces, 0)
        np.testing.assert_array_equal(body.normal_load, 0)
        self.assertTrue(np.all(body.foot_height > 0))
        body.angles[:, 1] = -.01
        body.step(.002, np.zeros((6, 2)), None, World())
        self.assertTrue(np.all(body.normal_load > 0))
        self.assertTrue(all(body.snapshot(World(), None)['contacts']))
        self.assertTrue(np.all(np.linalg.norm(body.ground_forces,axis=1) <=
                               body.friction_coefficient*body.normal_load + 1e-10))

    def test_compiled_legacy_substeps_match_body(self):
        pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)
        from veguinha_mini.fast_body import advance
        rng = np.random.default_rng(104)
        world = World()
        for mode, dt in [(m,d) for m in (0,1,2) for d in (.02, .01, .005, .002)]:
            fast, slow = Body(world), Body(world)
            slow.vertical_support = fast.vertical_support = mode == 2
            initial = rng.normal(0, .05, (6, 2))
            fast.angles[:] = initial; slow.angles[:] = initial
            for block in range(20):
                torque = rng.uniform(-.8, .8, (6, 2))
                n = round(.02 / dt)
                advance(fast, world, dt, torque, n, mode)
                for _ in range(n):
                    slow.step(dt, torque, slow.angles[:, 1] < 0 if mode == 0 else None, world)
                for attr in ('angles', 'joint_velocity', 'position', 'velocity',
                             'heading', 'angular_velocity', 'energy', 'ground_forces',
                             'height', 'vertical_velocity'):
                    np.testing.assert_allclose(getattr(fast, attr), getattr(slow, attr),
                                               atol=2e-10, rtol=1e-10, err_msg=attr)


if __name__ == '__main__':
    unittest.main()
