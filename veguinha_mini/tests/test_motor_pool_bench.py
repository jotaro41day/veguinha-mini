import unittest
import numpy as np
from veguinha_mini.tools.motor_pool_bench import CASES,stimulus,decode_pool,cycle_stats


class MotorPoolBenchTests(unittest.TestCase):
    def test_stimulus_repeats_without_input_count_jitter(self):
        i,t=stimulus(42,shared=True)
        e=t[i<12]
        counts=np.histogram(e,np.arange(54)*.38)[0][:52]
        np.testing.assert_array_equal(counts,np.full(52,12))
        np.testing.assert_allclose(np.diff(t[i==0]),.38,atol=1e-12)

    def test_pool_size_does_not_multiply_torque(self):
        counts=np.zeros((1000,8),int);counts[5::19]=1
        a=decode_pool(counts,120)
        b=decode_pool(np.tile(counts,(1,4)),120)
        np.testing.assert_allclose(a['torque'],b['torque'],atol=1e-12)

    def test_each_case_changes_one_factor_against_its_parent(self):
        fields=('pool','filter_ms','noise','tonic','threshold_span','shared')
        for name,c in CASES.items():
            if name=='baseline':continue
            p=CASES[c['parent']]
            self.assertEqual(sum(c[k]!=p[k] for k in fields),1,name)

    def test_perfect_periodic_output_has_zero_rate_cv(self):
        c=np.zeros((1000,8),int);c[5::19]=1
        r=cycle_stats(c.sum(1)/8,np.ones(1000))
        self.assertAlmostEqual(r['rate_cv'],0)
        self.assertAlmostEqual(r['torque_cv'],0)


if __name__=='__main__':unittest.main()
