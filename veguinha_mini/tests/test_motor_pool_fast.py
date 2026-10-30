import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from veguinha_mini.tools.motor_pool_fast import raw,compare
from veguinha_mini.motor_pool_kernel import simulate


class MotorPoolFastTests(unittest.TestCase):
    def test_fast_results_cannot_be_used_as_brian2_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'baseline_42.json').write_text(json.dumps({'engine':'validated Cython Euler'}))
            with patch('veguinha_mini.tools.motor_pool_fast.OUT',Path(tmp)):
                with self.assertRaises(ValueError):compare('baseline',42)

    def test_tonic_motor_current_cannot_fire_without_premotor_input(self):
        v=-65+np.random.default_rng(42).normal(0,2,64)
        counts,_,_=simulate(v,np.zeros((20000,64)),np.zeros((760,64)),8,2.,0.,np.linspace(25,35,8))
        self.assertEqual(int(counts.sum()),0)

    def test_more_motor_cells_preserve_the_existing_cells_input_and_noise(self):
        a=raw('baseline',42)[0]
        b=raw('n32',42)[0]
        np.testing.assert_array_equal(a,b[:,:8])


if __name__=='__main__':unittest.main()
