import importlib.util
import unittest

import numpy as np


class ScaffoldEvalTests(unittest.TestCase):
    def test_antagonist_score_distinguishes_alternation_from_coactivation(self):
        self.assertIsNotNone(importlib.util.find_spec("veguinha_mini.scaffold_eval"))
        from veguinha_mini.scaffold_eval import antagonist_correlation

        blocks = np.tile(np.r_[np.ones(10), np.zeros(10)], 3)
        alternating = np.column_stack((blocks, 1 - blocks))
        coactive = np.column_stack((blocks, blocks))
        self.assertLess(antagonist_correlation(alternating), -.9)
        self.assertGreater(antagonist_correlation(coactive), .9)
        self.assertEqual(antagonist_correlation(np.ones((30, 2))), 0.0)


if __name__ == "__main__":
    unittest.main()
