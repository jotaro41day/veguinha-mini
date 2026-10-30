import unittest

import numpy as np

from veguinha_mini.verification import kenyon_fraction_per_50ms


class SparsityTests(unittest.TestCase):
    def test_counts_unique_kenyon_cells_in_each_bin(self):
        times = np.array([1, 5, 15, 60, 65, 70, 75], dtype=float)
        neurons = np.array([10, 10, 11, 12, 12, 0, 0], dtype=int)
        fraction = kenyon_fraction_per_50ms(times, neurons, np.arange(10, 20), 0, 100)
        self.assertAlmostEqual(fraction, 0.15)


if __name__ == "__main__":
    unittest.main()
