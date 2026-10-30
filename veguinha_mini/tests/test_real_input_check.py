import inspect
import unittest
import numpy as np
from veguinha_mini.tools.leg_reflex_check import run

class RealInputTests(unittest.TestCase):
    def test_optional_instrumentation_preserves_default(self):
        args=inspect.signature(run).parameters
        self.assertFalse(args['feedback'].default)
        self.assertFalse(args['record_events'].default)

    def test_event_binning_conserves_events_at_block_boundaries(self):
        from veguinha_mini.tools.real_input_check import bin_events
        a=bin_events(np.array([0,.0195,.02,19.9995]),np.array([1,1,2,1]),3)
        self.assertEqual(a.sum(),4)
        self.assertEqual(a[0,1],2)
        self.assertEqual(a[1,2],1)
        self.assertEqual(a[-1,1],1)

    def test_partial_run_cannot_be_committed_as_twenty_seconds(self):
        from veguinha_mini.tools.leg_reflex_check import require_complete
        with self.assertRaises(RuntimeError): require_complete(18,900)
        with self.assertRaises(RuntimeError): require_complete(20,999)
        require_complete(20,1000)

    def test_strict_stop_threshold(self):
        from veguinha_mini.tools.real_input_check import passes
        self.assertFalse(passes(24,30))
        self.assertTrue(passes(25,30))

if __name__=='__main__': unittest.main()
