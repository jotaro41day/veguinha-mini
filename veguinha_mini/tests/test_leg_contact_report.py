import unittest
import numpy as np
from veguinha_mini.tools.leg_contact_report import cycle_variability


class ContactReportTests(unittest.TestCase):
    def test_cycle_boundaries_do_not_cut_repeatable_neural_bursts(self):
        c=np.zeros((1000,196),dtype=int)
        for start in range(0,980,20):
            for joint in range(3):
                for action in (0,1):
                    for offset in (3,4,5):
                        t=start+offset+10*action
                        c[t,48+24*joint+12*action:48+24*joint+12*(action+1)]=1
                        c[t,16*joint+8*action:16*joint+8*(action+1)]=1
        r=cycle_variability(c)
        self.assertGreater(r['cycles'],15)
        np.testing.assert_allclose(r['premotor_cv'],0,atol=1e-12)
        np.testing.assert_allclose(r['motor_cv'],0,atol=1e-12)
        self.assertEqual(cycle_variability(np.zeros_like(c))['cycles'],0)


if __name__=='__main__':unittest.main()
