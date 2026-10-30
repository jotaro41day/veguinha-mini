import json
import os
from pathlib import Path
import tempfile
import unittest
from veguinha_mini.progress_server import read_status


class ProgressTests(unittest.TestCase):
    def test_reads_finished_tests_and_survives_partial_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            (p/'one.json').write_text(json.dumps({'profile':'gated','seed':42,'joints':{'coxa':{'rhythmic':True}}}))
            (p/'broken.json').write_text('{')
            (p/'unit_tests.log').write_text('Ran 3 tests in 0.02s\n\nOK\n')
            data=read_status(p)
            self.assertEqual(len(data['results']),1)
            self.assertEqual(data['unit_tests']['passed'],3)
            self.assertEqual(data['unit_tests']['state'],'passed')

    def test_stale_progress_does_not_look_like_a_running_test(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            (p/'progress.json').write_text(json.dumps({'pid':-1,'state':'executando','simulated_s':5}))
            self.assertEqual(read_status(p)['progress']['state'],'interrompido')
            (p/'progress.json').write_text(json.dumps({'pid':os.getpid(),'state':'executando'}))
            self.assertTrue(read_status(p)['progress']['alive'])


if __name__=='__main__':unittest.main()
