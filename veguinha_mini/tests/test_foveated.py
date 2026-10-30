import unittest
from dataclasses import replace
import numpy as np,mujoco
from veguinha_mini.config import Config
from veguinha_mini.perception import ExternalSenses
from veguinha_mini.mujoco_body import MujocoBody
from veguinha_mini.world import World


class FoveatedTests(unittest.TestCase):
    def config(self):
        return Config(vision_enabled=True,binocular_enabled=True,optical_flow_enabled=True,vision_width=320,vision_height=180,vision_interval_ms=200,foveated_enabled=True,vision_periphery_ms=100)

    def test_peripheral_transition_samples_intermediate_detail_without_blocks(self):
        s=ExternalSenses(self.config());samples=[]
        def ramp(body,world,eye,ys,xs):
            samples.append((ys.copy(),xs.copy()))
            return np.column_stack([xs//2,ys,np.full(len(xs),80)]).astype(np.uint8)
        s._render_eye=ramp
        s.scan(None,None)
        # A smooth real signal must not acquire four-pixel stair steps.
        self.assertLessEqual(np.abs(np.diff(s.rgb[0,10,:,0].astype(int))).max(),1)
        self.assertTrue(any(np.any((y==23)&(x==41)) for y,x in samples))
        layers=s.snapshot()['camera']['sampling']
        self.assertEqual([x['step_px'] for x in layers],[1,2,4])

    def test_native_center_is_identical_while_sampling_fewer_rays(self):
        body=MujocoBody(variant='mk1');c=self.config();world=World()
        a=ExternalSenses(replace(c,foveated_enabled=False));b=ExternalSenses(c)
        for s in (a,b):s.bind_model(body.model);s.scan(body,world)
        x0,y0,x1,y1=b.fovea
        np.testing.assert_array_equal(a.rgb[:,y0:y1,x0:x1],b.rgb[:,y0:y1,x0:x1])
        self.assertLess(b.last_ray_count,a.last_ray_count*.5)
        self.assertEqual(b.snapshot()['camera']['detail_period_ms'],200)

    def test_peripheral_and_native_clocks_do_not_fabricate_double_speed(self):
        body=MujocoBody(variant='mk1');s=ExternalSenses(self.config());world=World(food_enabled=False);s.bind_model(body.model);s.scan(body,world)
        first=s.rgb.copy();seq=s.sequence;native=s.detail_sequence
        body.data.qpos[1]+=.03;mujoco.mj_forward(body.model,body.data)
        for _ in range(5):s.scan(body,world,.02)
        self.assertEqual(s.sequence,seq+1);self.assertEqual(s.detail_sequence,native)
        x0,y0,x1,y1=s.fovea;np.testing.assert_array_equal(first[:,y0:y1,x0:x1],s.rgb[:,y0:y1,x0:x1])
        self.assertEqual(s.optical.previous.shape,(2,45,80))
        for _ in range(5):s.scan(body,world,.02)
        self.assertEqual(s.detail_sequence,native+1);self.assertEqual(s.sequence,seq+2)
        self.assertFalse(np.array_equal(first[:,y0:y1,x0:x1],s.rgb[:,y0:y1,x0:x1]))

    def test_stereo_never_uses_upsampled_peripheral_pixels_as_native_evidence(self):
        from brian2 import start_scope,prefs,Clock,Network,ms
        from veguinha_mini.binocular import BinocularCircuit
        start_scope();prefs.codegen.target='numpy';c=self.config();net=Network();brain=BinocularCircuit(c,Clock(dt=1*ms),net)
        rng=np.random.default_rng(62);left=rng.integers(0,256,(180,320,3),np.uint8);images=np.stack([left,np.roll(left,-12,axis=1)])
        rect=(80,45,240,135);brain.set_images(images,valid_rectangle=rect)
        allowed=(brain.x>=82)&(brain.x<238)&(brain.rx>=82)&(brain.rx<238)&(brain.y>=46)&(brain.y<134)
        self.assertFalse(brain.input_vectors['L'][~allowed].any());self.assertFalse(brain.input_vectors['R'][~allowed].any())
        net.run(400*ms,namespace={});r=brain.read(.4)
        estimates=np.asarray(r['disparity_px']);valid=np.asarray(r['confidence'])>.25
        self.assertTrue(valid.any());self.assertLess(abs(float(np.median(estimates[valid]))-12),1)

if __name__=='__main__':unittest.main()
