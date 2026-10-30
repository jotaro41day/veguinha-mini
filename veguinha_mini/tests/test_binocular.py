import unittest
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from brian2 import Clock,Network,ms,prefs,start_scope
from veguinha_mini.config import Config


class BinocularTests(unittest.TestCase):
    def test_calibrated_parallel_eyes_reproject_the_same_point_on_the_same_row(self):
        from veguinha_mini.perception import ExternalSenses
        c=Config(vision_enabled=True,binocular_enabled=True,vision_pitch_deg=0)
        s=ExternalSenses(c)
        rays=s.directions
        np.testing.assert_array_equal(rays[0],rays[1])
        f=s.focal_px;point=np.array([1.25,0,.07]);coords=[]
        for origin in s.eye_origins:
            delta=point-origin
            coords.append([(c.vision_width-1)/2-f*delta[1]/delta[0],(c.vision_height-1)/2-f*delta[2]/delta[0]])
        self.assertAlmostEqual(coords[0][1],coords[1][1])
        self.assertAlmostEqual(f*s.baseline_m/(coords[0][0]-coords[1][0]),1.)

    def test_images_are_fused_by_spikes_and_blank_or_monocular_input_has_no_depth(self):
        self.assertTrue(hasattr(Config(),'binocular_enabled'),'missing binocular neural circuit configuration')
        from veguinha_mini.binocular import BinocularCircuit
        start_scope();prefs.codegen.target='numpy';c=Config(vision_enabled=True,binocular_enabled=True,vision_pitch_deg=0)
        clock=Clock(dt=.5*ms);net=Network();circuit=BinocularCircuit(c,clock,net)
        rng=np.random.default_rng(1024);left=rng.integers(0,256,(24,48,3),dtype=np.uint8)
        frames=np.zeros((2,24,48,3),np.uint8);frames[0]=left;frames[1,:,:-4]=left[:,4:]
        circuit.set_images(frames);net.run(400*ms);result=circuit.read(.4)
        disparity=np.asarray(result['disparity_px']);valid=np.asarray(result['confidence'])>.2
        self.assertGreater(valid.sum(),10)
        self.assertLess(np.median(abs(disparity[valid]-4)),.51)
        depth=np.array([[np.nan if v is None else v for v in row] for row in result['axial_depth_m']])
        self.assertAlmostEqual(float(np.nanmedian(depth)),1.08,delta=.15)
        self.assertGreater(np.asarray(circuit.monitor.count[:]).sum(),0)
        for image in (np.full_like(frames,127),np.broadcast_to(np.array([36,48,64],dtype=np.uint8),frames.shape),np.stack([left,np.zeros_like(left)])):
            circuit.reset_readout();circuit.set_images(image);net.run(400*ms);result=circuit.read(.4)
            self.assertFalse(any(v is not None for row in result['axial_depth_m'] for v in row))

    def test_binocular_spikes_reach_brain_and_ablation_stops_the_projection(self):
        from brian2 import NeuronGroup
        from veguinha_mini.binocular import BinocularCircuit
        start_scope();prefs.codegen.target='numpy';clock=Clock(dt=1*ms)
        cortex=NeuronGroup(16,'dI_syn/dt=-I_syn/(10*ms):1',method='euler',clock=clock)
        model=SimpleNamespace(group=cortex,regions={'cx':np.arange(16)},inhibitory=np.zeros(16,bool))
        net=Network(cortex);circuit=BinocularCircuit(Config(vision_enabled=True,binocular_enabled=True),clock,net,model)
        rng=np.random.default_rng(55);left=rng.integers(0,256,(24,48,3),dtype=np.uint8)
        pair=np.zeros((2,24,48,3),np.uint8);pair[0]=left;pair[1,:,:-4]=left[:,4:]
        circuit.set_images(pair);net.run(400*ms,namespace={})
        self.assertGreater(float(np.max(cortex.I_syn[:])),.01)
        for synapse in circuit.synapses:
            sources=np.asarray(synapse.i[:]);weights=np.asarray(synapse.w[:])
            self.assertTrue((weights[sources<circuit.nexc]>=0).all())
            self.assertTrue((weights[sources>=circuit.nexc]<=0).all())
        circuit.read(.4)
        circuit.group.spike_gate=0;net.run(400*ms,namespace={});state=circuit.read(.4)
        self.assertLess(float(np.max(cortex.I_syn[:])),1e-8)
        self.assertFalse(any(v is not None for row in state['distance_m'] for v in row))

    def test_repeated_patterns_do_not_produce_false_confident_depth(self):
        from veguinha_mini.binocular import BinocularCircuit
        start_scope();prefs.codegen.target='numpy';net=Network()
        circuit=BinocularCircuit(Config(vision_enabled=True,binocular_enabled=True),Clock(dt=1*ms),net)
        columns=np.tile([20,20,220,220],12).astype(np.uint8)
        image=np.tile(columns[None,:,None],(24,1,3));circuit.set_images(np.stack([image,image]))
        net.run(400*ms,namespace={});state=circuit.read(.4)
        self.assertFalse(any(v is not None for row in state['distance_m'] for v in row))

if __name__=='__main__':unittest.main()
