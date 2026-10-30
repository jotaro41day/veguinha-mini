"""Local binocular energy neurons; only image contrast enters, never scene depth."""
import numpy as np
from brian2 import NeuronGroup,Synapses,ms
from brian2.codegen.runtime.numpy_rt import NumpyCodeObject
from .network import EmittedSpikeMonitor


class BinocularCircuit:
    rows,cols,disparities,features=3,8,13,45

    def __init__(self,config,clock,network,model=None):
        self.config=config
        self.disparities=max(12,int(config.vision_width*.2))+1
        self.nexc=self.rows*self.cols*self.disparities
        ninh=3*self.rows*self.cols
        self.focal_px=config.vision_width/(2*np.tan(np.radians(config.vision_fov_deg/2)))
        self.baseline_m=config.vision_baseline_m
        fields='\n'.join(f'{side}{k}:1' for side in ('L','R') for k in range(self.features))
        # Cache short dendritic sums: Cython's parser overflows on a long inlined sum.
        parts=[]
        for first in range(0,self.features,9):
            term='+'.join(f'L{k}*R{k}' for k in range(first,min(first+9,self.features)))
            parts.append(f'match{len(parts)}={term}:1 (constant over dt)')
        energy='+'.join(f'match{k}' for k in range(len(parts)))
        dendrites='\n'.join(parts)
        equations=f'''
        dv/dt=(.04*v*v+5*v+140-u+I_base+I_syn+binocular)/ms:1
        du/dt=a*(.2*v-u)/ms:1
        dI_syn/dt=-I_syn/(20*ms):1
        binocular=gain*clip(({energy})-.65,0,.35):1
        a:1
        c:1
        d:1
        I_base:1
        gain:1 (shared)
        spike_gate:1
        {fields}
        {dendrites}
        '''
        g=self.group=NeuronGroup(self.nexc+ninh,equations,threshold='v>=30',reset='v=c;u+=d',
            events={'emit':'v>=30 and spike_gate>0'},method='euler',clock=clock,name='binocular_cells')
        # The 90 retinal input arrays make Cython compilation exceed the phone's time limit.
        # Only the five vectorized dendritic sums use NumPy; spikes/synapses keep the configured backend.
        g.subexpression_updater.codeobj_class=NumpyCodeObject
        rng=np.random.default_rng(config.seed+17491)
        g.a[:self.nexc]=.02*rng.uniform(.99,1.01,self.nexc);g.a[self.nexc:]=.1
        g.c=-65;g.d[:self.nexc]=8;g.d[self.nexc:]=2;g.v=-65;g.u=-13
        g.I_base[self.nexc:]=-2;g.gain=config.binocular_current_gain;g.spike_gate=1
        exc=np.arange(self.nexc);inh=self.nexc+np.arange(ninh)
        source=np.r_[np.repeat(exc,3),np.repeat(inh,self.disparities)]
        target=np.r_[self.nexc+np.repeat(exc//self.disparities,3)*3+np.tile(np.arange(3),self.nexc),
            np.repeat(np.arange(ninh)//3,self.disparities)*self.disparities+np.tile(np.arange(self.disparities),ninh)]
        local=Synapses(g,g,'w:1',on_pre='I_syn_post+=w',on_event='emit',clock=clock,name='binocular_local')
        local.connect(i=source,j=target);local.w=np.r_[np.full(self.nexc*3,1.5),np.full(ninh*self.disparities,-.8)];local.delay=1*ms
        self.monitor=EmittedSpikeMonitor(g,'emit',record=False,name='binocular_spikes')
        self.objects=[g,local,self.monitor];self.synapses=[local]
        if model is not None:
            cx=model.regions['cx'];targets=np.array_split(cx[~model.inhibitory[cx]],2)
            loc=exc//self.disparities;pref=exc%self.disparities/(self.disparities-1)
            post=np.r_[targets[0][loc%len(targets[0])],targets[1][loc%len(targets[1])]]
            projection=Synapses(g,model.group,'w:1',on_pre='I_syn_post+=w',on_event='emit',clock=clock,name='binocular_to_cx')
            projection.connect(i=np.tile(exc,2),j=post)
            degree=np.bincount(post,minlength=len(model.group))
            projection.w=config.binocular_projection_gain*np.r_[pref,1-pref]/np.sqrt(degree[post]);projection.delay=2*ms
            self.objects.append(projection);self.synapses.append(projection)
        network.add(*self.objects)
        self.patch_stride=1
        ys=np.linspace(self.patch_stride,config.vision_height-1-self.patch_stride,self.rows).round().astype(int)
        xs=np.linspace(2*self.patch_stride+1,config.vision_width-1-2*self.patch_stride,self.cols).round().astype(int)
        yy,xx,dd=np.meshgrid(ys,xs,np.arange(self.disparities),indexing='ij')
        self.y=yy.ravel();self.x=xx.ravel();self.rx=(xx-dd).ravel()
        self.dy,self.dx=np.meshgrid(np.arange(-1,2),np.arange(-2,3),indexing='ij')
        self.dy=self.dy.ravel()*self.patch_stride;self.dx=self.dx.ravel()*self.patch_stride
        self.previous=np.zeros(len(g),np.int64);self.rates=np.zeros(self.nexc)
        self.last_image_seq=None;self.sequence=0

    def set_images(self,images,valid_rectangle=None):
        images=np.asarray(images)
        self.images=images.copy();self.consistency_key=None
        if images.shape!=(2,self.config.vision_height,self.config.vision_width,3):
            raise ValueError('imagens binoculares incompatíveis')
        # Retinal compression: strongest local contrast in each field, no object labels.
        gray=images[0].astype(float).mean(2)/255
        saliency=np.zeros_like(gray);saliency[:,1:]+=abs(np.diff(gray,axis=1));saliency[1:]+=abs(np.diff(gray,axis=0))
        self.valid_rectangle=valid_rectangle
        if valid_rectangle is not None:
            x0,y0,x1,y1=valid_rectangle
            native=np.zeros_like(gray,dtype=bool);native[y0+1:y1-1,x0+2:x1-2]=True
            saliency[~native]=0
        yy,xx=[],[]
        for ys in np.array_split(np.arange(self.patch_stride,self.config.vision_height-self.patch_stride),self.rows):
            for xs in np.array_split(np.arange(2*self.patch_stride,self.config.vision_width-2*self.patch_stride),self.cols):
                y,x=np.unravel_index(np.argmax(saliency[np.ix_(ys,xs)]),(len(ys),len(xs)))
                yy.append(ys[y]);xx.append(xs[x])
        self.y=np.repeat(yy,self.disparities);self.x=np.repeat(xx,self.disparities)
        self.rx=self.x-np.tile(np.arange(self.disparities),self.rows*self.cols)
        valid=(self.rx>=2*self.patch_stride)&(self.rx<self.config.vision_width-2*self.patch_stride)
        if valid_rectangle is not None:
            valid&=(self.x>=x0+2)&(self.x<x1-2)&(self.rx>=x0+2)&(self.rx<x1-2)&(self.y>=y0+1)&(self.y<y1-1)
        self.input_vectors={}
        for eye,x,side in ((0,self.x,'L'),(1,np.clip(self.rx,2*self.patch_stride,self.config.vision_width-1-2*self.patch_stride),'R')):
            patch=images[eye,self.y[:,None]+self.dy,x[:,None]+self.dx].astype(float)/255
            values=(patch-patch.mean(axis=(1,2),keepdims=True)).reshape(self.nexc,self.features)
            values/=np.sqrt((values*values).sum(1,keepdims=True)+1e-9)
            texture=((patch-patch.mean(axis=1,keepdims=True))**2).sum(axis=(1,2))
            values[(~valid)|(texture<1e-4)]=0
            self.input_vectors[side]=values
            for k in range(self.features):
                array=self.group.variables[f'{side}{k}'].get_value();array[:self.nexc]=values[:,k];array[self.nexc:]=0

    def _consistency(self,best):
        # Retinal quality check only: it rejects correspondences, never supplies depth.
        key=tuple(best.ravel())
        if getattr(self,'consistency_key',None)==key:return self.consistency_value
        y=self.y[::self.disparities];left_x=self.x[::self.disparities]
        reference_x=left_x-best.ravel();d=np.arange(self.disparities)
        candidates=reference_x[:,None]+d
        valid=(candidates>=2)&(candidates<self.config.vision_width-2)
        if getattr(self,'valid_rectangle',None) is not None:
            x0,y0,x1,y1=self.valid_rectangle
            valid&=(candidates>=x0+2)&(candidates<x1-2)&(y[:,None]>=y0+1)&(y[:,None]<y1-1)
        centers=np.clip(candidates,2,self.config.vision_width-3)
        patch=self.images[0,y[:,None,None]+self.dy[None,None,:],centers[:,:,None]+self.dx[None,None,:]].astype(float)/255
        vectors=(patch-patch.mean(axis=(2,3),keepdims=True)).reshape(len(y),self.disparities,self.features)
        vectors/=np.sqrt((vectors*vectors).sum(2,keepdims=True)+1e-9)
        texture=((patch-patch.mean(axis=2,keepdims=True))**2).sum(axis=(2,3))
        vectors[(~valid)|(texture<1e-4)]=0
        right=self.input_vectors['R'].reshape(len(y),self.disparities,self.features)[np.arange(len(y)),best.ravel()]
        correlation=(vectors*right[:,None,:]).sum(2)
        energy=np.where(valid,self.config.binocular_current_gain*np.clip(correlation-.65,0,.35),0)
        reverse=energy.argmax(1);peak=energy.max(1)
        other=np.where(abs(d[None,:]-reverse[:,None])<=2,0,energy).max(1)
        confidence=np.clip((peak-other)/np.maximum(peak,1e-9),0,1)
        confidence[abs(reverse-best.ravel())>1]=0
        self.consistency_key=key;self.consistency_value=confidence.reshape(self.rows,self.cols)
        return self.consistency_value

    def reset_readout(self):
        self.previous=np.array(self.monitor.count[:],dtype=np.int64);self.rates[:]=0
        self.last_image_seq=None

    def read(self,dt):
        count=np.array(self.monitor.count[:],dtype=np.int64)
        observed=(count-self.previous)[:self.nexc]/dt;self.previous=count
        self.rates+=(1-np.exp(-dt/.12))*(observed-self.rates)
        rates=self.rates.reshape(self.rows,self.cols,self.disparities)
        best=rates.argmax(2);peak=rates.max(2)
        candidates=np.arange(self.disparities)[None,None,:]
        neighborhood=abs(candidates-best[:,:,None])<=2
        competitor=np.where(neighborhood,0,rates).max(2)
        confidence=(peak-competitor)/np.maximum(peak,1e-9)
        weights=np.where(neighborhood,rates**3,0)
        disparity=np.divide((weights*candidates).sum(2),weights.sum(2),out=np.zeros_like(peak),where=weights.sum(2)>0)
        evidence=self.config.binocular_current_gain*np.clip((self.input_vectors['L']*self.input_vectors['R']).sum(1).reshape(rates.shape)-.65,0,.35)
        chosen=np.take_along_axis(evidence,best[:,:,None],axis=2)[:,:,0]
        other_evidence=np.where(neighborhood,0,evidence).max(2)
        input_confidence=np.clip((chosen-other_evidence)/np.maximum(chosen,1e-9),0,1)
        confidence=np.minimum(confidence,np.minimum(input_confidence,self._consistency(best)))
        valid=(peak>=5)&(confidence>=.25)&(best>0)&(disparity>.5)
        depth=np.divide(self.focal_px*self.baseline_m,disparity,out=np.zeros_like(confidence),where=disparity>0)
        axial=[[round(float(depth[r,c]),3) if valid[r,c] else None for c in range(self.cols)] for r in range(self.rows)]
        u=(self.x[::self.disparities].reshape(self.rows,self.cols)-(self.config.vision_width-1)/2)/self.focal_px
        v=((self.config.vision_height-1)/2-self.y[::self.disparities].reshape(self.rows,self.cols))/self.focal_px
        ranges=np.sqrt(depth**2+(u*depth-self.baseline_m/2)**2+(v*depth)**2)
        distance=[[round(float(ranges[r,c]),3) if valid[r,c] else None for c in range(self.cols)] for r in range(self.rows)]
        self.sequence+=1
        return dict(rows=self.rows,cols=self.cols,seq=self.sequence,camera_seq=self.last_image_seq[1] if isinstance(self.last_image_seq,tuple) else None,disparity_px=disparity.round(3).tolist(),
            confidence=np.where(valid,confidence,0).round(3).tolist(),distance_m=distance,axial_depth_m=axial,
            mean_hz=round(float(self.rates.mean()),3),valid_fraction=round(float(valid.mean()),3))
