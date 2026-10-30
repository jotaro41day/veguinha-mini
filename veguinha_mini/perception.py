"""Physical sensory transduction and nutrient transfer; never chooses leg actions."""
import base64
import numpy as np
import mujoco
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from threading import local
from types import SimpleNamespace

_ray_local=local()

@lru_cache(maxsize=2)
def _ray_pool(workers):
    return ThreadPoolExecutor(max_workers=workers,thread_name_prefix='veguinha-eye')


def _interpolate_samples(values,ys,xs,height,width):
    """Bilinear reconstruction at the actual ray coordinates, without pixel blocks."""
    def weights(samples,size):
        coordinates=np.clip(np.arange(size),samples[0],samples[-1])
        lower=np.clip(np.searchsorted(samples,coordinates,side='right')-1,0,len(samples)-2)
        fraction=(coordinates-samples[lower])/(samples[lower+1]-samples[lower])
        return lower,fraction
    ix,wx=weights(xs,width);iy,wy=weights(ys,height)
    horizontal=values[:,:,ix]*(1-wx)[None,None,:,None]+values[:,:,ix+1]*wx[None,None,:,None]
    return horizontal[:,iy]*(1-wy)[None,:,None,None]+horizontal[:,iy+1]*wy[None,:,None,None]


@lru_cache(maxsize=4)
def _foveated_layout(width,height):
    x0,y0,x1,y1=width//4,height//4,width-width//4,height-height//4
    yy,xx=np.meshgrid(np.arange(height),np.arange(width),indexing='ij')
    eccentricity=np.maximum(abs((xx+.5-width/2)/(width/4)),abs((yy+.5-height/2)/(height/4)))
    def fade(start,end):
        t=np.clip((eccentricity-start)/(end-start),0,1)
        return (1-t*t*(3-2*t))[None,:,:,None]
    # A small native overlap lets detail fade OUTSIDE the stereo-valid rectangle.
    pad_x=max(1,int(np.ceil(width/64)));pad_y=max(1,int(np.ceil(height/64)))
    detail=(max(0,x0-pad_x),max(0,y0-pad_y),min(width,x1+pad_x),min(height,y1+pad_y))
    my=np.arange(height//8+1,height-height//8,2)
    mx=np.arange(width//8+1,width-width//8,2)
    gy,gx=np.meshgrid(my,mx,indexing='ij')
    ring=~((gx>=x0+2)&(gx<x1-2)&(gy>=y0+2)&(gy<y1-2))
    return detail,my,mx,gy,gx,ring,fade(1.,1.0625),fade(1.25,1.7)

def _render_chunk(args):
    senses,body,world,eye,ys,xs=args
    # mj_multiRay writes scratch buffers in mjData: never share them across eyes.
    if getattr(_ray_local,'model',None) is not body.model:
        _ray_local.model=body.model;_ray_local.data=mujoco.MjData(body.model)
    mujoco.mj_copyData(_ray_local.data,body.model,body.data)
    snapshot=SimpleNamespace(model=body.model,data=_ray_local.data,body_id=body.body_id,floor_id=body.floor_id)
    return senses._render_eye(snapshot,world,eye,ys,xs)


class ExternalSenses:
    """Coarse compound-eye ray samples and force-sensitive nociceptors."""
    def __init__(self,config):
        self.format_version=5 if config.foveated_enabled else 4;self.config=config
        self.projection='rectilinear' if config.binocular_enabled else 'panoramic'
        self.binocular=None
        self.optical=None
        self.retina=np.zeros((2,8,4))
        self.pain=np.zeros(2);self.wall_peak_n=np.zeros(2);self.wall_impulse_ns=0.
        self.tactile=np.zeros((9,3));self.tactile_previous=np.zeros(9)
        self.baseline_m=config.vision_baseline_m
        self.eye_origins=np.array([[.25,self.baseline_m/2,.07],[.25,-self.baseline_m/2,.07]])
        width,height=config.vision_width,config.vision_height
        self.rgb=np.zeros((2,height,width,3),dtype=np.uint8)
        self.fovea=(width//4,height//4,width-width//4,height-height//4) if config.foveated_enabled else None
        self.detail_sequence=0;self.detail_age_s=0.;self.last_ray_count=0
        self.center_rgb=None
        self.previous_gray=None;self.sequence=0;self.age_s=0.;self.last_world=None
        self.expectation=None;self.novelty=0.
        self.encoded=['','']
        self.focal_px=width/(2*np.tan(np.radians(config.vision_fov_deg/2)))
        yy,xx=np.meshgrid(np.arange(height),np.arange(width),indexing='ij')
        if config.binocular_enabled:
            pitch=np.radians(config.vision_pitch_deg)
            forward=np.array([np.cos(pitch),0,np.sin(pitch)])
            up=np.array([-np.sin(pitch),0,np.cos(pitch)])
            rays=forward+((xx-(width-1)/2)/self.focal_px)[:,:,None]*np.array([0,-1,0])+(((height-1)/2-yy)/self.focal_px)[:,:,None]*up
            rays/=np.linalg.norm(rays,axis=2,keepdims=True)
            self.directions=np.stack([rays,rays])
        else:
            pitch=np.radians(20-(yy+.5)*80/height)
            yaw=np.radians(-15+(xx+.5)*180/width)
            self.directions=np.array([np.stack([np.cos(pitch)*np.cos(yaw),sign*np.cos(pitch)*np.sin(yaw),np.sin(pitch)],axis=2) for sign in (1,-1)])
        self.geom_id=np.array([-1],dtype=np.int32)
        self.groups=np.array([1,0,int(config.binocular_enabled),0,0,0],dtype=np.uint8)

    def bind_model(self,model):
        if self.config.vision_enabled:
            # Group labels filter sensor rays; they do not affect mass/contact.
            moving=model.geom_bodyid!=0
            model.geom_group[moving]=2
            if self.config.binocular_enabled:
                model.geom_group[moving&((model.geom_contype!=0)|(model.geom_conaffinity!=0))]=3

    def begin_block(self):
        self.wall_peak_n[:]=0;self.wall_impulse_ns=0.

    def initialize_tactile(self,body):
        from .somatic import tactile_sample
        force=tactile_sample(body)
        self.tactile=np.column_stack([force,np.zeros(9)])
        self.tactile_previous=force[:,0].copy()

    def transduce_force(self,forces,dt):
        forces=np.asarray(forces,float);total=float(forces.sum())
        stimulus=np.clip((total-self.config.pain_force_threshold_n)/self.config.pain_force_scale_n,0,1)
        sides=forces/max(total,1e-12)
        self.pain=np.maximum(self.pain*np.exp(-dt/self.config.pain_decay_s),stimulus*sides)
        self.wall_peak_n=np.maximum(self.wall_peak_n,forces)
        self.wall_impulse_ns+=total*dt

    def observe(self,body,dt):
        if self.config.somatic_enabled:
            from .somatic import tactile_sample
            if not hasattr(self,'tactile'):self.tactile=np.zeros((9,3));self.tactile_previous=np.zeros(9)
            force=tactile_sample(body)
            onset=np.maximum(force[:,0]-self.tactile_previous,0)
            self.tactile_previous=force[:,0].copy()
            target=np.column_stack([force,onset])
            self.tactile+=-np.expm1(-dt/.02)*(target-self.tactile)
        if not self.config.nociception_enabled:return
        force=np.zeros(6);sides=np.zeros(2)
        lateral=body.data.xmat[body.body_id].reshape(3,3)[:,1]
        for i,contact in enumerate(body.data.contact):
            if int(contact.geom1) not in body.wall_ids and int(contact.geom2) not in body.wall_ids:continue
            mujoco.mj_contactForce(body.model,body.data,i,force)
            left=.5*(1+np.tanh(float((contact.pos-body.data.xpos[body.body_id])@lateral)/.05))
            sides+=max(0.,force[0])*np.array([left,1-left])
        self.transduce_force(sides,dt)

    def scan(self,body,world,dt=None):
        if not self.config.vision_enabled:return
        signature=(world.odor_x,world.odor_y,world.food_enabled and world.food_energy>0) if world else None
        changed=signature!=self.last_world
        interval=self.config.vision_periphery_ms if self.config.foveated_enabled else self.config.vision_interval_ms
        if dt is not None:
            self.age_s+=dt
            self.detail_age_s=getattr(self,'detail_age_s',0.)+dt
            if self.sequence and not changed and self.age_s+1e-9<interval/1000:return
        sample_dt=self.age_s if self.age_s>0 else None
        self.age_s=0.;self.last_world=signature;self.sequence+=1
        self.last_ray_count=0
        if self.config.foveated_enabled:
            h,w=self.rgb.shape[1:3];ys=np.arange(2,h,4);xs=np.arange(2,w,4)
            yy,xx=np.meshgrid(ys,xs,indexing='ij')
            x0,y0,x1,y1=self.fovea
            detail,my,mx,gy,gx,ring,detail_alpha,middle_alpha=_foveated_layout(w,h)
            # Only the surrounding ring needs extra rays; its unsampled interior
            # is completely covered by native detail, never used for stereo.
            # Batch both grids so the existing worker pool can share the work.
            samples=self._render_pair(body,world,np.concatenate([yy.ravel(),gy[ring]]),np.concatenate([xx.ravel(),gx[ring]]))
            coarse=samples[:,:yy.size].reshape(2,h//4,w//4,3)
            middle=np.zeros((2,len(my),len(mx),3),dtype=np.uint8)
            middle[:,ring]=samples[:,yy.size:]
            detail_due=(not hasattr(self,'detail_patch_rgb') or not self.detail_sequence or self.detail_age_s+1e-9>=self.config.vision_interval_ms/1000 or changed or dt is None)
            if detail_due:
                dx0,dy0,dx1,dy1=detail
                cy,cx=np.meshgrid(np.arange(dy0,dy1),np.arange(dx0,dx1),indexing='ij')
                self.detail_patch_rgb=self._render_pair(body,world,cy.ravel(),cx.ravel()).reshape(2,dy1-dy0,dx1-dx0,3)
                self.center_rgb=self.detail_patch_rgb[:,y0-dy0:y1-dy0,x0-dx0:x1-dx0].copy()
                self.detail_sequence+=1;self.detail_age_s=0.
            outer=_interpolate_samples(coarse,ys,xs,h,w)
            surround=_interpolate_samples(middle,my,mx,h,w)
            image=outer*(1-middle_alpha)+surround*middle_alpha
            dx0,dy0,dx1,dy1=detail
            alpha=detail_alpha[:,dy0:dy1,dx0:dx1]
            image[:,dy0:dy1,dx0:dx1]*=1-alpha
            image[:,dy0:dy1,dx0:dx1]+=self.detail_patch_rgb*alpha
            self.rgb[:]=np.rint(np.clip(image,0,255)).astype(np.uint8)
            self.rgb[:,y0:y1,x0:x1]=self.center_rgb
            flow_gray=coarse.astype(float).mean(3)/255;flow_focal=self.focal_px/4
        else:
            yy,xx=np.meshgrid(np.arange(self.config.vision_height),np.arange(self.config.vision_width),indexing='ij')
            self.rgb[:]=self._render_pair(body,world,yy.ravel(),xx.ravel()).reshape(self.rgb.shape)
            self.detail_sequence=self.sequence;self.detail_age_s=0.
            flow_gray=self.rgb.astype(float).mean(3)/255;flow_focal=self.focal_px
        image=self.rgb.astype(float)/255
        gray=image.mean(3)
        if self.config.optical_flow_enabled:
            from .optical_flow import OpticalFlow
            if getattr(self,'optical',None) is None:self.optical=OpticalFlow()
            self.optical.update(flow_gray,sample_dt,flow_focal)
        if self.expectation is None:self.expectation=np.zeros_like(gray)
        self.novelty=float(np.clip(10*np.abs(gray-self.expectation).mean(),0,1))
        self.expectation+=(1-np.exp(-(sample_dt if sample_dt is not None else interval/1000)/2))*(gray-self.expectation)
        edges=np.zeros_like(gray);edges[:,:,1:]=abs(np.diff(gray,axis=2));edges[:,1:,:]+=abs(np.diff(gray,axis=1))
        opponent=np.clip(image[:,:,:,0]-image[:,:,:,2],0,1)
        motion=np.zeros_like(gray)
        if self.previous_gray is not None:
            motion[:,:,1:]=self.previous_gray[:,:,:-1]*gray[:,:,1:]-self.previous_gray[:,:,1:]*gray[:,:,:-1]
        self.previous_gray=gray.copy()
        for eye in range(2):
            for sector,columns in enumerate(np.array_split(np.arange(self.config.vision_width),8)):
                self.retina[eye,sector]=[np.clip(5*edges[eye][:,columns].mean(),0,1),
                    np.clip(4*opponent[eye][:,columns].mean(),0,1),
                    np.clip(20*np.maximum(motion[eye][:,columns],0).mean(),0,1),
                    np.clip(20*np.maximum(-motion[eye][:,columns],0).mean(),0,1)]
        self.encoding='rgb'
        if self.config.vision_width>=160:
            from PIL import Image
            from io import BytesIO
            self.encoding='jpeg';self.encoded=[]
            for im in self.rgb:
                buffer=BytesIO();Image.fromarray(im).save(buffer,format='JPEG',quality=65)
                self.encoded.append(base64.b64encode(buffer.getvalue()).decode('ascii'))
        else:self.encoded=[base64.b64encode(im.tobytes()).decode('ascii') for im in self.rgb]

    def _render_pair(self,body,world,ys,xs):
        workers=self.config.vision_workers
        self.last_ray_count+=2*len(xs)
        if workers==1 or len(xs)<8192:
            return np.array([self._render_eye(body,world,e,ys,xs) for e in range(2)])
        chunks=np.array_split(np.arange(len(xs)),workers//2)
        jobs=[(self,body,world,e,ys[ix],xs[ix]) for e in range(2) for ix in chunks]
        frames=list(_ray_pool(workers).map(_render_chunk,jobs))
        return np.array([np.concatenate(frames[e*len(chunks):(e+1)*len(chunks)]) for e in range(2)])

    def _render_eye(self,body,world,eye,ys,xs):
        rotation=body.data.xmat[body.body_id].reshape(3,3)
        has_food=world is not None and world.food_enabled and world.food_energy>0
        count=len(xs)
        ids=np.empty(count,dtype=np.int32);dist=np.empty(count);normals=np.zeros((count,3))
        light=np.array([.3,-.4,1.]);light/=np.linalg.norm(light)
        if count:
            origin=body.data.xpos[body.body_id]+rotation@self.eye_origins[eye]
            rays=np.ascontiguousarray(self.directions[eye,ys,xs]@rotation.T)
            mujoco.mj_multiRay(body.model,body.data,origin,rays.ravel(),self.groups,True,
                body.body_id if self.config.binocular_enabled else -1,ids,dist,
                normals.ravel() if self.config.binocular_enabled else None,count,1e5)
            colors=np.tile([36.,48.,64.],(count,1));hit=ids>=0
            colors[hit]=body.model.geom_rgba[ids[hit],:3]*255
            floor=hit&(ids==body.floor_id)
            points=origin+dist[floor,None]*rays[floor]
            line=np.min(np.abs(points[:,:2]/.25-np.round(points[:,:2]/.25)),axis=1)*.25<.008
            floor_indices=np.flatnonzero(floor)
            colors[floor_indices[line]]=[60,80,90]
            if has_food:
                delta=world.food_position_m-origin;projection=rays@delta
                disc=world.food_radius_m**2-(delta@delta-projection**2)
                food_t=projection-np.sqrt(np.maximum(disc,0))
                food=(disc>=0)&(food_t>0)&((~hit)|(food_t<dist))
                colors[food]=np.array([243,188,104])*(.4+.6*np.clip(world.food_energy,0,1))
                if self.config.binocular_enabled:
                    normals[food]=(origin+food_t[food,None]*rays[food]-world.food_position_m)/world.food_radius_m
                    hit|=food
                dist[food]=food_t[food]
            if self.config.binocular_enabled:
                colors[hit]*=(.4+.6*np.clip(normals[hit]@light,0,1))[:,None]
            if world is not None:
                points=origin+dist[hit,None]*rays[hit]
                colors[hit]*=world.illumination_at(points)[:,None]
            return np.clip(colors,0,255).astype(np.uint8)

    def snapshot(self):
        from .somatic import looming_from_flow
        return dict(retina=self.retina.tolist(),pain=self.pain.tolist(),binocular=self.binocular,
                    tactile=getattr(self,'tactile',np.zeros((9,3))).tolist(),
                    looming=looming_from_flow(self.optical.result) if self.config.somatic_enabled and getattr(self,'optical',None) is not None else 0.,
                    optical_flow=self.optical.result if getattr(self,'optical',None) is not None else None,
                    novelty=self.novelty,
                    camera=dict(width=self.config.vision_width,height=self.config.vision_height,
                                eyes=self.encoded,seq=self.sequence,age_ms=round(self.age_s*1000),
                                period_ms=self.config.vision_periphery_ms if self.config.foveated_enabled else self.config.vision_interval_ms,
                                detail_period_ms=self.config.vision_interval_ms,detail_seq=getattr(self,'detail_sequence',self.sequence),
                                detail_age_ms=round(getattr(self,'detail_age_s',0)*1000),fovea=getattr(self,'fovea',None),rays=getattr(self,'last_ray_count',0),
                                sampling=[dict(region='center',step_px=1),dict(region='transition',step_px=2),dict(region='outer',step_px=4)] if self.config.foveated_enabled else [dict(region='full',step_px=1)],
                                encoding=getattr(self,'encoding','rgb'),projection=self.projection,focal_px=self.focal_px,baseline_m=self.baseline_m),
                    wall_peak_n=self.wall_peak_n.tolist(),wall_impulse_ns=self.wall_impulse_ns)


def mouth_position(body):
    # Measured mandible bounds: x .229–.376, y ±.108, z -.135–-.014 m.
    return body.data.xpos[body.body_id]+body.data.xmat[body.body_id].reshape(3,3)@np.array([.30,0,-.09])


def mouth_contact(body,world):
    if world is None or not world.food_enabled or world.food_energy<=0:return 0.
    # Oral volume fitted between the measured mandibles, expanded by food radius.
    rotation=body.data.xmat[body.body_id].reshape(3,3)
    local=rotation.T@(world.food_position_m-mouth_position(body))
    radii=np.array([.075,.095,.060])+world.food_radius_m
    return float(np.clip((1-np.linalg.norm(local/radii))/.3,0,1))


def feed(body,world,closing_activation,dt,rate):
    """Oral pressure transfers finite food only at physical mouth overlap."""
    demand=mouth_contact(body,world)*np.clip(closing_activation,0,1)*rate*dt
    world,taken=world.consume(min(float(demand),max(0.,1-body.energy)))
    body.energy=min(1.,body.energy+taken)
    return world,taken
