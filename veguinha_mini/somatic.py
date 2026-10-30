"""Chemical/contact receptors and retinal expansion; no actions or reward scalars."""
import numpy as np
import mujoco


def looming_from_flow(flow):
    if flow is None:return 0.
    velocity=np.asarray(flow['angular_rad_s'],float)
    confidence=np.asarray(flow['confidence'],float)
    h,w=confidence.shape[1:]
    y,x=np.meshgrid((np.arange(h)+.5)/h-.5,(np.arange(w)+.5)/w-.5,indexing='ij')
    # Unknown matches are not stationary pixels; fit translation only to valid flow.
    roi=(abs(x)<.35)&(y<=0)
    risk=0.
    for eye in range(2):
        valid=roi&(confidence[eye]>.1)&np.isfinite(velocity[eye]).all(axis=-1)
        # A single moving limb/patch cannot establish a coherent approaching surface.
        if valid.sum()<4 or not (x[valid]<0).any() or not (x[valid]>0).any():continue
        weights=np.clip(confidence[eye][valid],0,1);weights/=weights.sum()
        points=np.column_stack([x[valid],y[valid]]);motion=velocity[eye][valid]
        points-=np.sum(points*weights[:,None],axis=0)
        motion=motion-np.sum(motion*weights[:,None],axis=0)
        extent=float(np.sum(weights*np.sum(points*points,axis=1)))
        variation=float(np.sum(weights*np.sum(motion*motion,axis=1)))
        if extent<1e-9 or variation<1e-9:continue
        expansion=float(np.sum(weights*np.sum(points*motion,axis=1))/extent)
        residual=float(np.sum(weights*np.sum((motion-expansion*points)**2,axis=1)))
        coherence=1-residual/variation
        risk=max(risk,float(np.clip(expansion/2,0,1)*np.clip((coherence-.5)/.5,0,1)))
    return risk


def taste_sample(body,world,config):
    from .perception import mouth_position
    origin=mouth_position(body);rotation=body.data.xmat[body.body_id].reshape(3,3)
    papillae=np.array([[0,.025,0],[0,-.025,0]])
    dose=np.zeros((2,3))
    if world is not None and world.food_enabled and world.food_energy>0:
        local=(world.food_position_m-origin)@rotation-papillae
        contact=np.clip((1-np.linalg.norm(local/(np.array([.075,.095,.060])+world.food_radius_m),axis=1))/.3,0,1)
        dose+=contact[:,None]*np.asarray(config.ball_taste)
    ecology=getattr(body,'ecology_state',None)
    for k in range(getattr(body,'prey_count',0)):
        if ecology is not None and ecology.prey_energy[k]<=0:continue
        ids=[body.model.geom(f'prey{k}_collision{j}').id for j in range(4)]
        local=(body.data.geom_xpos[ids]-origin)@rotation
        radius=float(body.model.geom_size[ids[0],0])
        contact=np.clip((1-np.linalg.norm((local[None,:,:]-papillae[:,None,:])/(np.array([.075,.095,.060])+radius),axis=2).min(axis=1))/.3,0,1)
        dose+=contact[:,None]*np.asarray(config.prey_taste)
    return dose/(1+dose)


def tactile_sample(body):
    """Six legs, left/right head, trunk: normal N, tangential N, contact onset N."""
    pressure=np.zeros((9,2));force=np.zeros(6)
    model,data=body.model,body.data;rotation=data.xmat[body.body_id].reshape(3,3)
    roots=model.body_rootid[model.geom_bodyid]
    legs=('lf','lm','lh','rf','rm','rh')
    for i,c in enumerate(data.contact):
        a,b=int(c.geom1),int(c.geom2);own_a=roots[a]==body.body_id;own_b=roots[b]==body.body_id
        if own_a==own_b:continue
        geom=a if own_a else b;name=model.geom(geom).name or ''
        zone=next((k for k,leg in enumerate(legs) if name.startswith(leg+'_')),8)
        if any(part in name for part in ('head','jaw','mandib','antenna')):
            zone=6+int(float((c.pos-data.xpos[body.body_id])@rotation[:,1])<0)
        mujoco.mj_contactForce(model,data,i,force)
        pressure[zone,0]+=max(0.,force[0]);pressure[zone,1]+=float(np.linalg.norm(force[1:3]))
    return pressure
