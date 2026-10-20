"""Relative photographic appearance; no physical ceramic/shade calibration."""
import cv2
import numpy as np
PRESETS={'natural':(16.,.58),'brighter':(22.,.42),'bright':(30.,.28)}


def harmonize_completed_arch(image,mask,reference_lab):
    """Match an arch's relative photographic shade without flattening enamel detail."""
    core=cv2.erode(mask,np.ones((5,5),np.uint8))>0
    if np.count_nonzero(core)<20:return image
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype(np.float32)
    current=np.median(lab[core],axis=0)
    shift=np.clip(np.asarray(reference_lab)-current,[-35,-24,-24],[35,24,24])
    shift*=np.array([.65,.9,.9])
    lab[mask>0]+=shift
    converted=cv2.cvtColor(np.clip(lab,0,255).astype(np.uint8),cv2.COLOR_LAB2BGR)
    result=image.copy();result[mask>0]=converted[mask>0]
    return result


def reduce_lab_chroma(channel,factor):
    return np.clip(128.+(channel.astype(np.float32)-128.)*factor,0,255)


def shade_foreground(image,mask,shade='natural'):
    boost,chroma=PRESETS[shade]
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype(np.float32)
    light=lab[:,:,0]
    lab[:,:,0]=light+boost*np.clip((250.-light)/100.,0,1)
    lab[:,:,2]=reduce_lab_chroma(lab[:,:,2],chroma)
    lab[:,:,1]=reduce_lab_chroma(lab[:,:,1],.96)
    colored=cv2.cvtColor(np.clip(lab,0,255).astype(np.uint8),cv2.COLOR_LAB2BGR)
    interior=cv2.distanceTransform((mask>0).astype(np.uint8),cv2.DIST_L2,5)
    radius=max(1.,min(4.,np.sqrt(max(1,np.count_nonzero(mask)))/35.))
    alpha=np.minimum(interior/radius,1.)[:,:,None]
    out=np.clip(image.astype(float)*(1-alpha)+colored.astype(float)*alpha,0,255).astype(np.uint8)
    out[mask==0]=image[mask==0]
    return out


def preserve_enamel_detail(source,generated,mask,strength=.35):
    """Transfer restrained high-frequency luminance, never old contour edges."""
    if strength<=0 or not np.any(mask): return generated
    src=cv2.cvtColor(source,cv2.COLOR_BGR2LAB).astype(np.float32)
    dst=cv2.cvtColor(generated,cv2.COLOR_BGR2LAB).astype(np.float32)
    detail=src[:,:,0]-cv2.GaussianBlur(src[:,:,0],(0,0),1.2)
    interior=cv2.erode((mask>0).astype(np.uint8),np.ones((5,5),np.uint8))
    dst[:,:,0]+=np.clip(detail,-10,10)*strength*interior
    colored=cv2.cvtColor(np.clip(dst,0,255).astype(np.uint8),cv2.COLOR_LAB2BGR)
    result=generated.copy();result[interior>0]=colored[interior>0]
    return result


def match_crown_appearance(reference,generated,regions,detail_strength=.35):
    """Match local chroma/illumination to the designed source crown.

    Spatially warped reference contains the selected contour and source lighting;
    averaging raw original teeth here would reintroduce old edges and defects.
    """
    ref=cv2.cvtColor(reference,cv2.COLOR_BGR2LAB).astype(np.float32)
    out=cv2.cvtColor(generated,cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_light=cv2.GaussianBlur(ref[:,:,0],(0,0),3.)
    out_light=cv2.GaussianBlur(out[:,:,0],(0,0),3.)
    ref_detail=np.clip(ref[:,:,0]-cv2.GaussianBlur(ref[:,:,0],(0,0),1.2),-10,10)
    all_target=np.zeros(generated.shape[:2],np.uint8)
    for region in regions:
        mask=region['target_mask']
        core=cv2.erode(mask,np.ones((5,5),np.uint8))>0
        if np.count_nonzero(core)<20: continue
        distance=cv2.distanceTransform((mask>0).astype(np.uint8),cv2.DIST_L2,5)
        alpha=np.clip(distance/3.,0,1)
        for channel in (1,2):
            ref_median=float(np.median(ref[:,:,channel][core]))
            # Reduce residual yellow stain while retaining captured white balance.
            if channel==2: ref_median=min(ref_median,144.)
            difference=np.clip(ref_median-float(np.median(out[:,:,channel][core])),-24,24)
            out[:,:,channel]+=difference*.85*alpha
        ref_median=float(np.median(ref_light[core]))
        out_median=float(np.median(out_light[core]))
        lighting=(ref_light-ref_median)-(out_light-out_median)
        level=np.clip(ref_median-out_median,-40,40)*.65
        # Roll off positive changes near existing highlights rather than clipping.
        headroom=np.clip((255.-out[:,:,0])/max(20.,255.-out_median),0,1.5) if level>0 else np.ones_like(alpha)
        out[:,:,0]+=(np.clip(lighting,-45,45)*.6+level*headroom+ref_detail*detail_strength)*alpha
        all_target|=mask
    colored=cv2.cvtColor(np.clip(out,0,255).astype(np.uint8),cv2.COLOR_LAB2BGR)
    result=generated.copy();result[all_target>0]=colored[all_target>0]
    return result
