"""Bounded 2D crown alignment. No root/bite movement or clinical prediction."""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


def _bounds(mask,axis,vertical):
    y,x=np.nonzero(mask)
    p=np.stack([x,y],axis=1)
    uv=p@np.array([axis,vertical]).T
    return tuple(np.percentile(uv[:,i],q) for i,q in ((0,1),(0,99),(1,1),(1,99)))


def aligned_crowns(roi,masks,arches,operations,frame,strength=.85,types=None,style='balanced',template='legacy',arch_design=None):
    """Fit a shared incisal curve and non-overlapping slots for contiguous crowns."""
    yy,xx=np.mgrid[:roi.shape[0],:roi.shape[1]]
    axis=frame[0];all_results={};diagnostics=[]
    for arch in ('upper','lower'):
        controls=(arch_design or {}).get(arch,{})
        vertical=frame[1]*(1 if arch=='upper' else -1)
        u=xx*axis[0]+yy*axis[1];v=xx*vertical[0]+yy*vertical[1]
        ids=[key for key in masks if arches[key]==arch and np.any(masks[key])]
        bounds={key:_bounds(masks[key],axis,vertical) for key in ids}
        ids.sort(key=lambda key:(bounds[key][0]+bounds[key][1])/2)
        typical=float(np.median([bounds[k][1]-bounds[k][0] for k in ids])) if ids else 1.
        groups=[];group=[]
        for key in ids:
            if operations[key] not in ('veneer_preview','repair_crown') or (types and types.get(key)=='premolar'):
                if group:groups.append(group);group=[]
                continue
            if group and bounds[key][0]-bounds[group[-1]][1]>.42*typical:
                groups.append(group);group=[]
            group.append(key)
        if group:groups.append(group)
        for group in groups:
            if len(group)<2:continue
            data=np.array([bounds[k] for k in group]);centers=data[:,:2].mean(1)
            widths=data[:,1]-data[:,0];heights=data[:,3]-data[:,2]
            height=float(np.median(heights));mid=float((centers[0]+centers[-1])/2)
            # Smooth arch shape with no linear slope in the arch's local frame.
            z=(centers-mid)/max(1.,(centers[-1]-centers[0])/2)
            if len(group)>=4:
                fit=np.linalg.lstsq(np.stack([np.ones_like(z),z*z],axis=1),data[:,3],rcond=None)[0]
                curvature=float(np.clip(fit[1],-.16*height,.08*height))
                level=float(np.median(data[:,3]-curvature*z*z))
            else:curvature=0.;level=float(np.mean(data[:,3]))
            desired=level+curvature*z*z
            desired += height*(controls.get('edge_offset',0)+controls.get('curve',0)*z*z)
            if template=='natural_v1':
                from pipeline.veneer_templates import incisal_offset
                desired += heights*np.array([incisal_offset(arch,(types or {}).get(k,'unknown')) for k in group])
            tips=data[:,3]+strength*np.clip(desired-data[:,3],-.3*heights,.3*heights)
            # Normalize asymmetric widths gently, retaining perspective and existing count.
            smooth=np.array([np.median(widths[max(0,i-1):min(len(widths),i+2)]) for i in range(len(widths))])
            weights=widths*.7+smooth*.3
            left,right=float(data[0,0]),float(data[-1,1])
            allocated=weights/weights.sum()*(right-left)
            boundaries=np.r_[left,left+np.cumsum(allocated)]
            target_centers=(boundaries[:-1]+boundaries[1:])/2
            shift=np.clip(target_centers-centers,-.3*widths,.3*widths)*strength
            target_centers=centers+shift
            # Common contact divisions keep neighboring crown bodies distinct.
            divisions=np.r_[left,(target_centers[:-1]+target_centers[1:])/2,right]
            group_changes=[]
            for i,key in enumerate(group):
                src=masks[key];u0,u1,top,bottom=data[i];old_h=max(1.,bottom-top)
                tip=float(tips[i]);new_h=max(1.,tip-top);center=float(target_centers[i])
                width=min(float(allocated[i]),2*min(center-divisions[i],divisions[i+1]-center))
                width=max(width,typical*.25)*controls.get('width',1.)
                ratio=(v-top)/new_h
                # Rounded cervical shoulders, straight bodies, small incisal embrasures.
                tip_width=.65 if style=='soft' else .78
                if types and types.get(key)=='canine':tip_width=min(tip_width,.6)
                profile=np.interp(ratio,[0,.08,.26,.78,.92,1.],[.48,.78,1.,.99,.94,tip_width])
                if template=='natural_v1':
                    from pipeline.veneer_templates import silhouette
                    profile=silhouette(ratio,arch,(types or {}).get(key,'unknown'),style)
                # Warp each patient's enamel using source row extents, removing original tilt.
                source_v=top+np.clip(ratio,0,1)*old_h
                rows=np.arange(int(np.floor(top)),int(np.ceil(bottom))+1)
                lows=[];highs=[]
                for row in rows:
                    values=u[(src>0)&(np.abs(v-row)<1.)]
                    lows.append(float(np.min(values)) if len(values) else u0)
                    highs.append(float(np.max(values)) if len(values) else u1)
                low=np.interp(source_v,rows,lows);high=np.interp(source_v,rows,highs)
                blend=np.clip((ratio-.16)/.42,0,1);blend=blend*blend*(3-2*blend)*strength
                left_edge=low*(1-blend)+(center-width*.5*profile)*blend
                right_edge=high*(1-blend)+(center+width*.5*profile)*blend
                # Move contact boundaries gradually too, avoiding a hard step
                # where the original cervical contour meets the straighter body.
                left_edge=np.maximum(left_edge,low*(1-blend)+(divisions[i]+.35)*blend)
                right_edge=np.minimum(right_edge,high*(1-blend)+(divisions[i+1]-.35)*blend)
                target=((ratio>=0)&(ratio<=1)&(u>=left_edge)&(u<=right_edge)).astype(np.uint8)*255
                collar=v<top+old_h*.16
                target[collar]=src[collar]
                lateral=np.clip((u-left_edge)/np.maximum(1.,right_edge-left_edge),.025,.975)
                source_u=low+(high-low)*lateral
                mapx=(source_u*axis[0]+source_v*vertical[0]).astype(np.float32)
                mapy=(source_u*axis[1]+source_v*vertical[1]).astype(np.float32)
                warped=cv2.remap(roi,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
                warped[collar]=roi[collar]
                # Nearest valid source foreground removes any gum/background sampled at thin tips.
                valid=cv2.remap(src,mapx,mapy,cv2.INTER_NEAREST)>0
                bad=(target>0)&~valid
                if np.any(bad) and np.any(valid&(target>0)):
                    _,nearest=distance_transform_edt(~(valid&(target>0)),return_indices=True)
                    warped[bad]=warped[nearest[0][bad],nearest[1][bad]]
                all_results[key]=(warped,target)
                group_changes.append(dict(tooth_id=key,incisal_shift_px=round(tip-bottom,3),
                                          lateral_shift_px=round(float(shift[i]),3)))
            diagnostics.append(dict(arch=arch,teeth=group,source_tip_spread_px=round(float(np.std(data[:,3]-curvature*z*z)),3),
                target_tip_spread_px=round(float(np.std(tips-curvature*z*z)),3),changes=group_changes))
    return all_results,diagnostics


def cleared_background(roi,all_teeth):
    """Local photographic background for pixels vacated by redesigned crowns."""
    blocked=cv2.dilate(all_teeth,np.ones((7,7),np.uint8))>0
    if np.all(blocked):return roi.copy()
    _,nearest=distance_transform_edt(blocked,return_indices=True)
    filled=roi.copy();filled[blocked]=roi[nearest[0][blocked],nearest[1][blocked]]
    softened=cv2.GaussianBlur(filled,(0,0),1.)
    filled[blocked]=softened[blocked]
    return filled


