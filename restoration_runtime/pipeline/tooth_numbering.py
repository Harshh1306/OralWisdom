"""Arch grouping; spatial FDI estimates never authorize missing-tooth edits."""
from copy import deepcopy
import cv2
import numpy as np


def arch_frame(instances):
    centers=np.array([np.mean(np.asarray(t['polygon'],float),axis=0) for t in instances])
    origin=np.median(centers,axis=0)
    axis=np.array([1.,0.])
    if len(centers)>=3:
        values,vectors=np.linalg.eigh(np.cov((centers-origin).T))
        candidate=vectors[:,np.argmax(values)]
        if candidate[0]<0: candidate=-candidate
        if abs(candidate[1])<=.65: axis=candidate
    return origin,np.array([axis,[-axis[1],axis[0]]])


def assign_tooth_numbers(instances,image_shape):
    valid=[]
    for index,original in enumerate(instances):
        pts=np.asarray(original.get('polygon',[]),float)
        if pts.ndim!=2 or pts.shape[1:]!=(2,) or len(pts)<3 or not np.isfinite(pts).all(): continue
        inst=deepcopy(original)
        inst['id']=str(inst.get('id',f'T{index+1:02d}'))
        inst['centroid']=pts.mean(0).tolist()
        duplicate=False
        for prior in valid:
            a,b=cv2.convexHull(pts.astype(np.float32)),cv2.convexHull(np.asarray(prior['polygon'],np.float32))
            overlap,_=cv2.intersectConvexConvex(a,b)
            if overlap/max(1.,min(cv2.contourArea(a),cv2.contourArea(b)))>.8:
                duplicate=True;break
        if not duplicate: valid.append(inst)
    if not valid: return []
    # Reject distant clothing/background detections before fitting the arch frame.
    median_y=float(np.median([t['centroid'][1] for t in valid]))
    median_height=float(np.median([np.ptp(np.asarray(t['polygon']),axis=0)[1] for t in valid]))
    distance=max(float(image_shape[0])*.35,median_height*2.8)
    valid=[t for t in valid if abs(t['centroid'][1]-median_y)<distance]
    if not valid: return []
    origin,frame=arch_frame(valid)
    local=np.array([(np.asarray(t['centroid'])-origin)@frame.T for t in valid])
    heights=[np.ptp((np.asarray(t['polygon'])-origin)@frame.T,axis=0)[1] for t in valid]
    size=max(1e-6,float(np.median(heights)))
    ys=np.sort(local[:,1]);best=None
    for i in range(1,len(ys)):
        score=i*np.var(ys[:i])+(len(ys)-i)*np.var(ys[i:])
        if best is None or score<best[0]: best=(score,(ys[i-1]+ys[i])/2,ys[i:].mean()-ys[:i].mean())
    two=best is not None and np.ptp(ys)>=.76*size and best[2]>=.5*size
    for i,t in enumerate(valid):
        known=t.get('arch') if t.get('arch_confirmed') else None
        t['arch']=known if known in ('upper','lower') else ('lower' if two and local[i,1]>=best[1] else 'upper')
        t['arch_assignment_source']='reviewed' if known else ('spatial_two_arch' if two else 'single_arch_assumed_upper')
        t['arch_assignment_uncertain']=not bool(known)
    for arch in ('upper','lower'):
        group=sorted([t for t in valid if t['arch']==arch],key=lambda t:(np.asarray(t['centroid'])-origin)@frame[0])
        if not group: continue
        xs=np.array([(np.asarray(t['centroid'])-origin)@frame[0] for t in group]);mid=(xs[0]+xs[-1])/2
        for t,x in zip(group,xs):
            side=[u for u in group if (((np.asarray(u['centroid'])-origin)@frame[0])<mid)==(x<mid)]
            side.sort(key=lambda u:abs((np.asarray(u['centroid'])-origin)@frame[0]-mid))
            rank=next(i for i,u in enumerate(side) if u is t)
            start=(11 if x<mid else 21) if arch=='upper' else (41 if x<mid else 31)
            t['fdi_estimate']=start+rank if rank<8 else None
            t['fdi_assignment_uncertain']=not bool(t.get('fdi_confirmed'))
            if not t.get('fdi_confirmed'): t['fdi_number']=None;t['universal_number']=None
            cls=str(t.get('class_name','')).lower()
            t['tooth_type']='canine' if cls=='canine' else ('central' if rank==0 else ('lateral' if rank==1 else 'premolar'))
            t['tooth_type_source']='detector' if cls=='canine' else 'position_estimate'
    return valid
