"""Patient-contour crown design and a single mask/task compiler."""
from uuid import uuid4
import hashlib
import cv2
import numpy as np
from config.restoration_plan import CrownAppearance,RestorativeAppearancePlan,validate_design_options
from pipeline.tooth_numbering import arch_frame
from pipeline.ceramic_appearance import shade_foreground


def polygon_mask(polygon,shape):
    mask=np.zeros(shape[:2],np.uint8)
    pts=np.asarray(polygon,dtype=np.float32)
    if len(pts)>=3: cv2.fillPoly(mask,[np.rint(pts).astype(np.int32)],255)
    return mask


def contour_polygon(mask):
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return []
    contour=max(contours,key=cv2.contourArea)
    contour=cv2.approxPolyDP(contour,max(.35,cv2.arcLength(contour,True)*.002),True)
    return contour.reshape(-1,2).astype(float).tolist()


def _design_crown(roi,source_mask,points,frame,arch,tooth_type,operation,style):
    """Conservative incisal smoothing/extension, fixed cervical boundary."""
    if operation not in ('veneer_preview','repair_crown') or style=='original':
        return roi.copy(),source_mask.copy()
    axis=frame[0];vertical=frame[1]*(1 if arch=='upper' else -1)
    coords=np.asarray(points,float)@np.array([axis,vertical]).T
    top,bottom=np.percentile(coords[:,1],[2,98]);height=max(1.,bottom-top)
    # Small relative extension, never a global arch cutoff or gum removal.
    extension=height*(.045 if tooth_type=='central' else (.025 if tooth_type=='lateral' else .015))
    if operation=='repair_crown': extension=min(height*.12,extension*2)
    yy,xx=np.mgrid[:roi.shape[0],:roi.shape[1]]
    u=xx*axis[0]+yy*axis[1];v=xx*vertical[0]+yy*vertical[1]
    ratio=np.clip((v-top)/height,0,1)
    shift=extension*ratio**2
    source_v=v-shift
    mapx=(u*axis[0]+source_v*vertical[0]).astype(np.float32)
    mapy=(u*axis[1]+source_v*vertical[1]).astype(np.float32)
    target=cv2.remap(source_mask,mapx,mapy,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
    radius=max(1,int(height*(.025 if style=='balanced' else .035)))
    radius=min(radius,4)
    closed=cv2.morphologyEx(target,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
    # Preserve all cervical/gingival source boundaries and upper body shape.
    target[v<top+height*.6]=source_mask[v<top+height*.6]
    target[v>=top+height*.6]=closed[v>=top+height*.6]
    warped=cv2.remap(roi,mapx,mapy,cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REFLECT_101)
    valid=cv2.remap(source_mask,mapx,mapy,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)>0
    invalid=(target>0)&~valid
    if np.any(invalid):
        # Fill only small contour-repair pixels from surrounding warped enamel.
        warped=cv2.inpaint(warped,invalid.astype(np.uint8)*255,2,cv2.INPAINT_TELEA)
    return warped,target


def compile_design(image,crop_box,instances,options=None,plan_id=None,image_hash=''):
    options=validate_design_options(options)
    from copy import deepcopy
    instances=deepcopy(instances)
    for tooth in instances:
        tooth['tooth_type']=options['tooth_type_overrides'].get(str(tooth['id']),tooth.get('tooth_type','unknown'))
    ids={str(t['id']) for t in instances}
    referenced=set(options['tooth_type_overrides'])|set(options['tooth_operations'])|set(options['arch_overrides'])|set(options['selected_ids'] or [])
    if referenced-ids: raise ValueError('Unknown selected tooth IDs: '+', '.join(sorted(referenced-ids)))
    x1,y1,x2,y2=map(int,crop_box);h,w=image.shape[:2]
    for region in options['replacements']+options['excluded_regions']:
        pts=np.asarray(region['polygon'],float)
        if np.any(pts<0) or np.any(pts[:,0]>=w) or np.any(pts[:,1]>=h): raise ValueError('A region lies outside the source image')
        if np.any(pts[:,0]<x1) or np.any(pts[:,0]>=x2) or np.any(pts[:,1]<y1) or np.any(pts[:,1]>=y2):
            raise ValueError('A replacement/correction must be inside the analyzed mouth crop')
    size=(512,512)
    scale=np.array([512/(x2-x1),512/(y2-y1)])
    roi=cv2.resize(image[y1:y2,x1:x2],size,interpolation=cv2.INTER_LANCZOS4)
    _,frame=arch_frame(instances)
    source_masks={};model_points={};arches={}
    for t in instances:
        tid=str(t['id']);points=(np.asarray(t['polygon'],float)-[x1,y1])*scale
        model_points[tid]=points
        source_masks[tid]=polygon_mask(points,roi.shape)
        arches[tid]=options['arch_overrides'].get(tid,t.get('arch','upper'))
    # Confidence-free edit selection: it is an appearance request, not diagnosis.
    operations={}
    for t in instances:
        tid=str(t['id']);op=options[arches[tid]+'_mode']
        if options['selected_ids'] is not None and tid not in options['selected_ids']: op='preserve'
        operations[tid]=options['tooth_operations'].get(tid,op)
    source_arch={a:np.zeros(roi.shape[:2],np.uint8) for a in ('upper','lower')}
    for tid,mask in source_masks.items(): source_arch[arches[tid]]|=mask
    excluded=np.zeros(roi.shape[:2],np.uint8)
    for region in options['excluded_regions']:
        excluded|=polygon_mask((np.asarray(region['polygon'])-[x1,y1])*scale,roi.shape)
    arch_masks={a:np.zeros(roi.shape[:2],np.uint8) for a in ('upper','lower')}
    shade_mask=np.zeros(roi.shape[:2],np.uint8)
    scaffold=roi.copy();crowns=[];regions=[];warnings=[];protected=np.zeros(roi.shape[:2],np.uint8)
    for tid,mask in source_masks.items():
        if operations[tid]=='preserve': protected|=mask
    observed_mask=source_arch['upper']|source_arch['lower']
    original_arch_masks={a:m.copy() for a,m in source_arch.items()}
    working_roi=roi.copy()
    proposals=[]
    if options['complete_smile']:
        from pipeline.smile_completion import propose_teeth
        proposals=propose_teeth(roi,source_masks,arches,operations,frame,excluded)
        for proposed in proposals:
            tid=proposed['id']
            while tid in source_masks: tid += '_gap'
            proposed['id']=tid
            arch=proposed['arch'];mask=proposed['mask'].copy()
            mask[(source_arch['upper']>0)|(source_arch['lower']>0)]=0
            points=np.asarray(contour_polygon(mask))
            if len(points)<3: continue
            source_masks[tid]=mask;model_points[tid]=points;arches[tid]=arch
            operations[tid]='veneer_preview';source_arch[arch]|=mask
            working_roi[mask>0]=proposed['texture'][mask>0]
            native=(points/scale+[x1,y1]).tolist()
            instances.append(dict(id=tid,arch=arch,tooth_type=proposed['tooth_type'],
                polygon=native,centroid=np.mean(native,axis=0).tolist(),
                proposed=True,donor_id=proposed['donor_id']))
        warnings.append(f'{len(proposals)} proposed smile teeth: review their outlines and count before rendering.')
    arch_shades={}
    for arch in {p['arch'] for p in proposals}:
        reference=shade_foreground(roi,original_arch_masks[arch],options['shade'])
        core=cv2.erode(original_arch_masks[arch],np.ones((5,5),np.uint8))>0
        if np.count_nonzero(core)>20:
            arch_shades[arch]=np.median(cv2.cvtColor(reference,cv2.COLOR_BGR2LAB)[core],axis=0)
    aligned={};alignment_diagnostics=[]
    if options['alignment']=='aligned' and options['shape']!='original' and options['alignment_strength']>0:
        from pipeline.arch_alignment import aligned_crowns,cleared_background
        aligned,alignment_diagnostics=aligned_crowns(working_roi,source_masks,arches,operations,frame,options['alignment_strength'],
                                                    {str(t['id']):t.get('tooth_type') for t in instances},options['shape'],
                                                    options['veneer_template'],options['arch_design'])
        background=cleared_background(roi,observed_mask)
        all_targets=np.zeros(roi.shape[:2],np.uint8)
        for _,target in aligned.values():all_targets|=target
        for tid in aligned:
            # Clear only enamel that the new design actually vacates. Clearing
            # the entire crown margin creates false dark collars at the gums.
            vacated=((source_masks[tid]>0)&(all_targets==0)).astype(np.uint8)*255
            expanded=cv2.dilate(vacated,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
            erase=(expanded>0)&(all_targets==0)&(protected==0)&(excluded==0)
            scaffold[erase]=background[erase]
    for t in instances:
        tid=str(t['id']);arch=arches[tid];op=operations[tid];src=source_masks[tid]
        warped,target=aligned[tid] if tid in aligned else _design_crown(working_roi,src,model_points[tid],frame,arch,t.get('tooth_type','tooth'),op,options['shape'])
        other='lower' if arch=='upper' else 'upper'
        if op=='preserve': target=src.copy()
        else: target[(source_arch[other]>0)|(protected>0)|(excluded>0)]=0
        if op not in ('preserve','shade_only'):
            region=src|target
            if tid in aligned:
                # Include a narrow transition band so resampling cannot leave the
                # original enamel silhouette just outside the approved edit.
                region=cv2.dilate(region,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
            region[(source_arch[other]>0)|(protected>0)|(excluded>0)]=0
            arch_masks[arch]|=region
            shaded=shade_foreground(warped,target,options['shade'])
            if arch in arch_shades:
                from pipeline.ceramic_appearance import harmonize_completed_arch
                shaded=harmonize_completed_arch(shaded,target,arch_shades[arch])
            scaffold[target>0]=shaded[target>0]
            regions.append(dict(tooth_id=tid,arch=arch,operation=op,source_mask=np.zeros_like(src) if t.get('proposed') else src,target_mask=target,edit_mask=region,aligned=tid in aligned))
        elif op=='shade_only':
            shade_mask|=target
        target_polygon=(np.asarray(contour_polygon(target))/scale+[x1,y1]).tolist() if np.any(target) else []
        crowns.append(CrownAppearance(tid,arch,t.get('tooth_type','tooth'),t.get('fdi_number'),t.get('fdi_estimate'),
            op,[] if t.get('proposed') else t['polygon'],target_polygon,
            presence='proposed_smile_tooth' if t.get('proposed') else 'observed',
            detection_confidence=None if t.get('proposed') else float(t.get('confidence',0) or 0),
            assignment_uncertain=bool(t.get('fdi_assignment_uncertain',True)),
            design_source='neighbor_enamel_proposal:'+t['donor_id'] if t.get('proposed') else ('coordinated_arch_alignment' if tid in aligned else 'patient_contour')))
        if t.get('arch_assignment_uncertain'): warnings.append(f'{tid}: estimated arch/tooth position; review selection')
    for index,slot in enumerate(options['replacements']):
        arch=slot['arch'];tid=f'replacement_{index+1}'
        target=polygon_mask((np.asarray(slot['polygon'])-[x1,y1])*scale,roi.shape)
        target[(source_arch['upper']>0)|(source_arch['lower']>0)|(excluded>0)]=0
        if np.count_nonzero(target)<20: raise ValueError('Replacement slot is too small or overlaps an existing tooth')
        arch_masks[arch]|=target
        # Neighbor foreground donor, never an unmasked rectangular gum crop.
        donors=[t for t in instances if arches[str(t['id'])]==arch]
        if not donors: raise ValueError('A replacement needs a visible same-arch donor tooth')
        center=np.asarray(slot['polygon']).mean(0)
        donor=min(donors,key=lambda t:np.linalg.norm(np.asarray(t['centroid'])-center))
        donor_mask=source_masks[str(donor['id'])]
        dx,dy,dw,dh=cv2.boundingRect(donor_mask);tx,ty,tw,th=cv2.boundingRect(target)
        crop=roi[dy:dy+dh,dx:dx+dw].copy();foreground=donor_mask[dy:dy+dh,dx:dx+dw]
        eroded=cv2.erode(foreground,np.ones((3,3),np.uint8))>0
        color=np.median(crop[eroded if np.any(eroded) else foreground>0],axis=0)
        crop[foreground==0]=color
        patch=cv2.resize(crop,(tw,th),interpolation=cv2.INTER_LANCZOS4)
        prepared=roi.copy();prepared[ty:ty+th,tx:tx+tw]=patch
        prepared=shade_foreground(prepared,target,options['shade'])
        scaffold[target>0]=prepared[target>0]
        zero=np.zeros_like(target)
        regions.append(dict(tooth_id=tid,arch=arch,operation='replacement_crown_preview',source_mask=zero,target_mask=target,edit_mask=target))
        polygon=(np.asarray(contour_polygon(target))/scale+[x1,y1]).tolist()
        crowns.append(CrownAppearance(tid,arch,slot.get('tooth_type','central'),None,None,'replacement_crown_preview',[],polygon,
            presence='reviewed_replacement_slot',detection_confidence=None,design_source='masked_neighbor_donor:'+str(donor['id'])))
    arch_masks['lower'][arch_masks['upper']>0]=0
    shade_mask[(arch_masks['upper']>0)|(arch_masks['lower']>0)|(excluded>0)]=0
    mask=arch_masks['upper']|arch_masks['lower']
    plan=RestorativeAppearancePlan(plan_id or uuid4().hex,image_hash,list(crop_box),options,crowns)
    metadata=dict(upper_mask=arch_masks['upper'],lower_mask=arch_masks['lower'],shade_mask=shade_mask,
        upper_scaffold=scaffold.copy(),lower_scaffold=scaffold.copy(),has_upper_teeth=bool(np.any(arch_masks['upper'])),
        has_lower_teeth=bool(np.any(arch_masks['lower'])),bilateral_mirror_applied=False,multi_unit_bridge_applied=bool(options['replacements']),
        rationale='Cosmetic layout with proposed gap teeth; not clinical tooth identity or bite prediction' if proposals else 'Bounded visible-crown design; no inferred missing teeth or root/bite movement',
        proposed_teeth_count=sum(c.presence=='proposed_smile_tooth' for c in crowns),
        completed_arch_shades={a:v.tolist() for a,v in arch_shades.items()},
        upper_actions=[dict(tooth_id=c.tooth_id,action=c.operation) for c in crowns if c.arch=='upper' and c.operation!='preserve'],
        lower_actions=[dict(tooth_id=c.tooth_id,action=c.operation) for c in crowns if c.arch=='lower' and c.operation!='preserve'],
        appearance_plan=plan.to_dict(),regions=regions,design_warnings=list(dict.fromkeys(warnings)),
        texture_preservation=options['texture_preservation'],shade=options['shade'],alignment_diagnostics=alignment_diagnostics,
        geometry_mode=options['alignment'],geometry_guided=bool(aligned),
        veneer_template=options['veneer_template'],arch_design=options['arch_design'])
    return dict(crop_box=tuple(crop_box),scaffold=scaffold,mask=mask,waxup=metadata,plan=plan.to_dict())


def native_mask(model_mask,crop_box,image_shape):
    mask=np.zeros(image_shape[:2],np.uint8)
    x1,y1,x2,y2=map(int,crop_box)
    mask[y1:y2,x1:x2]=cv2.resize(model_mask,(x2-x1,y2-y1),interpolation=cv2.INTER_NEAREST)
    return mask


def protect_native_mask(mask,appearance_plan):
    """Reapply original polygon boundaries after model-to-photo resampling."""
    protected=np.zeros(mask.shape,np.uint8)
    for crown in appearance_plan.get('crowns',[]):
        points=crown.get('source_polygon',[])
        if crown.get('operation')=='preserve' and len(points)>=3:
            cv2.fillPoly(protected,[np.asarray(points,np.int32)],255)
    for region in appearance_plan.get('options',{}).get('excluded_regions',[]):
        cv2.fillPoly(protected,[np.asarray(region['polygon'],np.int32)],255)
    result=mask.copy();result[protected>0]=0
    return result
