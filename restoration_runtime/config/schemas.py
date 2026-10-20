"""Validate user appearance intent without manufacturing clinical findings."""
from copy import deepcopy
import math
from config.restoration_plan import validate_design_options


def normalize_vlm_output(data,instances=None):
    if not isinstance(data,dict): return data
    result=deepcopy(data)
    keys=('full_mouth_rehabilitation_prompt','full_mouth_rehabilitation','prosthetic_specification','rehabilitation_plan')
    key=next((k for k in keys if k in data),None)
    if not key and ('teeth' in data or not any('veneer' in str(v).lower() for v in data.values())): return result
    rehab=data.get(key,data) if key else data
    if not isinstance(rehab,dict): raise ValueError('Rehabilitation intent must be an object')
    spec=rehab.get('prosthetic_specification',rehab)
    if not isinstance(spec,dict): raise ValueError('Prosthetic specification must be an object')
    shade=str(spec.get('shade','natural ivory'))
    style='bright' if any(v in shade.lower() for v in ('om1','om2','bl1','bleach')) else ('brighter' if any(v in shade.lower() for v in ('b1','a1','bright')) else 'natural')
    upper=spec.get('arch_1_maxillary_veneers_fdi')
    lower=spec.get('arch_2_mandibular_veneers_fdi')
    if any(v is not None and (not isinstance(v,list) or not all(isinstance(n,int) and not isinstance(n,bool) for n in v)) for v in (upper,lower)):
        raise ValueError('Target FDI ranges must be integer lists')
    request=dict(shade=style,upper_mode='veneer_preview',lower_mode='veneer_preview' if lower else 'preserve')
    if upper is not None or lower is not None:
        targets=set((upper or [])+(lower or []))
        request['selected_ids']=[str(t['id']) for t in (instances or []) if (t.get('fdi_number') or t.get('fdi_estimate')) in targets]
    teeth=[]
    for inst in instances or []:
        tid=str(inst['id']);selected=request.get('selected_ids')
        wants=(selected is None or tid in selected) and (inst.get('arch','upper')=='upper' or bool(lower))
        teeth.append(dict(tooth_id=tid,fdi_number=inst.get('fdi_number'),visibility='unknown',observations={},
            confidence={'overall':0.0},source='user_appearance_intent',
            edit_intent=dict(apply_porcelain_veneer=wants,target_shade=shade,preserve_gingival_margin=True)))
    result.update(smile_analysis=dict(source='cosmetic_preference_not_diagnosis',confidence=0.0,target_shade=shade),
        teeth=teeth,appearance_request=validate_design_options(request))
    return result


def validate_vlm_output(data,instances=None):
    if not isinstance(data,dict): return False,['Root data must be a dictionary']
    try: normalized=normalize_vlm_output(data,instances)
    except ValueError as exc: return False,[str(exc)]
    data.clear();data.update(normalized)
    errors=[]
    if not isinstance(data.get('smile_analysis'),dict): errors.append('smile_analysis must be an object')
    if not isinstance(data.get('teeth'),list): errors.append('teeth must be a list')
    if errors: return False,errors
    sa=data['smile_analysis']
    if not isinstance(sa.get('confidence'),(float,int)) or not math.isfinite(sa['confidence']) or not 0<=sa['confidence']<=1: sa['confidence']=0.0
    for index,tooth in enumerate(data['teeth']):
        if not isinstance(tooth,dict): errors.append(f'Tooth {index} must be an object');continue
        if not isinstance(tooth.get('tooth_id'),(str,int)): errors.append(f'Tooth {index} needs an instance ID')
        fdi=tooth.get('fdi_number')
        if fdi is not None and (not isinstance(fdi,int) or isinstance(fdi,bool) or fdi//10 not in (1,2,3,4) or not 1<=fdi%10<=8): errors.append(f'Invalid FDI for tooth {index}')
        if tooth.get('visibility') not in ('full','partial','hidden','unknown'): tooth['visibility']='unknown'
        for field in ('observations','edit_intent','confidence'):
            if field not in tooth: tooth[field]={}
            if not isinstance(tooth[field],dict): errors.append(f'{field} must be an object for tooth {index}')
        if isinstance(tooth.get('confidence'),dict):
            conf=tooth['confidence'].get('overall')
            if not isinstance(conf,(int,float)) or not math.isfinite(conf) or not 0<=conf<=1: tooth['confidence']['overall']=0.0
    if 'appearance_request' in data:
        try: data['appearance_request']=validate_design_options(data['appearance_request'])
        except ValueError as exc: errors.append(str(exc))
    return not errors,errors
