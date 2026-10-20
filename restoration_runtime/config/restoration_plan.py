"""Versioned cosmetic intent, independent from diagnostic observations."""
from dataclasses import dataclass,asdict
from typing import Optional
from copy import deepcopy
import math
import numpy as np

OPERATIONS={'preserve','shade_only','veneer_preview','repair_crown','replacement_crown_preview'}
SHADES={'natural','brighter','bright'}
SHAPES={'balanced','soft','original'}


def validate_design_options(options=None):
    if options is not None and not isinstance(options,dict): raise ValueError('Design options must be an object')
    options=deepcopy(options or {})
    allowed={'upper_mode','lower_mode','shade','shape','alignment','alignment_strength','selected_ids','tooth_operations','arch_overrides','replacements','excluded_regions','texture_preservation','veneer_template','arch_design','tooth_type_overrides','complete_smile'}
    if set(options)-allowed: raise ValueError('Unknown design options: '+', '.join(sorted(set(options)-allowed)))
    result=dict(upper_mode='veneer_preview',lower_mode='preserve',shade='natural',shape='balanced',selected_ids=None,
                tooth_operations={},arch_overrides={},replacements=[],excluded_regions=[],texture_preservation=.35,
                alignment='preserve',alignment_strength=.85,veneer_template='legacy',arch_design={},tooth_type_overrides={},complete_smile=False)
    result.update(options)
    if not isinstance(result['complete_smile'],bool): raise ValueError('complete_smile must be a boolean')
    if not isinstance(result['alignment'],str) or result['alignment'] not in ('aligned','preserve'):
        raise ValueError('Unknown geometry mode')
    value=result['alignment_strength']
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
        raise ValueError('Alignment strength must be between zero and one')
    for key in ('upper_mode','lower_mode'):
        if not isinstance(result[key],str) or result[key] not in OPERATIONS-{'replacement_crown_preview'}: raise ValueError(f'Invalid {key}')
    if not isinstance(result['shade'],str) or not isinstance(result['shape'],str) or result['shade'] not in SHADES or result['shape'] not in SHAPES: raise ValueError('Unknown shade or shape style')
    if result['veneer_template'] not in ('legacy','natural_v1'):
        raise ValueError('Unknown veneer template')
    arch_design=result['arch_design']
    if not isinstance(arch_design,dict) or set(arch_design)-{'upper','lower'}:
        raise ValueError('Invalid arch design')
    limits={'edge_offset':(-.12,.12),'curve':(-.12,.12),'width':(.85,1.15),
            'conditioning_scale':(0,2),'guidance_scale':(0,15),'control_guidance_end':(0,1)}
    for arch,settings in arch_design.items():
        if not isinstance(settings,dict) or set(settings)-set(limits): raise ValueError('Invalid arch settings')
        for key,value in settings.items():
            lo,hi=limits[key]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not lo<=value<=hi:
                raise ValueError(f'Invalid {arch} {key}')
    types=result['tooth_type_overrides']
    if not isinstance(types,dict) or len(types)>64 or any(not isinstance(k,str) or not isinstance(v,str) or v not in ('central','lateral','canine','premolar','unknown') for k,v in types.items()):
        raise ValueError('Invalid tooth type overrides')
    ids=result['selected_ids']
    if ids is not None and (not isinstance(ids,list) or len(ids)>64 or not all(isinstance(v,str) for v in ids) or len(set(ids))!=len(ids)):
        raise ValueError('selected_ids must contain unique tooth IDs')
    for key,choices in [('tooth_operations',OPERATIONS-{'replacement_crown_preview'}),('arch_overrides',{'upper','lower'})]:
        if not isinstance(result[key],dict) or len(result[key])>64 or any(not isinstance(k,str) or not isinstance(v,str) or v not in choices for k,v in result[key].items()): raise ValueError(f'Invalid {key}')
    strength=result['texture_preservation']
    if isinstance(strength,bool) or not isinstance(strength,(float,int)) or not math.isfinite(strength) or not 0<=strength<=.6:
        raise ValueError('Texture preservation must be between 0 and 0.6')
    for key in ('replacements','excluded_regions'):
        if not isinstance(result[key],list) or len(result[key])>16: raise ValueError(f'{key} must be a list with at most 16 regions')
        for region in result[key]:
            if not isinstance(region,dict): raise ValueError('Each region must be an object')
            try: pts=np.asarray(region.get('polygon',[]),dtype=float)
            except (ValueError,TypeError) as exc: raise ValueError('Invalid region polygon') from exc
            if pts.ndim!=2 or pts.shape[1:]!=(2,) or not 3<=len(pts)<=64 or not np.isfinite(pts).all(): raise ValueError('Region polygon must have 3 to 64 finite points')
            if key=='replacements' and (region.get('reviewed') is not True or region.get('arch') not in ('upper','lower')):
                raise ValueError('A replacement crown requires a reviewed slot and arch')
            if key=='replacements' and region.get('tooth_type','central') not in ('central','lateral','canine','premolar'):
                raise ValueError('Invalid replacement tooth type')
    return result


@dataclass
class CrownAppearance:
    tooth_id:str
    arch:str
    tooth_type:str
    fdi_number:Optional[int]
    fdi_estimate:Optional[int]
    operation:str
    source_polygon:list
    target_polygon:list
    presence:str='observed'
    detection_confidence:Optional[float]=None
    assignment_uncertain:bool=True
    design_source:str='patient_contour'


@dataclass
class RestorativeAppearancePlan:
    plan_id:str
    image_sha256:str
    crop_box:list
    options:dict
    crowns:list
    schema_version:int=1
    revision:int=1
    anatomy_validated:bool=False

    def to_dict(self): return asdict(self)
