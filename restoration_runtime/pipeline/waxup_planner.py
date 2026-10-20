"""Compatibility entry point for the canonical patient-contour designer."""
import cv2
import numpy as np
from pipeline.crown_designer import compile_design


class CosmeticWaxupPlanner:
    @classmethod
    def plan_bilateral_smile_restoration(cls,image,crop_box,instances,target_size=(512,512),force_bilateral_mirror=None,design_options=None):
        if force_bilateral_mirror is True:
            raise ValueError('Automatic mirroring is retired. Select a reviewed replacement slot or crown repair instead.')
        design=compile_design(image,crop_box,instances,design_options)
        return design['scaffold'],design['mask'],design['waxup']

    @classmethod
    def detect_arch_diastemas_and_missing_teeth(cls,instances,crop_box):
        groups={'upper':[],'lower':[]}
        for inst in instances:
            poly=np.asarray(inst.get('polygon',[]),float)
            if len(poly)<3: continue
            pts=poly-np.asarray(crop_box[:2]);lo=pts.min(0);hi=pts.max(0)
            arch=inst.get('arch','lower' if (inst.get('fdi_number') or 0)>=30 else 'upper')
            groups[arch].append(dict(tooth_id=inst.get('id'),fdi=inst.get('fdi_number'),poly=pts,
                bbox=(lo[0],lo[1],hi[0]-lo[0],hi[1]-lo[1]),cx=float(pts[:,0].mean()),
                min_x=float(lo[0]),max_x=float(hi[0]),min_y=float(lo[1]),max_y=float(hi[1])))
        gaps={a:[] for a in groups}
        for arch,teeth in groups.items():
            teeth.sort(key=lambda t:t['cx'])
            for left,right in zip(teeth,teeth[1:]):
                width=right['min_x']-left['max_x']
                scale=max(1e-6,(left['bbox'][3]+right['bbox'][3])/2)
                if width/scale>.08:
                    gaps[arch].append(dict(left_id=left['tooth_id'],right_id=right['tooth_id'],gap_width=width,
                        gap_relative=width/scale,is_missing_tooth=False,presence='uncertain_gap',requires_review=True))
        return bool(gaps['upper'] or gaps['lower']),gaps['upper'],gaps['lower'],groups['upper'],groups['lower']
