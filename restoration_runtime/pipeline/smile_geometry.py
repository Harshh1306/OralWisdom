"""Legacy scaffold entry point, using the canonical designer without surgery heuristics."""
import numpy as np
from pipeline.crown_designer import compile_design


class SmileDesignEngine:
    @classmethod
    def sculpt_smile_architecture(cls,scaffold_roi,instances,crop_box,target_size=(512,512),design_options=None):
        x1,y1,x2,y2=map(int,crop_box)
        canvas=np.zeros((y2,x2,3),np.uint8)
        canvas[y1:y2,x1:x2]=scaffold_roi
        design=compile_design(canvas,crop_box,instances,design_options)
        return design['scaffold'],design['mask'],design['waxup']
