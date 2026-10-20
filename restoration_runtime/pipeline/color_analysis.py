import cv2
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def extract_tooth_colors(
    image_bgr: np.ndarray,
    instances: list[dict],
) -> list[dict]:
    """
    For each tooth instance, compute mean CIELAB color within the mask polygon.
    
    Args:
        image_bgr: Original image in BGR format
        instances: List of tooth instances with 'polygon' field
        
    Returns:
        List of dicts, one per tooth:
        {
            "tooth_id": str,           # e.g. "T01"
            "fdi_number": int,         # if assigned
            "L_mean": float,           # Lightness (0-100)
            "a_mean": float,           # green-red (-128 to 127)
            "b_mean": float,           # blue-yellow (-128 to 127)
            "L_std": float,
            "a_std": float,
            "b_std": float,
        }
    """
    results = []
    
    # Convert entire image to LAB first (OpenCV format)
    # L: 0-255 (scaled by 255/100)
    # a, b: 0-255 (shifted by 128)
    img_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    
    for i, inst in enumerate(instances):
        poly = inst.get('polygon')
        if poly is None or len(poly) == 0:
            continue
            
        pts = np.array(poly, np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        
        # Extract pixels
        lab_pixels = img_lab[mask == 255]
        if len(lab_pixels) == 0:
            continue
            
        # Convert OpenCV LAB to standard CIELAB
        # L = L_cv * 100 / 255
        # a = a_cv - 128
        # b = b_cv - 128
        
        std_lab = np.zeros(lab_pixels.shape, dtype=np.float32)
        std_lab[:, 0] = lab_pixels[:, 0] * 100.0 / 255.0
        std_lab[:, 1] = lab_pixels[:, 1] - 128.0
        std_lab[:, 2] = lab_pixels[:, 2] - 128.0
        
        mean_val = np.mean(std_lab, axis=0)
        std_val = np.std(std_lab, axis=0)
        
        res = {
            "tooth_id": inst.get("id", f"T{i+1:02d}"),
            "fdi_number": inst.get("fdi_number"),
            "L_mean": float(mean_val[0]),
            "a_mean": float(mean_val[1]),
            "b_mean": float(mean_val[2]),
            "L_std": float(std_val[0]),
            "a_std": float(std_val[1]),
            "b_std": float(std_val[2]),
        }
        results.append(res)
        
    return results

def compute_color_deltas(
    color_data: list[dict],
    instances: list[dict],
) -> list[dict]:
    """
    For each tooth, compute color difference to immediate neighbors
    (mesial and distal adjacent teeth in the arch).
    
    Returns each tooth dict with added fields:
    - delta_L_mesial: float (L* difference to mesial neighbor)
    - delta_L_distal: float (L* difference to distal neighbor) 
    - delta_b_mesial: float (b* difference - yellowing)
    - delta_b_distal: float
    - delta_E_mesial: float (total color difference ΔE*ab)
    - delta_E_distal: float
    - is_outlier_shade: bool (True if ΔE > 3.0 to both neighbors)
    """
    # Create mapping from tooth_id to index
    id_to_idx = {d['tooth_id']: i for i, d in enumerate(color_data)}
    
    # Group by arch and sort by arch_position
    upper_teeth = [inst for inst in instances if inst.get('arch') == 'upper']
    lower_teeth = [inst for inst in instances if inst.get('arch') == 'lower']
    
    upper_teeth.sort(key=lambda x: x.get('centroid', [0, 0])[0])
    lower_teeth.sort(key=lambda x: x.get('centroid', [0, 0])[0])
    
    def process_arch(arch_instances):
        n = len(arch_instances)
        mid_idx = (n - 1) / 2.0
        
        for i, inst in enumerate(arch_instances):
            t_id = inst.get('id')
            if t_id not in id_to_idx:
                continue
                
            idx = id_to_idx[t_id]
            cd = color_data[idx]
            
            # Determine mesial and distal neighbors based on position relative to midline
            # Left side of image (patient's right, i < mid_idx):
            # Mesial (towards midline) is i + 1
            # Distal (away from midline) is i - 1
            # Right side of image (patient's left, i > mid_idx):
            # Mesial is i - 1
            # Distal is i + 1
            
            mesial_neighbor = None
            distal_neighbor = None
            
            if i < mid_idx:
                if i + 1 < n: mesial_neighbor = arch_instances[i + 1]
                if i - 1 >= 0: distal_neighbor = arch_instances[i - 1]
            else:
                if i - 1 >= 0: mesial_neighbor = arch_instances[i - 1]
                if i + 1 < n: distal_neighbor = arch_instances[i + 1]
                
            def compute_diff(neighbor):
                if neighbor is None: return None, None, None
                n_id = neighbor.get('id')
                if n_id not in id_to_idx: return None, None, None
                
                n_cd = color_data[id_to_idx[n_id]]
                
                dl = cd['L_mean'] - n_cd['L_mean']
                da = cd['a_mean'] - n_cd['a_mean']
                db = cd['b_mean'] - n_cd['b_mean']
                
                de = np.sqrt(dl**2 + da**2 + db**2)
                return dl, db, de
                
            dl_m, db_m, de_m = compute_diff(mesial_neighbor)
            dl_d, db_d, de_d = compute_diff(distal_neighbor)
            
            cd['delta_L_mesial'] = dl_m
            cd['delta_b_mesial'] = db_m
            cd['delta_E_mesial'] = de_m
            
            cd['delta_L_distal'] = dl_d
            cd['delta_b_distal'] = db_d
            cd['delta_E_distal'] = de_d
            
            is_outlier = False
            if de_m is not None and de_d is not None:
                if de_m > 3.0 and de_d > 3.0:
                    is_outlier = True
            
            cd['is_outlier_shade'] = is_outlier
            
    process_arch(upper_teeth)
    process_arch(lower_teeth)
    
    return color_data
