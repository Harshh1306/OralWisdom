"""Output edge observations, kept separate from an anatomical pass/fail claim."""
import cv2
import numpy as np

def inspect_fidelity(image, design):
    x1,y1,x2,y2 = design['crop_box']
    roi = cv2.resize(image[y1:y2,x1:x2], (512,512), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 35, 90)
    distance = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 5)
    distance = np.minimum(distance, float(np.hypot(*gray.shape)))
    observations = []
    warnings = []
    for region in design['waxup'].get('regions', []):
        mask = region['target_mask'].astype(np.uint8)
        core = cv2.erode(mask, np.ones((5,5),np.uint8)) > 0
        ring = (mask > 0) & ~core
        if np.count_nonzero(core) < 20 or not np.any(ring): continue
        h,w = cv2.boundingRect(mask)[3],cv2.boundingRect(mask)[2]
        scale = max(1., min(h,w))
        dark_border = float(np.mean(gray[ring] < max(20., float(np.median(gray[core]))*.35)))
        edge_distance = float(np.median(distance[ring])/scale)
        clipped = float(np.mean(gray[core] >= 250))
        observations.append(dict(tooth_id=region['tooth_id'], arch=region['arch'],
            boundary_edge_distance_normalized=edge_distance, dark_border_fraction=dark_border,
            clipped_core_fraction=clipped))
        if dark_border > .45: warnings.append(f"{region['tooth_id']}: inspect dark veneer margins")
        if edge_distance > .12: warnings.append(f"{region['tooth_id']}: generated edge may differ from the proposed outline")
    return dict(regions=observations, warnings=warnings, calibrated=False,
        topology_validated=False, note='Edges can be shadows or highlights. These signals do not certify tooth count or anatomy.')
