"""Photo-only proposals between visible teeth, not inferred clinical tooth identity."""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from pipeline.veneer_templates import silhouette


def propose_teeth(roi, masks, arches, operations, frame, excluded):
    """Fill tooth-sized internal spaces using adjacent geometry and local enamel.

    No extrapolation beyond visible arch endpoints. Proposals remain separately
    identified in the preview and never claim a true missing-tooth count.
    """
    yy, xx = np.mgrid[:roi.shape[0], :roi.shape[1]]
    axis = frame[0]
    u = xx*axis[0] + yy*axis[1]
    all_observed = np.zeros(roi.shape[:2], np.uint8)
    for mask in masks.values(): all_observed |= mask
    proposals = []
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    for arch in ('upper', 'lower'):
        vertical = frame[1] * (1 if arch == 'upper' else -1)
        v = xx*vertical[0] + yy*vertical[1]
        bounds = {}
        for key, mask in masks.items():
            if arches[key] != arch or np.count_nonzero(mask) < 20: continue
            bounds[key] = [*np.percentile(u[mask > 0], [1, 99]),
                           *np.percentile(v[mask > 0], [1, 99])]
        ids = sorted(bounds, key=lambda key: sum(bounds[key][:2]))
        if len(ids) < 2: continue
        typical = float(np.median([bounds[k][1]-bounds[k][0] for k in ids]))
        mid = (bounds[ids[0]][0] + bounds[ids[-1]][1])/2
        count = 0
        for left, right in zip(ids, ids[1:]):
            if any(operations[k] != 'veneer_preview' for k in (left, right)): continue
            a, b = bounds[left], bounds[right]
            gap = b[0]-a[1]
            if gap < .65*typical or gap > 4*typical: continue
            local_width=(a[1]-a[0]+b[1]-b[0])/2
            slots = max(1, min(3, int(np.floor(gap/max(1., local_width)+.5))))
            if count+slots > 6: continue
            pitch = gap/slots
            for index in range(slots):
                center = a[1]+pitch*(index+.5)
                fraction = (center-(a[0]+a[1])/2)/max(1., (b[0]+b[1]-a[0]-a[1])/2)
                top = a[2]*(1-fraction)+b[2]*fraction
                bottom = a[3]*(1-fraction)+b[3]*fraction
                height = max(1., bottom-top)
                ratio = (v-top)/height
                kind = 'central' if abs(center-mid) < 1.1*typical else 'lateral'
                profile = silhouette(np.clip(ratio, 0, 1), arch, kind)
                mask = ((ratio >= 0) & (ratio <= 1) &
                        (np.abs(u-center) <= pitch*.485*profile)).astype(np.uint8)*255
                mask[(all_observed > 0) | (excluded > 0)] = 0
                if np.count_nonzero(mask) < .45*pitch*height: continue
                # Choose the more proportionate adjacent foreground. Its original
                # photographed enamel supplies lighting/texture, not a flat white fill.
                def donor_score(key):
                    d=bounds[key];pixels=gray[masks[key]>0]
                    # Avoid copying a dark collar/shadow into every proposed tooth.
                    shadow=1-float(np.percentile(pixels,20))/255
                    distance=abs((d[0]+d[1])/2-center)/max(1.,bounds[ids[-1]][1]-bounds[ids[0]][0])
                    aspect=abs((d[1]-d[0])/max(1.,d[3]-d[2])-pitch/height)
                    return 2*shadow+.25*distance+.2*aspect
                donor = min((k for k in ids if operations[k]=='veneer_preview'),key=donor_score)
                d = bounds[donor]
                du = d[0] + np.clip((u-center)/max(1., pitch)+.5, 0, 1)*(d[1]-d[0])
                dv = d[2] + np.clip(ratio, 0, 1)*(d[3]-d[2])
                donor_roi = roi.copy()
                foreground = cv2.erode(masks[donor], np.ones((3,3), np.uint8)) > 0
                if not np.any(foreground): foreground = masks[donor] > 0
                _, nearest = distance_transform_edt(~foreground, return_indices=True)
                donor_roi[~foreground] = roi[nearest[0][~foreground], nearest[1][~foreground]]
                mx = (du*axis[0]+dv*vertical[0]).astype(np.float32)
                my = (du*axis[1]+dv*vertical[1]).astype(np.float32)
                texture = cv2.remap(donor_roi, mx, my, cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT_101)
                count += 1
                proposals.append(dict(id=f'preview_{arch}_{count}', arch=arch,
                                      tooth_type=kind, mask=mask, texture=texture,
                                      donor_id=donor, neighbors=[left, right]))
    return proposals
