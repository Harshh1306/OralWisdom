"""Photographic diagnostics. Thresholds are advisory, not clinical eligibility."""
import cv2
import numpy as np

def assess_input(image, instances):
    mask = np.zeros(image.shape[:2], np.uint8)
    widths = []
    for item in instances:
        points = np.asarray(item.get('polygon', []), np.int32)
        if points.ndim != 2 or len(points) < 3:
            continue
        cv2.fillPoly(mask, [points], 255)
        widths.append(float(np.ptp(points[:, 0])))
    warnings = []
    if not np.any(mask):
        return dict(status='review', warnings=['No visible tooth regions were measured.'], metrics={})
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    core = cv2.erode(mask, np.ones((3, 3), np.uint8)) > 0
    if not np.any(core): core = mask > 0
    pixels = image[core]
    clipped = float(np.mean(np.all(pixels >= 250, axis=1)))
    dark = float(np.mean(gray[core] < 25))
    detail = float(cv2.Laplacian(gray, cv2.CV_32F)[core].var())
    width = float(np.median(widths))
    if width < 16: warnings.append('Small tooth detail: a closer, sharper original photo may improve veneer edges.')
    if clipped > .25: warnings.append('Many tooth highlights are clipped; use softer lighting for more enamel detail.')
    if dark > .30: warnings.append('The visible teeth are dark; improve lighting before judging shade.')
    if detail < 8: warnings.append('Low visible tooth detail; check focus and avoid compressed screenshots.')
    return dict(status='review' if warnings else 'measured', warnings=warnings,
        metrics=dict(median_tooth_width_px=width, clipped_fraction=clipped,
                     dark_fraction=dark, interior_laplacian_variance=detail),
        calibrated=False, note='Advisory measurements; pose and hidden anatomy are not validated.')
