"""
shade_engine.py
───────────────
Realistic teeth shade-change via multi-layer frequency separation in CIELAB.

Algorithm overview
──────────────────
For each tooth mask:
  1. Extract the masked region with a small padding border.
  2. Apply a bilateral filter  →  low_freq  (broad color / tone)
  3. high_freq = original − low_freq  (fine detail, grooves, specular highlights)
  4. Convert low_freq to CIELAB.
  5. Shift L* relatively (brighter areas get proportionally more lightening).
  6. Shift b* relatively (yellower pixels lose proportionally more yellow).
  7. Shift a* absolutely (warmth / coolness tweak).
  8. Convert back to BGR  →  modified_low_freq.
  9. Recomposite: result = modified_low_freq + high_freq.
  10. Feather the mask edges (Gaussian blur) for a smooth blend back into the photo.

This means grooves, enamel micro-texture, and specular highlights are never
altered — only the broad colour tone of each tooth changes, exactly like real
dental bleaching.

VITA Classical shade presets
─────────────────────────────
Inspired by published CIELAB measurements of VITA Classical shades.
Shifts are *relative* to whatever tooth colour is already in the image.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# VITA Classical–inspired presets
#   Keys:   brightness   = L* additive target (relative boost)
#           yellow_red   = fraction to reduce b* (0 = none, 1 = remove all yellow)
#           warmth       = absolute a* shift  (−: cooler/greener, +: warmer/redder)
#   Values chosen to produce perceptually plausible relative shade changes
#   without over-whitening (which looks fake).
# ─────────────────────────────────────────────────────────────────────────────
VITA_PRESETS: dict[str, dict[str, float]] = {
    "original": {"brightness": 0.0,  "yellow_red": 0.0,  "warmth":  0.0},
    # B-group: reddish-yellowish, lightest values
    "B1":       {"brightness": 22.0, "yellow_red": 0.72, "warmth": -0.5},
    "B2":       {"brightness": 16.0, "yellow_red": 0.58, "warmth": -0.3},
    "B3":       {"brightness":  9.0, "yellow_red": 0.40, "warmth":  0.0},
    "B4":       {"brightness":  3.0, "yellow_red": 0.22, "warmth":  0.3},
    # A-group: reddish-brownish
    "A1":       {"brightness": 18.0, "yellow_red": 0.55, "warmth":  0.8},
    "A2":       {"brightness": 11.0, "yellow_red": 0.38, "warmth":  1.0},
    "A3":       {"brightness":  5.0, "yellow_red": 0.20, "warmth":  1.2},
    "A3.5":     {"brightness":  2.0, "yellow_red": 0.10, "warmth":  1.4},
    "A4":       {"brightness": -2.0, "yellow_red": 0.02, "warmth":  1.6},
    # C-group: greyish
    "C1":       {"brightness": 20.0, "yellow_red": 0.65, "warmth": -1.5},
    "C2":       {"brightness": 12.0, "yellow_red": 0.45, "warmth": -1.2},
    "C3":       {"brightness":  4.0, "yellow_red": 0.18, "warmth": -0.8},
    "C4":       {"brightness": -3.0, "yellow_red": 0.05, "warmth": -0.5},
    # D-group: reddish-grey
    "D2":       {"brightness": 14.0, "yellow_red": 0.48, "warmth":  0.5},
    "D3":       {"brightness":  6.0, "yellow_red": 0.28, "warmth":  0.8},
    "D4":       {"brightness":  0.0, "yellow_red": 0.10, "warmth":  1.0},
}


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bilateral_separate(
    region: np.ndarray,
    d: int = 15,
    sigma_color: float = 80.0,
    sigma_space: float = 80.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split *region* (uint8 BGR) into low-frequency (color/tone) and
    high-frequency (detail/texture) layers using a bilateral filter.

    Returns
    -------
    low_freq  : uint8 BGR  – broad color information
    high_freq : int16 BGR  – signed residual detail  (= original − low_freq)
    """
    low_freq = cv2.bilateralFilter(region, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    # Store high-freq as signed int16 so we don't clip during subtraction
    high_freq = region.astype(np.int16) - low_freq.astype(np.int16)
    return low_freq, high_freq


def _shift_lab(
    low_freq: np.ndarray,
    brightness: float,
    yellow_red: float,
    warmth: float,
    mask_roi: np.ndarray,
) -> np.ndarray:
    """
    Apply relative shade shift to the low-frequency colour layer (uint8 BGR).

    Shifts are applied *only* inside mask_roi (same spatial size as low_freq).

    Parameters
    ----------
    low_freq   : uint8 BGR region
    brightness : additive L* boost relative to per-pixel luminance
    yellow_red : fraction of b* to remove  (0–1)
    warmth     : absolute a* shift  (+= warmer / reddish, -= cooler / greenish)
    mask_roi   : uint8 single-channel mask (same HxW as low_freq)
    """
    # OpenCV LAB range: L*∈[0,255], a*∈[0,255] (128=neutral), b*∈[0,255] (128=neutral)
    lab = cv2.cvtColor(low_freq, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = cv2.split(lab)

    # Where the mask is active
    m = mask_roi.astype(np.float32) / 255.0  # soft weight 0..1

    if brightness != 0.0:
        # Relative brightening: pixels already bright get proportionally more lift.
        # Normalise L (0-255 → 0-1), compute relative factor, then add boost.
        L_norm = L / 255.0
        # A sigmoid-like relative scale: darks get less, lights get more
        relative_scale = 0.4 + 0.6 * L_norm
        L_shift = brightness * relative_scale * m
        L = np.clip(L + L_shift, 0, 255)

    if yellow_red != 0.0:
        # b* > 128 means yellow/warm; reduce that excess proportionally.
        b_neutral = 128.0
        b_excess = b - b_neutral  # positive in yellow range
        # Reduce the excess by yellow_red fraction, masked
        b_reduction = b_excess * yellow_red * m
        b = np.clip(b - b_reduction, 0, 255)

    if warmth != 0.0:
        # a* shift: absolute, masked
        a = np.clip(a + warmth * m, 0, 255)

    lab_shifted = cv2.merge([L, a, b]).astype(np.uint8)
    return cv2.cvtColor(lab_shifted, cv2.COLOR_LAB2BGR)


def _feather_mask(mask: np.ndarray, feather_px: int = 9) -> np.ndarray:
    """
    Apply Gaussian feathering to a binary mask for smooth edge blending.
    Returns a float32 mask in [0, 1].
    """
    kernel_size = feather_px * 2 + 1
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), feather_px / 3.0)
    # Re-normalise so the centre stays at ~1
    if blurred.max() > 0:
        blurred = blurred / blurred.max()
    return blurred


def _poly_to_mask(polygon: list[list[float]], height: int, width: int) -> np.ndarray:
    """Rasterise a polygon into a uint8 single-channel mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.round(np.array(polygon, dtype=np.float32)).astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_preset_params(preset_name: str) -> dict[str, float]:
    """
    Return the shade-shift parameters for a VITA preset name.
    Falls back to 'original' if *preset_name* is unknown.
    """
    return dict(VITA_PRESETS.get(preset_name, VITA_PRESETS["original"]))


def list_presets() -> list[str]:
    """Return all available preset names in display order."""
    return list(VITA_PRESETS.keys())


def apply_shade_to_image(
    image: np.ndarray,
    instances: list[dict],
    brightness: float = 18.0,
    yellow_red: float = 0.55,
    warmth: float = 0.8,
    feather_px: int = 9,
) -> np.ndarray:
    """
    Apply realistic shade change to all tooth masks in *image*.

    Parameters
    ----------
    image      : uint8 BGR image (H × W × 3)
    instances  : list of instance dicts from prediction_to_instances()
                 each must have a 'polygon' key (list of [x, y] pairs)
    brightness : L* relative additive shift  (0 = no change, 30 = noticeable whitening)
    yellow_red : fraction of b* yellow excess to remove  (0–1)
    warmth     : a* absolute shift  (0 = neutral, >0 = warmer/redder)
    feather_px : Gaussian feathering radius for mask edges

    Returns
    -------
    result : uint8 BGR image with shade-changed teeth
    """
    if not instances:
        return image.copy()

    h, w = image.shape[:2]
    result = image.copy().astype(np.float32)

    for inst in instances:
        polygon = inst.get("polygon", [])
        if len(polygon) < 3:
            continue

        # ── 1. Build full-image binary mask for this tooth ──────────────────
        tooth_mask = _poly_to_mask(polygon, h, w)

        # ── 2. Bounding box (with small padding) for efficient processing ───
        pts = np.array(polygon, dtype=np.int32)
        x, y, bw, bh = cv2.boundingRect(pts)
        pad = max(10, feather_px + 2)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)

        # ── 3. Extract ROI regions ───────────────────────────────────────────
        roi_orig  = image[y1:y2, x1:x2].copy()
        mask_roi  = tooth_mask[y1:y2, x1:x2]

        if roi_orig.size == 0 or mask_roi.sum() == 0:
            continue

        # ── 4. Frequency separation ──────────────────────────────────────────
        low_freq, high_freq = _bilateral_separate(roi_orig)

        # ── 5. Shift colour in low-frequency layer ───────────────────────────
        modified_low = _shift_lab(low_freq, brightness, yellow_red, warmth, mask_roi)

        # ── 6. Recomposite: modified_low + high_freq ─────────────────────────
        recomposited = np.clip(
            modified_low.astype(np.int16) + high_freq,
            0, 255
        ).astype(np.uint8)

        # ── 7. Blend with feathered mask ─────────────────────────────────────
        alpha = _feather_mask(mask_roi, feather_px=feather_px)  # [0,1] float
        alpha_3ch = alpha[:, :, np.newaxis]

        result[y1:y2, x1:x2] = (
            recomposited.astype(np.float32) * alpha_3ch
            + result[y1:y2, x1:x2] * (1.0 - alpha_3ch)
        )

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_shade_from_preset(
    image: np.ndarray,
    instances: list[dict],
    preset_name: str,
    intensity: float = 1.0,
) -> np.ndarray:
    """
    Convenience wrapper: apply a named VITA preset at a given intensity.

    Parameters
    ----------
    preset_name : one of the keys in VITA_PRESETS
    intensity   : 0.0 = no change, 1.0 = full preset strength
    """
    params = get_preset_params(preset_name)
    return apply_shade_to_image(
        image,
        instances,
        brightness  = params["brightness"]  * intensity,
        yellow_red  = params["yellow_red"]  * intensity,
        warmth      = params["warmth"]      * intensity,
    )
