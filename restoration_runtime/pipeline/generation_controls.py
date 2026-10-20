"""Pure generation controls shared by API, engine and regression tests."""
import math
import cv2
import numpy as np


def validate_options(seed=42, steps=32, guidance_scale=7.5,
                     conditioning_scale=0.58, control_guidance_end=0.74,
                     quality_mode="fast", conditioning_mode="scaffold"):
    values = dict(seed=seed, steps=steps, guidance_scale=guidance_scale,
                  conditioning_scale=conditioning_scale, control_guidance_end=control_guidance_end)
    limits = dict(seed=(0, 2**32 - 2), steps=(1, 80), guidance_scale=(0, 15),
                  conditioning_scale=(0, 2), control_guidance_end=(0, 1))
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        low, high = limits[name]
        if not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}")
        if name in ("seed", "steps") and int(value) != value:
            raise ValueError(f"{name} must be an integer")
    if quality_mode not in ("fast", "quality"):
        raise ValueError("quality_mode must be fast or quality")
    if conditioning_mode not in ("scaffold", "masked"):
        raise ValueError("conditioning_mode must be scaffold or masked")
    values.update(seed=int(seed), steps=int(steps), quality_mode=quality_mode,
                  conditioning_mode=conditioning_mode)
    return values


def composite_inside(original, generated, mask, feather_px=2.0):
    """Feather inward only. Every pixel outside the allowed mask is exact."""
    allowed = (mask > 0).astype(np.uint8)
    distance = cv2.distanceTransform(allowed, cv2.DIST_L2, 5)
    alpha = np.clip(distance / max(float(feather_px), 1.0), 0, 1)[..., None]
    result = np.rint(generated.astype(np.float32) * alpha + original.astype(np.float32) * (1-alpha))
    result = np.clip(result, 0, 255).astype(np.uint8)
    result[allowed == 0] = original[allowed == 0]
    return result


def conditioning_image(image, mask, mode):
    result = image.copy()
    if mode == "masked":
        result[mask > 0] = 0
    return result


def prepare_prompt(tokenizer, text):
    """Keep user intent in order and report any unavoidable encoder truncation."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Prompts must be non-empty text")
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    budget = tokenizer.model_max_length - 2
    effective = tokenizer.decode(ids[:budget], skip_special_tokens=True) if len(ids) > budget else text
    return effective, {"tokens": len(ids) + 2, "limit": tokenizer.model_max_length,
                       "truncated": len(ids) > budget}
