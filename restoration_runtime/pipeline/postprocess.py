import logging
from typing import Any, List
import cv2
import numpy as np

logger = logging.getLogger(__name__)

def composite_seamless(
    original: np.ndarray,
    generated: np.ndarray,
    mask: np.ndarray,
    method: str = "gaussian",
) -> np.ndarray:
    """Seamlessly blend generated content into original image.
    
    Args:
        original: Original BGR image array
        generated: Generated BGR image array (same size as original)
        mask: Binary mask indicating the region to blend (0 background, 255 foreground)
        method: Blending method ('gaussian', 'poisson', or 'both')
        
    Returns:
        Composited BGR image array
    """
    if original.shape != generated.shape:
        raise ValueError(f"Shape mismatch: original {original.shape}, generated {generated.shape}")
        
    # Ensure mask is binary 0/255 and single channel
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    mask = (mask > 0).astype(np.uint8) * 255
    
    if np.sum(mask) == 0:
        return original.copy()

    if method in ("direct", "none"):
        return generated.copy()

    if method == "gaussian":
        # Gaussian blur the mask edges (feathering) then alpha blend
        mask_float = mask.astype(np.float32) / 255.0
        blurred_mask = cv2.GaussianBlur(mask_float, (15, 15), 0)
        blurred_mask = np.stack([blurred_mask] * 3, axis=-1)
        
        result = original.astype(np.float32) * (1 - blurred_mask) + generated.astype(np.float32) * blurred_mask
        return np.clip(result, 0, 255).astype(np.uint8)
        
    elif method == "poisson":
        # Use cv2.seamlessClone for gradient-domain compositing
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return generated.copy()
            
        # Get bounding box for the mask to find the center
        x, y, w, h = cv2.boundingRect(contours[0])
        center = (x + w // 2, y + h // 2)
        
        try:
            return cv2.seamlessClone(generated, original, mask, center, cv2.NORMAL_CLONE)
        except cv2.error as e:
            logger.error(f"Poisson blending failed, falling back to basic paste: {e}")
            result = original.copy()
            mask_3ch = np.stack([mask] * 3, axis=-1)
            result[mask_3ch > 0] = generated[mask_3ch > 0]
            return result
            
    elif method == "both":
        # Apply gaussian first, then poisson on the result
        gaussian_res = composite_seamless(original, generated, mask, "gaussian")
        return composite_seamless(original, gaussian_res, mask, "poisson")
        
    else:
        raise ValueError(f"Unsupported blending method: {method}")


def apply_edits_sequential(
    original: np.ndarray,
    edit_specs: List[Any],
    engine: Any,
    blend_method: str = "gaussian",
) -> np.ndarray:
    """Apply multiple edits sequentially, each building on previous result.
    
    Args:
        original: Original BGR image array
        edit_specs: List of EditSpecification objects
        engine: Instantiated BrushNetEngine
        blend_method: Method to use for seamless compositing
        
    Returns:
        Final BGR image after all edits
    """
    current_image = original.copy()
    
    for i, spec in enumerate(edit_specs):
        logger.info(f"Applying edit {i+1}/{len(edit_specs)}")
        
        try:
            # Generate new image using the engine
            generated = engine.inpaint(spec, current_image)
            
            # Extract mask to use for blending
            mask = getattr(spec, "defect_mask", None)
            if mask is None:
                mask = getattr(spec, "context_mask", None)
            
            edit_type = getattr(spec, "edit_type", "")
            if edit_type == "porcelain_veneer":
                # Master ceramist engine already applies ultra-precise feathered blending
                current_image = generated
            elif mask is not None:
                # Composite the edit onto the current image
                current_image = composite_seamless(
                    current_image, 
                    generated, 
                    mask, 
                    method=blend_method
                )
            else:
                logger.warning(f"No mask found in edit spec {i+1}, skipping compositing")
                
        except Exception as e:
            logger.error(f"Error applying edit {i+1}: {e}", exc_info=True)
            
    return current_image
