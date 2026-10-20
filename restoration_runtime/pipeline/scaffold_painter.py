from __future__ import annotations
import logging
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

class ScaffoldPainter:
    """
    Paints geometrically ideal tooth shapes onto the scaffold image before BrushNet processing.
    """

    @classmethod
    def paint(
        cls,
        scaffold_512: np.ndarray,
        mask_512: np.ndarray,
        instances: List[Dict[str, Any]],
        crop_box: Tuple[int, int, int, int],
        tooth_scores: Optional[Dict[int, str]] = None
    ) -> np.ndarray:
        """
        Paints ideal tooth shapes onto the 512x512 scaffold image.
        """
        # In cosmetic dentistry, we NEVER overwrite natural enamel or mirrored CAD/CAM
        # wax-up patches with flat 2D synthetic gradients or artificial embrasure spikes.
        return scaffold_512.copy()

    @classmethod
    def render_bezier_tooth(cls, bbox: Tuple[int, int, int, int], tooth_type: str = 'central') -> np.ndarray:
        x, y, w, h = bbox
        
        # Define control points relative to bounding box
        # Scale to bounding box size
        if tooth_type == 'central':
            cervical_p0, cervical_p3 = (x + w * 0.2, y + h * 0.1), (x + w * 0.8, y + h * 0.1)
            cervical_p1, cervical_p2 = (x + w * 0.4, y - h * 0.1), (x + w * 0.6, y - h * 0.1)
            
            incisal_p0, incisal_p3 = (x + w * 0.1, y + h * 0.9), (x + w * 0.9, y + h * 0.9)
            incisal_p1, incisal_p2 = (x + w * 0.4, y + h * 1.0), (x + w * 0.6, y + h * 1.0)
            
            mesial_p0, mesial_p3 = cervical_p0, incisal_p0
            mesial_p1, mesial_p2 = (x - w * 0.1, y + h * 0.4), (x, y + h * 0.7)
            
            distal_p0, distal_p3 = cervical_p3, incisal_p3
            distal_p1, distal_p2 = (x + w * 1.1, y + h * 0.4), (x + w * 1.0, y + h * 0.7)
            
        elif tooth_type == 'lateral':
            cervical_p0, cervical_p3 = (x + w * 0.25, y + h * 0.1), (x + w * 0.75, y + h * 0.1)
            cervical_p1, cervical_p2 = (x + w * 0.4, y), (x + w * 0.6, y)
            
            incisal_p0, incisal_p3 = (x + w * 0.15, y + h * 0.85), (x + w * 0.85, y + h * 0.85)
            incisal_p1, incisal_p2 = (x + w * 0.4, y + h * 0.95), (x + w * 0.6, y + h * 0.95)
            
            mesial_p0, mesial_p3 = cervical_p0, incisal_p0
            mesial_p1, mesial_p2 = (x, y + h * 0.4), (x + w * 0.05, y + h * 0.6)
            
            distal_p0, distal_p3 = cervical_p3, incisal_p3
            distal_p1, distal_p2 = (x + w, y + h * 0.4), (x + w * 0.95, y + h * 0.6)
            
        else: # canine
            cervical_p0, cervical_p3 = (x + w * 0.3, y + h * 0.1), (x + w * 0.7, y + h * 0.1)
            cervical_p1, cervical_p2 = (x + w * 0.45, y), (x + w * 0.55, y)
            
            # Prominent cusp tip for canine
            incisal_p0, incisal_p3 = (x + w * 0.1, y + h * 0.7), (x + w * 0.9, y + h * 0.7)
            incisal_p1, incisal_p2 = (x + w * 0.5, y + h * 1.1), (x + w * 0.5, y + h * 1.1)
            
            mesial_p0, mesial_p3 = cervical_p0, incisal_p0
            mesial_p1, mesial_p2 = (x, y + h * 0.3), (x + w * 0.05, y + h * 0.5)
            
            distal_p0, distal_p3 = cervical_p3, incisal_p3
            distal_p1, distal_p2 = (x + w, y + h * 0.3), (x + w * 0.95, y + h * 0.5)

        curve_cervical = cls._cubic_bezier(cervical_p0, cervical_p1, cervical_p2, cervical_p3)
        curve_distal = cls._cubic_bezier(distal_p0, distal_p1, distal_p2, distal_p3)
        curve_incisal = cls._cubic_bezier(incisal_p3, incisal_p2, incisal_p1, incisal_p0)
        curve_mesial = cls._cubic_bezier(mesial_p3, mesial_p2, mesial_p1, mesial_p0)
        
        poly = np.vstack((curve_cervical, curve_distal, curve_incisal, curve_mesial))
        return poly

    @classmethod
    def paint_tooth_gradient(cls, canvas: np.ndarray, tooth_mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = bbox
        
        # Create a gradient image for the bounding box
        gradient_lab = np.zeros((h, w, 3), dtype=np.float32)
        
        for i in range(h):
            t = i / float(h - 1) if h > 1 else 0
            
            # Vertical gradient interpolation
            if t < 0.5:
                # cervical to middle
                t_vert = t * 2
                L = 200 * (1 - t_vert) + 225 * t_vert
                a = 8 * (1 - t_vert) + 4 * t_vert
                b = 18 * (1 - t_vert) + 12 * t_vert
            else:
                # middle to incisal
                t_vert = (t - 0.5) * 2
                L = 225 * (1 - t_vert) + 210 * t_vert
                a = 4 * (1 - t_vert) + 0 * t_vert
                b = 12 * (1 - t_vert) + 5 * t_vert
            
            for j in range(w):
                # Horizontal Gaussian falloff
                cx = w / 2.0
                dist = abs(j - cx) / cx if cx > 0 else 0
                falloff = np.exp(-0.5 * (dist / 0.5)**2)
                
                # Darken edges slightly based on falloff
                L_pixel = L * (0.8 + 0.2 * falloff)
                
                gradient_lab[i, j] = [L_pixel, a + 128, b + 128] # OpenCV LAB format shift
        
        gradient_lab = np.clip(gradient_lab, 0, 255).astype(np.uint8)
        gradient_bgr = cv2.cvtColor(gradient_lab, cv2.COLOR_LAB2BGR)
        
        # Alpha blend onto canvas
        result = canvas.copy()
        opacity = 0.65
        
        # We need to overlay gradient_bgr onto the canvas at (x, y) masked by tooth_mask
        canvas_h, canvas_w = canvas.shape[:2]
        
        # Ensure we don't go out of bounds
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(canvas_w, x + w), min(canvas_h, y + h)
        
        grad_x1 = x1 - x
        grad_y1 = y1 - y
        grad_x2 = grad_x1 + (x2 - x1)
        grad_y2 = grad_y1 + (y2 - y1)
        
        if x2 > x1 and y2 > y1:
            crop_grad = gradient_bgr[grad_y1:grad_y2, grad_x1:grad_x2]
            crop_mask = tooth_mask[y1:y2, x1:x2]
            
            mask_indices = crop_mask > 0
            
            roi = result[y1:y2, x1:x2]
            blended = cv2.addWeighted(roi, 1 - opacity, crop_grad, opacity, 0)
            roi[mask_indices] = blended[mask_indices]
            result[y1:y2, x1:x2] = roi
            
        return result

    @classmethod
    def add_specular_highlight(cls, canvas: np.ndarray, tooth_mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = bbox
        result = canvas.copy()
        
        highlight_layer = np.zeros_like(canvas, dtype=np.uint8)
        
        center_x = x + w // 2
        center_y = int(y + h * 0.35)
        
        axes = (max(1, w // 6), max(1, h // 4))
        
        cv2.ellipse(highlight_layer, (center_x, center_y), axes, 0, 0, 360, (255, 255, 255), -1)
        
        highlight_layer = cv2.GaussianBlur(highlight_layer, (0, 0), sigmaX=4, sigmaY=4)
        
        opacity = 0.25
        # Apply only within tooth_mask
        highlight_layer = cv2.bitwise_and(highlight_layer, highlight_layer, mask=tooth_mask)
        
        # Blend
        mask_indices = tooth_mask > 0
        blended = cv2.addWeighted(canvas, 1 - opacity, highlight_layer, opacity, 0)
        result[mask_indices] = blended[mask_indices]
        
        return result

    @classmethod
    def paint_embrasures(cls, canvas: np.ndarray, embrasure_polys: List[np.ndarray]) -> np.ndarray:
        result = canvas.copy()
        dark_color = (20, 20, 20) # Very dark grey/black
        
        for poly in embrasure_polys:
            poly_np = np.array(poly, dtype=np.int32)
            cv2.fillPoly(result, [poly_np], dark_color)
            
        return result

    @staticmethod
    def _cubic_bezier(p0: Tuple[float, float], p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float], num_points: int = 30) -> np.ndarray:
        t = np.linspace(0, 1, num_points)
        t = t[:, np.newaxis]
        
        p0 = np.array(p0)
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        
        curve = (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3
        return curve

    @staticmethod
    def _get_tooth_type(fdi: int) -> str:
        if fdi in [11, 21]:
            return 'central'
        elif fdi in [12, 22]:
            return 'lateral'
        elif fdi in [13, 23]:
            return 'canine'
        return 'central' # default fallback
