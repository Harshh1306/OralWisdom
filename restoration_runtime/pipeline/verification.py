from __future__ import annotations
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    passed: bool = True
    issues: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    attempt: int = 1
    
    def to_dict(self) -> dict:
        def _clean(val):
            if isinstance(val, (np.bool_, bool)):
                return bool(val)
            if isinstance(val, (np.floating, float)):
                return float(val)
            if isinstance(val, (np.integer, int)):
                return int(val)
            if isinstance(val, dict):
                return {k: _clean(v) for k, v in val.items()}
            if isinstance(val, list):
                return [_clean(v) for v in val]
            return val

        return _clean(asdict(self))


class RestorationVerifier:
    """
    Post-edit verification.
    
    Checks:
    1. Unmasked region unchanged (SSIM > threshold)
    2. Edit region actually modified (not a no-op)
    3. No color bleeding outside mask boundary
    4. No extreme pixel artifacts (black patches, white patches, NaN)
    5. Optional: re-run YOLO to verify tooth count preserved
    """
    
    def __init__(self, ssim_threshold: float = 0.95):
        self.ssim_threshold = ssim_threshold

    def check_design_regions(self, original, restored, design, result, metadata):
        """Local artifact signals, never an automated clinical anatomy score."""
        x1,y1,x2,y2 = design['crop_box']
        before = cv2.resize(original[y1:y2,x1:x2],(512,512),interpolation=cv2.INTER_AREA)
        after = cv2.resize(restored[y1:y2,x1:x2],(512,512),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(after,cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_before = cv2.cvtColor(before,cv2.COLOR_BGR2GRAY).astype(np.float32)
        regional = []
        for region in design['waxup'].get('regions',[]):
            interior = cv2.erode(region['target_mask'],np.ones((5,5),np.uint8))>0
            if np.count_nonzero(interior)<20:
                continue
            pixels = after[interior]
            item = dict(tooth_id=region['tooth_id'],operation=region['operation'],
                luminance_std=float(gray[interior].std()),
                clipped_fraction=float(np.mean(np.all(pixels>=250,axis=1))),
                dark_fraction=float(np.mean(np.all(pixels<=20,axis=1))),
                mean_change=float(np.mean(np.abs(gray[interior]-gray_before[interior]))))
            regional.append(item)
            old_clipped=float(np.mean(np.all(before[interior]>=250,axis=1)))
            if item['luminance_std']<1.2 or item['clipped_fraction']>max(.35,old_clipped+.2):
                result.passed=False
                result.issues.append(f"{region['tooth_id']}: flat or excessively clipped crown appearance")
        result.metrics['crown_regions']=regional
        result.metrics['expected_designed_crowns']=len(design['waxup'].get('regions',[]))
        result.metrics['replacement_crowns']=sum(c['presence']=='reviewed_replacement_slot' for c in design['plan']['crowns'])
        result.metrics['automated_topology_validated']=False
        from pipeline.design_fidelity import inspect_fidelity
        result.metrics['design_fidelity']=inspect_fidelity(restored,design)
        if np.any(design['mask']) and metadata.get('engine_type')=='anatomical':
            result.passed=False
            result.issues.append('Color fallback did not complete the requested crown restoration')
        
    def compute_unmasked_ssim(self, original: np.ndarray, restored: np.ndarray, mask: np.ndarray) -> float:
        """Compute structural similarity for unmasked regions."""
        # Ensure mask is binary
        mask = (mask > 0).astype(np.uint8) * 255
        inv_mask = cv2.bitwise_not(mask)
        
        if np.sum(inv_mask) == 0:
            return 1.0 # Fully masked
            
        orig_unmasked = cv2.bitwise_and(original, original, mask=inv_mask)
        rest_unmasked = cv2.bitwise_and(restored, restored, mask=inv_mask)
        
        try:
            from skimage.metrics import structural_similarity as ssim
            score, _ = ssim(orig_unmasked, rest_unmasked, channel_axis=2, full=True)
            return float(score)
        except ImportError:
            logger.warning("skimage not installed. Falling back to simple MSE for SSIM approximation.")
            diff = cv2.absdiff(original, restored).astype(np.float32)
            # Only consider differences in unmasked regions
            mse = np.mean(np.square(diff[inv_mask > 0]))
            # Rough approximation: map max difference to 0 SSIM, no difference to 1.0
            return max(0.0, 1.0 - (mse / 65025.0))
            
    def check_edit_happened(self, original: np.ndarray, restored: np.ndarray, mask: np.ndarray) -> bool:
        """Check if the masked region was actually modified."""
        mask = (mask > 0).astype(np.uint8) * 255
        if np.sum(mask) == 0:
            return False
            
        diff = cv2.absdiff(original, restored)
        mean_diff = diff[mask > 0].mean()
        
        # If the mean difference is very small, it's essentially a no-op
        return mean_diff > 2.0
        
    def check_color_bleed(self, original: np.ndarray, restored: np.ndarray, mask: np.ndarray, ring_px: int = 8) -> bool:
        """Check for color bleeding immediately outside the mask boundary."""
        mask = (mask > 0).astype(np.uint8)
        
        # Dilate the mask to get the ring region
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_px, ring_px))
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        ring_mask = cv2.subtract(dilated_mask, mask)
        
        if np.sum(ring_mask) == 0:
            return True
            
        diff = cv2.absdiff(original, restored)
        mean_diff = float(diff[ring_mask > 0].mean())
        
        # Accommodates soft feather blending and incisal contour leveling
        return mean_diff < 35.0
        
    def check_artifacts(self, restored: np.ndarray, mask: np.ndarray) -> bool:
        """Check for extreme pixel artifacts (pure black/white patches) in the restored region."""
        mask = (mask > 0).astype(np.uint8)
        
        if np.sum(mask) == 0:
            return True
            
        restored_hsv = cv2.cvtColor(restored, cv2.COLOR_BGR2HSV)
        v_channel = restored_hsv[:, :, 2]
        
        masked_v = v_channel[mask > 0]
        if len(masked_v) == 0:
            return True
            
        black_pixels = np.sum(masked_v < 5)
        white_pixels = np.sum(masked_v > 252)
        
        total_pixels = len(masked_v)
        extreme_ratio = (black_pixels + white_pixels) / max(1, total_pixels)
        
        return extreme_ratio < 0.15
        
    def verify(
        self, 
        original: np.ndarray, 
        restored: np.ndarray, 
        edit_specs: List[Any], 
        original_instances: Optional[List[Dict]] = None, 
        original_geometry: Optional[Dict] = None,
        actual_mask: Optional[np.ndarray] = None,
    ) -> VerificationResult:
        """Run all verification checks on the restored image."""
        result = VerificationResult(passed=True, issues=[], metrics={})
        if restored.shape != original.shape or not np.isfinite(restored).all():
            return VerificationResult(passed=False, issues=["Invalid generated image shape or pixels"])
        
        h, w = original.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        for spec in edit_specs:
            for attr in ("defect_mask", "context_mask", "tooth_mask"):
                m = getattr(spec, attr, None)
                if m is not None:
                    combined_mask = cv2.bitwise_or(combined_mask, (m > 0).astype(np.uint8) * 255)

        # Dilate mask for facial unmasked evaluation to cover feathering zone
        if actual_mask is not None:
            combined_mask = (actual_mask > 0).astype(np.uint8) * 255
        eval_mask = combined_mask
        changed = np.any(original != restored, axis=2)
        outside_changes = int(np.count_nonzero(changed & (eval_mask == 0)))
        result.metrics.update(protected_pixels_changed=outside_changes,
                              requires_visual_review=True, anatomy_validated=False)
        if outside_changes:
            result.passed = False
            result.issues.append("Pixels outside the allowed edit mask changed")
                
        # 1. Unmasked region unchanged (facial preservation)
        ssim_score = self.compute_unmasked_ssim(original, restored, eval_mask)
        result.metrics['unmasked_ssim'] = ssim_score
        if ssim_score < self.ssim_threshold:
            result.passed = False
            result.issues.append(f"Unmasked region altered too much (SSIM: {ssim_score:.3f} < {self.ssim_threshold})")
            
        # 2. Edit region actually modified
        edit_happened = self.check_edit_happened(original, restored, combined_mask)
        result.metrics['edit_happened'] = edit_happened
        if not edit_happened:
            result.passed = False
            result.issues.append("Edit region was not meaningfully modified")
            
        # 3. No color bleeding
        no_bleed = self.check_color_bleed(original, restored, eval_mask)
        result.metrics['no_color_bleed'] = no_bleed
        if not no_bleed:
            result.passed = False
            result.issues.append("Color bleeding detected outside mask boundary")
            
        # 4. No extreme pixel artifacts
        no_artifacts = self.check_artifacts(restored, combined_mask)
        result.metrics['no_artifacts'] = no_artifacts
        if not no_artifacts:
            result.passed = False
            result.issues.append("Extreme pixel artifacts detected in edit region")

        # Catch flat painted blocks. This is a technical rejection, not anatomy approval.
        interior = cv2.erode((combined_mask > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        if np.count_nonzero(interior) >= 25:
            gray = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY).astype(np.float32)
            texture_std = float(gray[interior].std())
            result.metrics["edit_luminance_std"] = texture_std
            if texture_std < 1.0:
                result.passed = False
                result.issues.append("Generated region is unnaturally uniform; review required")
            
        return result
