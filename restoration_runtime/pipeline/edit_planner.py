"""
edit_planner.py
───────────────
Deterministic edit planning from VLM analysis output and geometric measurements.
Generates binary defect and context masks and BrushNet prompts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple

import cv2
import numpy as np

from config.dental_standards import CONFIDENCE_AUTO_EDIT, CONFIDENCE_CAUTIOUS
from pipeline.prompt_generator import generate_brushnet_prompt

logger = logging.getLogger(__name__)


@dataclass
class EditSpecification:
    """A single edit operation to be executed by BrushNet."""
    tooth_id: str
    fdi_number: Optional[int]
    tooth_type: str
    edit_type: str
    
    # Binary masks (H, W uint8: 0 or 255)
    defect_mask: Optional[np.ndarray] = None
    context_mask: Optional[np.ndarray] = None
    tooth_mask: Optional[np.ndarray] = None
    
    # Prompting
    brushnet_prompt: str = ""
    negative_prompt: str = ""
    edge_guide: Optional[np.ndarray] = None
    
    # Meta
    confidence: float = 0.0
    priority: str = "medium"
    severity: str = "moderate"
    description: str = ""

    def to_dict(self) -> dict:
        """Serialize metadata excluding large numpy arrays."""
        return {
            "tooth_id": self.tooth_id,
            "fdi_number": self.fdi_number,
            "tooth_type": self.tooth_type,
            "edit_type": self.edit_type,
            "brushnet_prompt": self.brushnet_prompt,
            "negative_prompt": self.negative_prompt,
            "confidence": self.confidence,
            "priority": self.priority,
            "severity": self.severity,
            "description": self.description,
            "has_defect_mask": self.defect_mask is not None,
            "has_context_mask": self.context_mask is not None,
        }


class EditPlanner:
    """Deterministic edit planner turning VLM output into EditSpecifications."""

    MIN_MASK_AREA = 30
    CONTEXT_DILATION_PX = 8

    def plan(
        self,
        vlm_output: dict,
        geometry: dict,
        instances: list[dict],
        image_shape: Tuple[int, int],
    ) -> List[EditSpecification]:
        """
        Generate a list of EditSpecification objects from VLM findings and geometry.
        """
        h, w = image_shape[:2]
        edit_specs: List[EditSpecification] = []
        
        teeth_analysis = vlm_output.get("teeth", [])
        inst_by_id = {inst.get("id"): inst for inst in instances}
        inst_by_fdi = {inst.get("fdi_number"): inst for inst in instances if inst.get("fdi_number") is not None}

        for tooth in teeth_analysis:
            tooth_id = tooth.get("tooth_id")
            fdi_number = tooth.get("fdi_number")
            
            # Find matching instance: match by FDI first (anatomical truth), then by ID
            inst = None
            if fdi_number is not None:
                try:
                    inst = inst_by_fdi.get(int(fdi_number))
                except (ValueError, TypeError):
                    pass
            if not inst and tooth_id:
                inst = inst_by_id.get(str(tooth_id))
            if not inst:
                continue

            tooth_type = inst.get("tooth_type", "tooth")
            polygon = np.array(inst.get("polygon", []), dtype=np.int32)
            if len(polygon) < 3:
                continue

            tooth_mask = self._create_tooth_mask(polygon, (h, w))
            edit_intent = tooth.get("edit_intent", {})
            observations = tooth.get("observations", {})
            conf_overall = tooth.get("confidence", {}).get("overall", 0.85)

            # Check if porcelain veneer rehabilitation is requested
            wants_veneer = (
                edit_intent.get("apply_porcelain_veneer", False)
                or "veneer" in str(vlm_output.get("smile_analysis", {})).lower()
                or "rehabilitation" in str(vlm_output.get("smile_analysis", {})).lower()
            )

            if wants_veneer:
                fdi_val = fdi_number or inst.get("fdi_number")
                is_upper = True
                if fdi_val is not None:
                    try:
                        is_upper = (int(fdi_val) < 30)
                    except (ValueError, TypeError):
                        pass
                else:
                    centroid = inst.get("centroid", (0, h * 0.5))
                    is_upper = (centroid[1] < h * 0.55)

                is_anterior = (
                    (fdi_val in (11, 21, 12, 22, 13, 23))
                    or (tooth_type in ("central", "lateral", "canine"))
                )

                x, y, bw, bh = cv2.boundingRect(polygon)
                ext_px = 0
                if is_upper and is_anterior:
                    if tooth_type == "central" or fdi_val in (11, 21):
                        ext_px = int(bh * 0.22)
                    elif tooth_type == "lateral" or fdi_val in (12, 22):
                        ext_px = int(bh * 0.16)
                    elif tooth_type == "canine" or fdi_val in (13, 23):
                        ext_px = int(bh * 0.14)
                elif is_upper and (fdi_val in (14, 15, 24, 25) or "premolar" in tooth_type):
                    ext_px = int(bh * 0.08)

                if ext_px > 0:
                    bot_half = np.zeros((h, w), dtype=np.uint8)
                    bot_half[y + int(bh * 0.55):min(h, y + bh + ext_px), max(0, x - 1):min(w, x + bw + 1)] = 255
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, max(3, ext_px * 2 + 1)))
                    dilated = cv2.dilate(tooth_mask, kernel)
                    extended_mask = cv2.bitwise_or(tooth_mask, cv2.bitwise_and(dilated, bot_half))
                    extended_mask = cv2.morphologyEx(extended_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
                else:
                    extended_mask = tooth_mask.copy()

                shade = edit_intent.get("target_shade") or vlm_output.get("smile_analysis", {}).get("target_shade", "OM2")
                prompt = f"porcelain veneer {shade} bleach shade, lifelike translucency, seamless gingival margin"
                neg_prompt = "yellow, stained, chipped, dark, broken, unnatural"
                context_mask = self._dilate_mask(extended_mask, self.CONTEXT_DILATION_PX)

                spec = EditSpecification(
                    tooth_id=str(tooth_id or inst.get("id")),
                    fdi_number=fdi_val,
                    tooth_type=tooth_type,
                    edit_type="porcelain_veneer",
                    defect_mask=extended_mask,
                    context_mask=context_mask,
                    tooth_mask=tooth_mask,
                    brushnet_prompt=prompt,
                    negative_prompt=neg_prompt,
                    confidence=float(conf_overall),
                    priority="high",
                    description=f"Cosmetic appearance preview ({shade}) on {tooth_type} (#{fdi_val or tooth_id}); geometry is approximate"
                )
                edit_specs.append(spec)
                continue

            # Map intents to edit types
            active_edits = []
            if edit_intent.get("restore_incisal_contour") or observations.get("chip", {}).get("present"):
                active_edits.append(("chip_repair", "incisal", "high"))
            if edit_intent.get("match_neighboring_shade"):
                active_edits.append(("shade_match", "surface", "medium"))
            if edit_intent.get("smooth_contour"):
                active_edits.append(("contour_smooth", "margin", "medium"))
            if edit_intent.get("close_gap"):
                active_edits.append(("gap_close", "interdental", "high"))
            if edit_intent.get("reduce_apparent_rotation"):
                active_edits.append(("rotation_correct", "full", "medium"))
            if edit_intent.get("restore_missing_structure"):
                active_edits.append(("structure_restore", "full", "high"))

            for edit_type, locus, priority in active_edits:
                defect_mask = self._generate_defect_mask(tooth_mask, polygon, edit_type, (h, w))
                if defect_mask is None or cv2.countNonZero(defect_mask) < self.MIN_MASK_AREA:
                    defect_mask = tooth_mask.copy()

                context_mask = self._dilate_mask(defect_mask, self.CONTEXT_DILATION_PX)
                prompt, neg_prompt = generate_brushnet_prompt(edit_type, tooth_type, fdi_number)

                spec = EditSpecification(
                    tooth_id=str(tooth_id or inst.get("id")),
                    fdi_number=fdi_number or inst.get("fdi_number"),
                    tooth_type=tooth_type,
                    edit_type=edit_type,
                    defect_mask=defect_mask,
                    context_mask=context_mask,
                    tooth_mask=tooth_mask,
                    brushnet_prompt=prompt,
                    negative_prompt=neg_prompt,
                    confidence=float(conf_overall),
                    priority=priority,
                    description=f"{edit_type.replace('_', ' ').title()} on {tooth_type} (#{fdi_number or tooth_id})"
                )
                edit_specs.append(spec)

        # Sort: high priority first, then highest confidence
        priority_weights = {"high": 3, "medium": 2, "low": 1}
        edit_specs.sort(key=lambda s: (priority_weights.get(s.priority, 1), s.confidence), reverse=True)
        return edit_specs

    def _create_tooth_mask(self, polygon: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        return mask

    def _dilate_mask(self, mask: np.ndarray, px: int) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
        return cv2.dilate(mask, kernel)

    def _generate_defect_mask(
        self,
        tooth_mask: np.ndarray,
        polygon: np.ndarray,
        edit_type: str,
        shape: Tuple[int, int]
    ) -> np.ndarray:
        """Generate targeted binary mask for the specific defect type."""
        h, w = shape
        if edit_type == "chip_repair":
            # Incisal edge is the bottom 25% of the tooth bounding box
            x, y, bw, bh = cv2.boundingRect(polygon)
            incisal_y = y + int(bh * 0.75)
            band = np.zeros(shape, dtype=np.uint8)
            band[incisal_y:y + bh, x:x + bw] = 255
            return cv2.bitwise_and(tooth_mask, band)
            
        elif edit_type == "shade_match":
            # Erode 2px to avoid bleeding into gingiva/lips
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            return cv2.erode(tooth_mask, kernel, iterations=1)
            
        elif edit_type == "contour_smooth":
            # Outer boundary ring of the tooth
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            eroded = cv2.erode(tooth_mask, kernel, iterations=1)
            return cv2.subtract(tooth_mask, eroded)
            
        return tooth_mask.copy()
