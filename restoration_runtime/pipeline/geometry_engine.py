"""
geometry_engine.py
──────────────────
Deterministic dental geometry computation from YOLOv11 segmentation masks.

This is the measurement backbone of the smile restoration pipeline.
It computes all geometric metrics that the VLM and Edit Planner depend on:
  - Tooth dimensions (width, height, W/H ratio)
  - PCA-based rotation angles and deviation from ideal
  - Inter-tooth gaps and overlaps
  - Golden Percentage proportions
  - Smile arc fitting (polynomial through incisal edges)
  - Dental midline analysis
  - Bilateral symmetry comparison
  - Contact zone analysis

Design principle: This module is PURELY DETERMINISTIC.
All measurements are mathematical computations on mask polygons.
No ML inference, no subjective judgment.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Optional

import cv2
import numpy as np

from config.dental_standards import (
    GOLDEN_PERCENTAGE,
    WH_RATIOS,
    FDI_ANTERIOR_UPPER,
    FDI_TO_TYPE,
    SMILE_ARC_CONCORDANCE_THRESHOLD,
    evaluate_wh_ratio,
    get_tooth_type,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes for structured output
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ToothGeometry:
    """Geometric measurements for a single tooth."""
    tooth_id: str                          # e.g. "T01"
    fdi_number: Optional[int] = None       # e.g. 11
    universal_number: Optional[int] = None # e.g. 8
    tooth_type: str = "unknown"            # central/lateral/canine/premolar
    arch: str = "upper"                    # upper/lower

    # Dimensions
    width_px: float = 0.0                  # mesiodistal (oriented)
    height_px: float = 0.0                 # incisogingival (oriented)
    wh_ratio: float = 0.0                 # width / height
    wh_assessment: str = ""               # "in_range" / "narrow" / "wide"
    area_px: float = 0.0

    # Rotation
    rotation_deg: float = 0.0             # PCA long-axis angle
    rotation_deviation_deg: float = 0.0   # deviation from expected orientation
    rotation_assessment: str = "normal"   # normal/mild/moderate/severe

    # Position
    centroid: tuple[float, float] = (0.0, 0.0)
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    # Spacing
    gap_mesial_px: float = 0.0            # gap to mesial neighbor (toward midline)
    gap_distal_px: float = 0.0            # gap to distal neighbor (away from midline)
    has_gap_mesial: bool = False
    has_gap_distal: bool = False
    has_overlap_mesial: bool = False
    has_overlap_distal: bool = False

    # Golden Percentage
    golden_pct: float = 0.0               # this tooth's % of total visible width
    golden_pct_target: float = 0.0        # target %
    golden_pct_deviation: float = 0.0     # actual - target

    # Symmetry
    contralateral_fdi: Optional[int] = None
    symmetry_width_ratio: float = 1.0     # this_width / contralateral_width
    symmetry_height_ratio: float = 1.0
    symmetry_area_ratio: float = 1.0

    # Incisal edge
    incisal_edge_y: float = 0.0           # y-coordinate of the incisal edge point

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SmileGeometry:
    """Smile-level geometric analysis."""
    # Smile arc
    arc_type: str = "unknown"              # consonant/flat/reverse
    arc_polynomial_coeffs: list[float] = field(default_factory=list)
    arc_curvature_radius: float = 0.0
    arc_concordance: float = 0.0

    # Midline
    midline_shift_px: float = 0.0
    midline_cant_deg: float = 0.0
    midline_assessment: str = "normal"

    # Golden percentage summary
    golden_pct_score: float = 0.0         # 0-100 overall score
    golden_pct_deviations: dict = field(default_factory=dict)

    # Overall
    num_teeth_upper: int = 0
    num_teeth_lower: int = 0
    total_visible_width_px: float = 0.0
    estimated_px_per_mm: float = 0.0       # rough pixel-to-mm scale estimate

    def to_dict(self) -> dict:
        values = asdict(self)
        for key, value in values.items():
            if isinstance(value, float) and not math.isfinite(value):
                values[key] = None
        return values


@dataclass
class GeometryResult:
    """Complete geometry analysis output."""
    teeth: list[ToothGeometry] = field(default_factory=list)
    smile: SmileGeometry = field(default_factory=SmileGeometry)

    def to_dict(self) -> dict:
        return {
            "teeth": [t.to_dict() for t in self.teeth],
            "smile": self.smile.to_dict(),
        }

    def get_tooth(self, fdi: int) -> Optional[ToothGeometry]:
        """Get tooth geometry by FDI number."""
        for t in self.teeth:
            if t.fdi_number == fdi:
                return t
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main Engine
# ─────────────────────────────────────────────────────────────────────────────

class GeometryEngine:
    """
    Deterministic dental geometry computation from segmentation masks.

    Usage:
        engine = GeometryEngine(instances, image.shape[:2])
        result = engine.analyze()
        # result.teeth -> list of ToothGeometry
        # result.smile -> SmileGeometry
    """

    # Expected long-axis orientation for each tooth type (degrees from vertical)
    # Incisors should be roughly vertical (0°), canines tilted slightly (~5-10°)
    EXPECTED_ORIENTATIONS = {
        "central": 0.0,
        "lateral": 0.0,
        "canine": 7.0,
        "premolar": 5.0,
        "unknown": 0.0,
    }

    # Rotation severity thresholds (degrees of deviation from expected)
    ROTATION_THRESHOLDS = {
        "mild": 5.0,
        "moderate": 10.0,
        "severe": 20.0,
    }

    # Minimum gap to be considered aesthetically significant (pixels)
    GAP_SIGNIFICANCE_PX = 3.0

    def __init__(
        self,
        instances: list[dict],
        image_shape: tuple[int, int],
        image_bgr: Optional[np.ndarray] = None,
    ):
        """
        Args:
            instances: Tooth instances from prediction_to_instances(), must have
                       fdi_number, tooth_type, arch fields (from tooth_numbering).
            image_shape: (height, width) of the original image.
            image_bgr: Optional original image for color analysis.
        """
        self.instances = instances
        self.img_h, self.img_w = image_shape
        self.image_bgr = image_bgr
        self.result = GeometryResult()

    def analyze(self) -> GeometryResult:
        """Run the full geometric analysis pipeline. Returns structured result."""
        logger.info("Starting geometry analysis on %d tooth instances", len(self.instances))

        # Step 1: Build ToothGeometry objects with basic dimensions
        self._build_tooth_geometries()

        # Step 2: Compute oriented dimensions (width = mesiodistal, height = incisogingival)
        self._compute_dimensions()

        # Step 3: Compute rotation deviations
        self._compute_rotations()

        # Step 4: Compute inter-tooth gaps and overlaps
        self._compute_gaps_overlaps()

        # Step 5: Compute Golden Percentage
        self._compute_golden_percentage()

        # Step 6: Fit smile arc through incisal edges
        self._compute_smile_arc()

        # Step 7: Analyze dental midline
        self._compute_midline()

        # Step 8: Bilateral symmetry
        self._compute_symmetry()

        # Step 9: Estimate pixel-to-mm scale
        self._estimate_scale()

        logger.info(
            "Geometry analysis complete: %d upper, %d lower teeth",
            self.result.smile.num_teeth_upper,
            self.result.smile.num_teeth_lower,
        )
        return self.result

    # ── Step 1: Build tooth geometry objects ──────────────────────────────

    def _build_tooth_geometries(self) -> None:
        """Create ToothGeometry for each instance, copying basic fields."""
        for inst in self.instances:
            tg = ToothGeometry(
                tooth_id=inst.get("id", ""),
                fdi_number=inst.get("fdi_number"),
                universal_number=inst.get("universal_number"),
                tooth_type=inst.get("tooth_type", "unknown"),
                arch=inst.get("arch", "upper"),
                area_px=inst.get("area_px", 0.0),
                centroid=tuple(inst.get("centroid", [0.0, 0.0])),
                bbox=tuple(inst.get("bbox", [0.0, 0.0, 0.0, 0.0])),
                rotation_deg=inst.get("principal_axis_angle_deg", 0.0),
            )
            self.result.teeth.append(tg)

        # Count by arch
        self.result.smile.num_teeth_upper = sum(
            1 for t in self.result.teeth if t.arch == "upper"
        )
        self.result.smile.num_teeth_lower = sum(
            1 for t in self.result.teeth if t.arch == "lower"
        )

    # ── Step 2: Oriented dimensions ──────────────────────────────────────

    def _compute_dimensions(self) -> None:
        """
        Compute proper mesiodistal width and incisogingival height.

        For anterior teeth in frontal view:
        - Width = the HORIZONTAL extent of the tooth (mesiodistal)
        - Height = the VERTICAL extent (incisogingival)

        We use minAreaRect for oriented dimensions, but for frontal
        smile photos, the axis-aligned bbox is usually a good approximation
        since teeth are roughly upright.
        """
        for i, inst in enumerate(self.instances):
            tg = self.result.teeth[i]
            poly = np.array(inst["polygon"], dtype=np.float32)

            if len(poly) < 3:
                continue

            # Use axis-aligned bounding box width/height for frontal view
            # This is more reliable than minAreaRect for teeth that are
            # slightly rotated — we want the apparent frontal dimensions
            _, _, bbox_w, bbox_h = cv2.boundingRect(poly.astype(np.int32))

            # Also get oriented rect for comparison
            rect = cv2.minAreaRect(poly)
            ow, oh = rect[1]
            oriented_w = min(ow, oh)  # mesiodistal = shorter axis
            oriented_h = max(ow, oh)  # incisogingival = longer axis

            # For frontal photos, use bbox dimensions as primary
            # (teeth are mostly upright, and bbox gives the visible dimensions)
            tg.width_px = float(bbox_w)
            tg.height_px = float(bbox_h)

            if tg.height_px > 0:
                tg.wh_ratio = tg.width_px / tg.height_px
            else:
                tg.wh_ratio = 0.0

            # Evaluate against dental standards
            if tg.tooth_type in WH_RATIOS:
                assessment = evaluate_wh_ratio(tg.tooth_type, tg.wh_ratio)
                tg.wh_assessment = assessment.get("assessment", "")
            else:
                tg.wh_assessment = "unknown_type"

    # ── Step 3: Rotation ─────────────────────────────────────────────────

    def _compute_rotations(self) -> None:
        """
        Compute rotation deviation from expected orientation.

        The PCA angle (principal_axis_angle_deg) gives the long-axis
        orientation. For anterior teeth in frontal view:
        - Central incisors should be ~0° (vertical)
        - Lateral incisors should be ~0° (vertical)
        - Canines may be slightly tilted ~5-10°
        """
        for tg in self.result.teeth:
            expected = self.EXPECTED_ORIENTATIONS.get(tg.tooth_type, 0.0)
            # Detector PCA angles use horizontal=0, vertical=90.
            vertical_angle = (tg.rotation_deg % 180) - 90
            deviation = abs(vertical_angle - expected)

            # Normalize to 0-90 range (symmetry about vertical)
            if deviation > 90:
                deviation = 180 - deviation

            tg.rotation_deviation_deg = round(deviation, 1)

            # Classify severity
            if deviation < self.ROTATION_THRESHOLDS["mild"]:
                tg.rotation_assessment = "normal"
            elif deviation < self.ROTATION_THRESHOLDS["moderate"]:
                tg.rotation_assessment = "mild"
            elif deviation < self.ROTATION_THRESHOLDS["severe"]:
                tg.rotation_assessment = "moderate"
            else:
                tg.rotation_assessment = "severe"

    # ── Step 4: Gaps and overlaps ────────────────────────────────────────

    def _compute_gaps_overlaps(self) -> None:
        """
        Compute inter-tooth spacing for adjacent teeth within the same arch.

        For each adjacent pair:
        - Find minimum distance between their polygon contours
        - Positive distance = gap, negative = overlap (using signed distance)
        """
        # Group teeth by arch
        upper = [t for t in self.result.teeth if t.arch == "upper"]
        lower = [t for t in self.result.teeth if t.arch == "lower"]

        for arch_teeth in [upper, lower]:
            if len(arch_teeth) < 2:
                continue

            # Sort by centroid x (left to right)
            arch_teeth_sorted = sorted(arch_teeth, key=lambda t: t.centroid[0])

            for j in range(len(arch_teeth_sorted) - 1):
                left_tooth = arch_teeth_sorted[j]
                right_tooth = arch_teeth_sorted[j + 1]

                # Get polygons from instances
                left_poly = self._get_polygon(left_tooth.tooth_id)
                right_poly = self._get_polygon(right_tooth.tooth_id)

                if left_poly is None or right_poly is None:
                    continue

                gap = self._min_contour_distance(left_poly, right_poly)

                # Left tooth's distal gap = right tooth's mesial gap
                left_tooth.gap_distal_px = round(gap, 1)
                right_tooth.gap_mesial_px = round(gap, 1)

                if gap > self.GAP_SIGNIFICANCE_PX:
                    left_tooth.has_gap_distal = True
                    right_tooth.has_gap_mesial = True
                elif gap < -self.GAP_SIGNIFICANCE_PX:
                    left_tooth.has_overlap_distal = True
                    right_tooth.has_overlap_mesial = True

    def _get_polygon(self, tooth_id: str) -> Optional[np.ndarray]:
        """Retrieve polygon array for a tooth by its ID."""
        for inst in self.instances:
            if inst.get("id") == tooth_id:
                return np.array(inst["polygon"], dtype=np.float32)
        return None

    @staticmethod
    def _min_contour_distance(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
        """
        Compute minimum distance between two polygon contours.

        Returns positive value for gap, negative for overlap.
        Uses brute-force nearest point for accuracy.
        """
        # Subsample if polygons are very dense (performance)
        if len(poly_a) > 100:
            indices = np.linspace(0, len(poly_a) - 1, 100, dtype=int)
            poly_a = poly_a[indices]
        if len(poly_b) > 100:
            indices = np.linspace(0, len(poly_b) - 1, 100, dtype=int)
            poly_b = poly_b[indices]

        # Compute pairwise distances
        # poly_a: (N, 2), poly_b: (M, 2)
        diff = poly_a[:, np.newaxis, :] - poly_b[np.newaxis, :, :]  # (N, M, 2)
        dists = np.sqrt((diff ** 2).sum(axis=2))  # (N, M)
        min_dist = float(dists.min())

        # Check if polygons overlap by testing if any point of A is inside B
        # Use pointPolygonTest on a sample of points
        contour_b = poly_b.reshape(-1, 1, 2).astype(np.float32)
        for pt in poly_a[:10]:  # check first 10 points
            result = cv2.pointPolygonTest(contour_b, (float(pt[0]), float(pt[1])), False)
            if result >= 0:  # inside or on edge
                return -min_dist  # negative = overlap

        return min_dist

    # ── Step 5: Golden Percentage ────────────────────────────────────────

    def _compute_golden_percentage(self) -> None:
        """
        Compute each tooth's visible width as a percentage of total
        intercanine width. Compare against target [13%, 15%, 22%, 22%, 15%, 13%].
        """
        upper = [t for t in self.result.teeth if t.arch == "upper"]

        if not upper:
            return

        total_width = sum(t.width_px for t in upper)
        self.result.smile.total_visible_width_px = total_width

        if total_width <= 0:
            return

        deviations = {}

        for tg in upper:
            tg.golden_pct = round((tg.width_px / total_width) * 100, 1)

            # Get target based on tooth type
            if tg.tooth_type in GOLDEN_PERCENTAGE:
                target = GOLDEN_PERCENTAGE[tg.tooth_type]["target"]
            else:
                target = 0.0

            tg.golden_pct_target = target
            tg.golden_pct_deviation = round(tg.golden_pct - target, 1)

            key = f"{tg.tooth_type}_{tg.fdi_number or tg.tooth_id}"
            deviations[key] = tg.golden_pct_deviation

        self.result.smile.golden_pct_deviations = deviations

        # Compute overall score: 100 minus sum of absolute deviations
        total_dev = sum(abs(d) for d in deviations.values())
        self.result.smile.golden_pct_score = round(max(0, 100 - total_dev * 5), 1)

    # ── Step 6: Smile arc ────────────────────────────────────────────────

    def _compute_smile_arc(self) -> None:
        """
        Fit a curve through the incisal edges of upper teeth.

        For each tooth, the incisal edge point is the point in the polygon
        with the maximum y-value (lowest in image, since y increases downward).

        Fit a 2nd-order polynomial: y = ax² + bx + c
        If a > 0 (concave up = curves downward = consonant with lip)
        If a ≈ 0 (flat)
        If a < 0 (convex = reverse)
        """
        upper = sorted(
            [t for t in self.result.teeth if t.arch == "upper"],
            key=lambda t: t.centroid[0],
        )

        if len(upper) < 3:
            self.result.smile.arc_type = "insufficient_data"
            return

        # Find incisal edge points (max y in polygon for each tooth)
        incisal_points = []
        for tg in upper:
            poly = self._get_polygon(tg.tooth_id)
            if poly is not None and len(poly) > 0:
                # Incisal edge = point with maximum y value (bottom of tooth)
                max_y_idx = np.argmax(poly[:, 1])
                incisal_pt = poly[max_y_idx]
                tg.incisal_edge_y = float(incisal_pt[1])
                incisal_points.append(incisal_pt)

        if len(incisal_points) < 3:
            self.result.smile.arc_type = "insufficient_data"
            return

        points = np.array(incisal_points)
        x_vals = points[:, 0]
        y_vals = points[:, 1]

        # Fit 2nd-order polynomial
        try:
            coeffs = np.polyfit(x_vals, y_vals, 2)
            a, b, c = coeffs

            self.result.smile.arc_polynomial_coeffs = [float(a), float(b), float(c)]

            # Compute curvature radius at the vertex
            # R = (1 + y'^2)^(3/2) / |y''|
            # At vertex: y' = 0, so R = 1 / |2a|
            if abs(a) > 1e-10:
                self.result.smile.arc_curvature_radius = round(1.0 / (2.0 * abs(a)), 1)
            else:
                self.result.smile.arc_curvature_radius = float("inf")

            # Classify arc type
            # In image coordinates, y increases downward.
            # Central incisal edges lower in the image give a < 0.
            # This estimates edge curvature; lower-lip consonance needs visual review.
            # a ≈ 0 means flat

            # Normalize 'a' by the x-range to make the threshold scale-independent
            x_range = float(x_vals.max() - x_vals.min())
            if x_range > 0:
                tooth_heights = [t.height_px for t in self.result.teeth if t.arch == 'upper' and t.height_px > 0]
                height = float(np.median(tooth_heights)) if tooth_heights else x_range
                normalized_curvature = abs(a) * (x_range ** 2) / max(height, 1e-6)
            else:
                normalized_curvature = 0.0

            if normalized_curvature < 0.05:  # nearly flat, normalized by tooth height
                self.result.smile.arc_type = "flat"
            elif a < 0:
                self.result.smile.arc_type = "consonant"
            else:
                self.result.smile.arc_type = "reverse"

            logger.debug(
                "Smile arc: a=%.6f, type=%s, curvature=%.1f",
                a, self.result.smile.arc_type, normalized_curvature,
            )

        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning("Smile arc fitting failed: %s", e)
            self.result.smile.arc_type = "error"

    # ── Step 7: Midline ──────────────────────────────────────────────────

    def _compute_midline(self) -> None:
        """
        Analyze dental midline alignment.

        The dental midline is the point between the two central incisors.
        Compare to the image center (horizontal midpoint).

        In a frontal smile photo, the facial midline is approximately
        at the horizontal center of the image.
        """
        # Find the two central incisors
        centrals = [
            t for t in self.result.teeth
            if t.tooth_type == "central" and t.arch == "upper"
        ]

        if len(centrals) < 2:
            self.result.smile.midline_assessment = "insufficient_data"
            return

        # Sort centrals by x to get right and left
        centrals_sorted = sorted(centrals, key=lambda t: t.centroid[0])
        right_central = centrals_sorted[0]  # lower x = patient's right
        left_central = centrals_sorted[1]

        # Dental midline = midpoint between the two centrals
        dental_midline_x = (right_central.centroid[0] + left_central.centroid[0]) / 2
        dental_midline_y = (right_central.centroid[1] + left_central.centroid[1]) / 2

        # Image center as proxy for facial midline
        image_center_x = self.img_w / 2

        # Linear shift (horizontal deviation)
        shift_px = dental_midline_x - image_center_x
        self.result.smile.midline_shift_px = round(shift_px, 1)

        # Angular cant: angle of the line connecting the two central centroids
        dx = left_central.centroid[0] - right_central.centroid[0]
        dy = left_central.centroid[1] - right_central.centroid[1]
        if abs(dx) > 1e-6:
            cant_rad = math.atan2(dy, dx)
            cant_deg = math.degrees(cant_rad)
        else:
            cant_deg = 90.0 if dy > 0 else -90.0

        self.result.smile.midline_cant_deg = round(cant_deg, 1)

        # Assessment
        # Note: we can't convert to mm without a scale reference,
        # so we use pixel-based thresholds relative to central incisor width
        avg_central_width = sum(c.width_px for c in centrals) / len(centrals)
        relative_shift = abs(shift_px) / max(avg_central_width, 1.0)

        if relative_shift < 0.15 and abs(cant_deg) < 2.0:
            self.result.smile.midline_assessment = "normal"
        elif relative_shift < 0.30 and abs(cant_deg) < 4.0:
            self.result.smile.midline_assessment = "mild_deviation"
        else:
            self.result.smile.midline_assessment = "significant_deviation"

    # ── Step 8: Bilateral symmetry ───────────────────────────────────────

    def _compute_symmetry(self) -> None:
        """
        Compare contralateral tooth pairs (11↔21, 12↔22, 13↔23).

        For each pair, compute ratios of width, height, and area.
        Perfect symmetry = 1.0.
        """
        # Define contralateral pairs (FDI)
        PAIRS = [(11, 21), (12, 22), (13, 23)]

        for fdi_r, fdi_l in PAIRS:
            right = self.result.get_tooth(fdi_r)
            left = self.result.get_tooth(fdi_l)

            if right is None or left is None:
                continue

            right.contralateral_fdi = fdi_l
            left.contralateral_fdi = fdi_r

            # Width ratio
            if left.width_px > 0:
                right.symmetry_width_ratio = round(right.width_px / left.width_px, 3)
            if right.width_px > 0:
                left.symmetry_width_ratio = round(left.width_px / right.width_px, 3)

            # Height ratio
            if left.height_px > 0:
                right.symmetry_height_ratio = round(right.height_px / left.height_px, 3)
            if right.height_px > 0:
                left.symmetry_height_ratio = round(left.height_px / right.height_px, 3)

            # Area ratio
            if left.area_px > 0:
                right.symmetry_area_ratio = round(right.area_px / left.area_px, 3)
            if right.area_px > 0:
                left.symmetry_area_ratio = round(left.area_px / right.area_px, 3)

    # ── Step 9: Pixel-to-mm scale estimation ─────────────────────────────

    def _estimate_scale(self) -> None:
        """
        Rough estimate of pixels-per-mm using average central incisor width.

        Average central incisor width ≈ 8.6mm (literature).
        If we can identify central incisors, their average pixel width
        gives us a scale factor.

        This is approximate — real calibration requires a physical reference.
        """
        centrals = [
            t for t in self.result.teeth
            if t.tooth_type == "central" and t.arch == "upper"
        ]

        if centrals:
            avg_width_px = sum(c.width_px for c in centrals) / len(centrals)
            # Average central incisor width = 8.6mm
            if avg_width_px > 0:
                self.result.smile.estimated_px_per_mm = round(avg_width_px / 8.6, 2)

    # ── Utility ──────────────────────────────────────────────────────────

    def get_upper_teeth_sorted(self) -> list[ToothGeometry]:
        """Get upper arch teeth sorted left-to-right by centroid x."""
        return sorted(
            [t for t in self.result.teeth if t.arch == "upper"],
            key=lambda t: t.centroid[0],
        )

    def get_geometry_summary_text(self) -> str:
        """
        Generate a human-readable text summary of all geometry facts.
        This is what gets sent to the VLM as part of the input pack.
        """
        lines = ["=== DENTAL GEOMETRY ANALYSIS ===", ""]

        # Smile-level
        sm = self.result.smile
        lines.append(f"Teeth detected: {sm.num_teeth_upper} upper, {sm.num_teeth_lower} lower")
        lines.append(f"Smile arc: {sm.arc_type}")
        lines.append(f"Midline: shift={sm.midline_shift_px:.1f}px, cant={sm.midline_cant_deg:.1f}°, assessment={sm.midline_assessment}")
        lines.append(f"Golden Percentage score: {sm.golden_pct_score}/100")
        if sm.estimated_px_per_mm > 0:
            lines.append(f"Estimated scale: {sm.estimated_px_per_mm:.1f} px/mm")
        lines.append("")

        # Per-tooth
        lines.append("=== PER-TOOTH MEASUREMENTS ===")
        for tg in self.get_upper_teeth_sorted():
            fdi = tg.fdi_number or "?"
            lines.append(f"\nTooth FDI #{fdi} ({tg.tooth_type}, {tg.tooth_id}):")
            lines.append(f"  Width: {tg.width_px:.0f}px, Height: {tg.height_px:.0f}px")
            lines.append(f"  W/H ratio: {tg.wh_ratio:.3f} ({tg.wh_assessment})")
            lines.append(f"  Rotation: {tg.rotation_deg:.1f}° (deviation: {tg.rotation_deviation_deg:.1f}°, {tg.rotation_assessment})")
            lines.append(f"  Golden %: {tg.golden_pct:.1f}% (target: {tg.golden_pct_target:.1f}%, dev: {tg.golden_pct_deviation:+.1f}%)")

            if tg.has_gap_mesial:
                lines.append(f"  GAP mesial: {tg.gap_mesial_px:.1f}px")
            if tg.has_gap_distal:
                lines.append(f"  GAP distal: {tg.gap_distal_px:.1f}px")
            if tg.has_overlap_mesial:
                lines.append(f"  OVERLAP mesial: {abs(tg.gap_mesial_px):.1f}px")
            if tg.has_overlap_distal:
                lines.append(f"  OVERLAP distal: {abs(tg.gap_distal_px):.1f}px")

            if tg.contralateral_fdi:
                lines.append(f"  Symmetry (vs #{tg.contralateral_fdi}): width={tg.symmetry_width_ratio:.3f}, height={tg.symmetry_height_ratio:.3f}")

        return "\n".join(lines)
