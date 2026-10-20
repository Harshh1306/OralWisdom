"""
Smile Aesthetic Knowledge Base.
Encodes ALL dental aesthetic numerical constants and provides evaluation helpers.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Golden Percentage (population-validated, per side, both sides sum to 100%)
GOLDEN_PERCENTAGE: Dict[str, Dict[str, float]] = {
    "central": {"target": 22.0, "min": 21.0, "max": 23.0},
    "lateral": {"target": 15.0, "min": 14.5, "max": 16.0},
    "canine":  {"target": 13.0, "min": 12.0, "max": 13.5},
}

# Width/Height ratio targets
WH_RATIOS: Dict[str, Dict[str, float]] = {
    "central": {"target": 0.78, "min": 0.75, "max": 0.85},
    "lateral": {"target": 0.73, "min": 0.70, "max": 0.75},
    "canine":  {"target": 0.73, "min": 0.70, "max": 0.80},
}

# Contact zone heights (% of central crown height) — 50-40-30 rule
CONTACT_ZONES: Dict[str, float] = {
    "central_central": 0.50,
    "central_lateral": 0.40,
    "lateral_canine":  0.30,
}

# Midline tolerances
MIDLINE_SHIFT_THRESHOLD_MM: float = 2.0
MIDLINE_CANT_THRESHOLD_DEG: float = 2.0

# Gingival zenith offsets
GINGIVAL_ZENITH_LATERAL_OFFSET_MM: Dict[str, float] = {"min": 1.0, "max": 1.5}

# Smile arc concordance
SMILE_ARC_CONCORDANCE_THRESHOLD: float = 0.85

# Edit confidence thresholds  
CONFIDENCE_AUTO_EDIT: float = 0.85
CONFIDENCE_CAUTIOUS: float = 0.65

# FDI numbering for anterior teeth (frontal view, patient's right to left)
FDI_ANTERIOR_UPPER = [13, 12, 11, 21, 22, 23]
FDI_ANTERIOR_LOWER = [43, 42, 41, 31, 32, 33]
UNIVERSAL_ANTERIOR_UPPER = [6, 7, 8, 9, 10, 11]
UNIVERSAL_ANTERIOR_LOWER = [27, 26, 25, 24, 23, 22]

# FDI to tooth type mapping
FDI_TO_TYPE: Dict[int, str] = {
    13: "canine", 12: "lateral", 11: "central",
    21: "central", 22: "lateral", 23: "canine",
    43: "canine", 42: "lateral", 41: "central",
    31: "central", 32: "lateral", 33: "canine",
}

# FDI to Universal mapping
FDI_TO_UNIVERSAL: Dict[int, int] = {
    13: 6, 12: 7, 11: 8, 21: 9, 22: 10, 23: 11,
    43: 27, 42: 26, 41: 25, 31: 24, 32: 23, 33: 22,
}

# Average tooth dimensions in mm (for pixel-to-mm estimation)
AVERAGE_DIMENSIONS_MM: Dict[str, Dict[str, float]] = {
    "central": {"width": 8.6, "height": 10.8},
    "lateral": {"width": 6.8, "height": 9.0},
    "canine":  {"width": 7.8, "height": 10.5},
}

# Incisal embrasure expected progression
EMBRASURE_PROGRESSION = ["smallest", "medium", "largest"]
# central-central < central-lateral < lateral-canine


def evaluate_golden_pct(widths: Dict[str, float]) -> Dict[str, Any]:
    """
    Evaluates the Golden Percentage for a given set of tooth widths.
    
    Args:
        widths (dict[str, float]): Dictionary mapping tooth types to their visible widths.
            Expected keys: 'central', 'lateral', 'canine'.
            
    Returns:
        dict: Deviations from target percentages for each tooth type.
    """
    try:
        total_width = sum(widths.values())
        if total_width <= 0:
            raise ValueError("Total width must be strictly positive.")
            
        results = {}
        for tooth_type, width in widths.items():
            if tooth_type not in GOLDEN_PERCENTAGE:
                logger.warning(f"Tooth type '{tooth_type}' not found in GOLDEN_PERCENTAGE targets.")
                continue
            
            actual_pct = (width / total_width) * 100 * 2 # * 2 since it's per side, both sum to 100
            target_pct = GOLDEN_PERCENTAGE[tooth_type]["target"]
            results[tooth_type] = {
                "actual_percentage": round(actual_pct, 2),
                "deviation_from_target": round(actual_pct - target_pct, 2),
                "in_range": GOLDEN_PERCENTAGE[tooth_type]["min"] <= actual_pct <= GOLDEN_PERCENTAGE[tooth_type]["max"]
            }
        return results
    except Exception as e:
        logger.error(f"Error evaluating golden percentage: {e}")
        return {}


def evaluate_wh_ratio(tooth_type: str, ratio: float) -> Dict[str, Any]:
    """
    Evaluates the width/height ratio against dental standards.
    
    Args:
        tooth_type (str): The type of tooth (e.g., 'central', 'lateral', 'canine').
        ratio (float): The actual calculated width/height ratio.
        
    Returns:
        dict: Evaluation results including in_range flag, deviation, and assessment string.
    """
    if tooth_type not in WH_RATIOS:
        logger.error(f"Unknown tooth type '{tooth_type}' for WH ratio evaluation.")
        return {"in_range": False, "deviation": 0.0, "assessment": "unknown tooth type"}
        
    limits = WH_RATIOS[tooth_type]
    in_range = limits["min"] <= ratio <= limits["max"]
    deviation = ratio - limits["target"]
    
    if in_range:
        assessment = "optimal"
    elif ratio < limits["min"]:
        assessment = "too long/narrow"
    else:
        assessment = "too short/wide"
        
    return {
        "in_range": in_range,
        "deviation": round(deviation, 3),
        "assessment": assessment
    }


def get_tooth_type(fdi: int) -> str:
    """
    Returns the tooth type ('central', 'lateral', 'canine') for a given FDI number.
    
    Args:
        fdi (int): The FDI tooth number.
        
    Returns:
        str: Tooth type or 'unknown' if not mapped.
    """
    return FDI_TO_TYPE.get(fdi, "unknown")


def fdi_to_universal(fdi: int) -> int:
    """
    Converts an FDI tooth number to the Universal numbering system.
    
    Args:
        fdi (int): The FDI tooth number.
        
    Returns:
        int: Universal tooth number or -1 if invalid.
    """
    return FDI_TO_UNIVERSAL.get(fdi, -1)
