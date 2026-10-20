"""
vlm_prompts.py
──────────────
System prompts and message templates for the Dental Cosmetic Reasoning Agent (VLM).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

SYSTEM_PROMPT = """You are an AI system specialized in aesthetic analysis of anterior dental photographs for a smile restoration pipeline.

YOUR ROLE: Contextual INTERPRETATION and REASONING only.
You do NOT measure — the geometry engine provides all measurements.
You REASON about what the measurements and visual evidence mean aesthetically.

CRITICAL RULES:
1. Never invent a tooth not present in the segmentation data.
2. Never fabricate numerical measurements — use ONLY the provided geometry.
3. Never alter tooth identity or suggest adding teeth that aren't there.
4. Only propose modifications with sufficient visual + geometric evidence.
5. Separate observed findings from derived geometry from aesthetic interpretation from edit recommendations.

For each tooth:
- Identify observable defects (chip, wear, fracture, discoloration, spacing, rotation).
- State the aesthetic priority (none, low, medium, high).
- Specify edit intent flags:
  * restore_incisal_contour
  * reduce_apparent_rotation
  * match_neighboring_shade
  * close_gap
  * smooth_contour
  * restore_missing_structure
  * preserve_gingival_margin (default: true)
  * preserve_tooth_position (default: true)
  * preserve_root_axis (default: true)

OUTPUT: Strict JSON matching the required schema. No conversational prose."""

EXAMPLE_OUTPUT = {
    "smile_analysis": {
        "smile_arc": "consonant",
        "midline_shift_mm": 0.5,
        "midline_cant_degrees": 1.0,
        "overall_symmetry": "good",
        "confidence": 0.88
    },
    "teeth": [
        {
            "tooth_id": "T01",
            "fdi_number": 11,
            "visibility": "full",
            "observations": {
                "chip": {"present": True, "location": "mesio-incisal", "severity": "mild"},
                "rotation": {"present": False, "severity": None},
                "spacing": {"type": "normal", "severity": "none"},
                "relative_color": {"finding": "normal", "severity": "none"},
                "contour": {"regular": False, "finding": "incisal notch"},
                "wear": {"present": False, "severity": None},
                "fracture": {"present": False, "severity": None},
                "missing_structure": {"present": True, "location": "incisal edge"}
            },
            "confidence": {"overall": 0.92},
            "aesthetic_priority": "high",
            "edit_intent": {
                "restore_incisal_contour": True,
                "reduce_apparent_rotation": False,
                "match_neighboring_shade": False,
                "close_gap": False,
                "smooth_contour": True,
                "restore_missing_structure": True,
                "preserve_gingival_margin": True,
                "preserve_tooth_position": True,
                "preserve_root_axis": True
            }
        }
    ]
}


def build_vlm_messages(
    image_b64: str,
    geometry_text: str,
    schema_text: str = ""
) -> List[Dict[str, Any]]:
    """Build messages list for OpenAI-compatible multimodal VLM API."""
    user_content = [
        {"type": "text", "text": f"Here is the geometric analysis of the smile:\n\n{geometry_text}\n\nPlease analyze the smile image and provide your findings in strict JSON format."},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_b64}"
            }
        }
    ]
    
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
