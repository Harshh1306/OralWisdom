"""
prompt_generator.py
───────────────────
Generates constrained natural-language prompts for BrushNet inpainting.
"""

from __future__ import annotations

from typing import Tuple, Optional

PROMPT_TEMPLATES = {
    "chip_repair": (
        "Restore the missing enamel on the incisal edge of this {tooth_type}. "
        "Natural {tooth_type} morphology with smooth contour. "
        "Match existing tooth color, texture, and translucency."
    ),
    "shade_match": (
        "Adjust the shade of this {tooth_type} to match its neighboring teeth. "
        "Preserve natural tooth texture, surface detail, and translucency."
    ),
    "contour_smooth": (
        "Smooth the irregular contour of this {tooth_type} to achieve natural morphology. "
        "Preserve tooth position and adjacent teeth."
    ),
    "gap_close": (
        "Close the visible gap with natural tooth enamel. "
        "Preserve individual tooth identity and gingival papilla."
    ),
    "rotation_correct": (
        "Correct the rotation of this {tooth_type} to proper alignment. "
        "Natural {tooth_type} morphology facing directly forward. "
        "Preserve tooth size, color, and gingival margin."
    ),
    "wear_repair": (
        "Restore the worn incisal edge of this {tooth_type}. "
        "Natural unworn {tooth_type} edge morphology. "
        "Match existing tooth color and texture."
    ),
    "fracture_repair": (
        "Repair the fractured area of this {tooth_type}. "
        "Restore complete natural {tooth_type} morphology. "
        "Match existing tooth color, texture, and translucency."
    ),
    "structure_restore": (
        "Restore the missing structure of this {tooth_type}. "
        "Natural complete {tooth_type} morphology. "
        "Match neighboring teeth in color, size, and texture."
    ),
}

PRESERVE_SUFFIX = (
    " Preserve: existing tooth position, gingival margin, adjacent teeth, "
    "lip geometry, facial identity, natural lighting. "
    "Do NOT modify: neighboring teeth, gingiva, lips, skin, background."
)

NEGATIVE_PROMPT = (
    "blurry, distorted, unnatural, extra teeth, deformed, artifact, "
    "oversaturated, underexposed, noise, jpeg artifacts, cartoon, drawing"
)


def generate_brushnet_prompt(
    edit_type: str,
    tooth_type: str,
    fdi_number: Optional[int] = None
) -> Tuple[str, str]:
    """
    Generate (positive_prompt, negative_prompt) for BrushNet inpainting.
    """
    template = PROMPT_TEMPLATES.get(
        edit_type,
        "Restore natural appearance and morphology of this {tooth_type}."
    )
    
    positive_prompt = template.format(tooth_type=tooth_type) + PRESERVE_SUFFIX
    if fdi_number:
        positive_prompt = f"Tooth FDI #{fdi_number}: " + positive_prompt
        
    return positive_prompt, NEGATIVE_PROMPT
