"""
vlm_agent.py
────────────
VLM analysis agent with support for Manual Input mode (testing) and API mode.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Tuple

from config.schemas import validate_vlm_output
from config.model_config import VLM_PROVIDER

logger = logging.getLogger(__name__)


class BaseVLMAgent(ABC):
    """Abstract base class for VLM analysis agents."""

    @abstractmethod
    def analyze(
        self,
        image_path: Path,
        geometry: dict,
        instances: list[dict],
        tooth_crops: Optional[list[Path]] = None,
    ) -> dict:
        """Run aesthetic analysis. Returns validated JSON matching VLM_OUTPUT_SCHEMA."""
        ...


class ManualInputAgent(BaseVLMAgent):
    """
    Manual input mode — user provides the analysis JSON directly.
    Used for testing the pipeline without an active VLM API.
    """

    def __init__(self):
        self._manual_analysis: Optional[dict] = None

    def set_analysis(self, analysis: dict, instances: Optional[list[dict]] = None) -> Tuple[bool, List[str]]:
        """Set the manual analysis JSON and validate against schema."""
        is_valid, errors = validate_vlm_output(analysis, instances=instances)
        if is_valid:
            self._manual_analysis = analysis
        return is_valid, errors

    def analyze(
        self,
        image_path: Path,
        geometry: dict,
        instances: list[dict],
        tooth_crops: Optional[list[Path]] = None,
    ) -> dict:
        if self._manual_analysis is None:
            raise ValueError(
                "No manual analysis set. Provide manual_analysis JSON to /analyze."
            )
        return self._manual_analysis


class APIAgent(BaseVLMAgent):
    """Cloud VLM API agent (placeholder for future activation)."""

    def analyze(
        self,
        image_path: Path,
        geometry: dict,
        instances: list[dict],
        tooth_crops: Optional[list[Path]] = None,
    ) -> dict:
        raise NotImplementedError(
            "API VLM agent is not yet active. Set VLM_PROVIDER=manual in .env "
            "and provide analysis JSON manually."
        )


def create_vlm_agent() -> BaseVLMAgent:
    """Factory function to instantiate the appropriate VLM agent."""
    if VLM_PROVIDER == "manual":
        return ManualInputAgent()
    elif VLM_PROVIDER == "api":
        return APIAgent()
    else:
        logger.warning("Unknown VLM_PROVIDER=%s, falling back to manual", VLM_PROVIDER)
        return ManualInputAgent()
