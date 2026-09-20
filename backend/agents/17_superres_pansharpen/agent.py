"""
SatQuery AI — Agent 17: SuperResPansharpenAgent
Problem Statement ID: 26167 (ISRO / SAC)
Wing: Super_Resolution
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np

from core.interfaces import BaseAgent
from core.models import ComputationMode
from .tools import tool_gram_schmidt_pansharpen, tool_subpixel_spectral_unmixing


class SuperResPansharpenAgent(BaseAgent):
    agent_id = "superres_pansharpen"
    agent_name = "SuperResPansharpenAgent"
    mempalace_wing = "Wing: Super_Resolution"

    async def _execute(self, **kwargs: Any) -> tuple[dict[str, Any], ComputationMode, float, Optional[str], Optional[str]]:
        return {}, ComputationMode.RULE_BASED, 0.0, (
            "Super-resolution pansharpening is not implemented in this build — "
            "it requires a true high-resolution panchromatic band (e.g. "
            "Cartosat-2S 0.25m Pan), which is not available. No fusion, no "
            "sub-pixel unmixing, and no 0.25m output resolution were produced. "
            "This capability is on the roadmap, not shipped."
        ), None
