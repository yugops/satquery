"""
SatQuery AI — Agent 22: CloudPenetrationDiffuser Tools
Problem Statement ID: 26167 (ISRO / SAC)
"""

from __future__ import annotations

from typing import Any
import numpy as np


def tool_sar_to_optical_diffusion_synthesis(vv_array: np.ndarray | None = None) -> dict[str, Any]:
    """
    Synthesizes optical RGB reconstruction conditioned on SAR radar backscatter.
    MANDATORY: Watermarked as 'reconstructed_not_observed' for disaster assessment safety.
    """
    return {
        "synthetic_rgb_generated": False,  # was True — no image was ever produced
        "watermark_applied": False,        # was True — nothing to watermark; see note below
        "note": (
            "SAR-to-optical diffusion synthesis is not implemented in this "
            "build — no trained diffusion model exists. No image was "
            "generated. IMPORTANT: if this capability is implemented in "
            "the future, every output image MUST have the "
            "'RECONSTRUCTED (NOT DIRECTLY OBSERVED)' text burned into the "
            "actual pixels (e.g. via PIL ImageDraw), not just set as a "
            "boolean flag in this response — a downstream UI could drop "
            "the flag and display the image as if it were a real "
            "observation. This is a safety requirement, not optional."
        ),
    }
