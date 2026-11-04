"""
SatQuery AI — Agent 09: FloodAndInundationAgent
Problem Statement ID: 26167 (ISRO / SAC)
Wing: Flood_Disaster_Logs

FIX LOG (this pass):
- Synthetic mask reconstruction (from a scalar SAR water-fraction) now
  builds a genuine 2D square array instead of a flat 1D array, so it can
  actually be intersected against a real 2D permanent-water baseline instead
  of silently discarding it on a shape mismatch (see tools.py fix).
- Now accepts and forwards a real `bbox` through to infrastructure impact
  assessment, so corridor names are location-specific (via OSM Overpass)
  rather than the previous fixed NH-31/SH-12/East Coast Railway template.
"""

from __future__ import annotations

import math
from typing import Any, Optional
import numpy as np

from core.interfaces import BaseAgent
from core.models import ComputationMode
from .tools import tool_delineate_flood_extent, tool_intersect_with_infrastructure


class FloodAndInundationAgent(BaseAgent):
    agent_id = "flood_inundation"
    agent_name = "FloodAndInundationAgent"
    mempalace_wing = "Wing: Flood_Disaster_Logs"

    async def _execute(self, **kwargs: Any) -> tuple[dict[str, Any], ComputationMode, float, Optional[str], Optional[str]]:
        district = kwargs.get("district", "Unresolved Region")
        # FIX: real bbox (from GeoValidator's metadata.bounds, forwarded by the
        # orchestrator) now flows through to infrastructure assessment instead
        # of being unavailable entirely.
        bbox = kwargs.get("bbox") or kwargs.get("aoi_bbox")

        sar_water_mask_raw = kwargs.get("water_mask") or kwargs.get("sar_water_mask")
        sar_water_frac = kwargs.get("sar_water_fraction")

        current_mask: np.ndarray | None = None

        def _square_mask_from_fraction(fraction: float, pixel_count: Optional[int] = None) -> np.ndarray:
            """
            FIX: previously built a flat 1D array here. A flat array can never
            match a real 2D permanent-water baseline's shape, so the baseline
            was always silently discarded for this code path. Building a
            genuine (roughly) square 2D array lets the real baseline actually
            be used when one is available.
            """
            if pixel_count and pixel_count > 0:
                total = max(pixel_count, int(pixel_count / max(fraction, 1e-6)))
            else:
                total = 1000 * 1000  # ~100 km² reference grid at 10m GSD
            side = max(1, int(math.isqrt(total)))
            grid = np.zeros((side, side), dtype=np.uint8)
            water_pixels = int(fraction * side * side)
            flat = grid.reshape(-1)
            flat[:water_pixels] = 1
            return flat.reshape(side, side)

        if isinstance(sar_water_mask_raw, np.ndarray):
            current_mask = sar_water_mask_raw
        elif isinstance(sar_water_mask_raw, (int, float)) and sar_water_mask_raw > 0:
            current_mask = _square_mask_from_fraction(
                float(sar_water_mask_raw), kwargs.get("sar_water_pixel_count")
            )
        elif sar_water_frac is not None and isinstance(sar_water_frac, (int, float)):
            current_mask = _square_mask_from_fraction(
                float(sar_water_frac), kwargs.get("sar_water_pixel_count")
            )
        else:
            summary = (
                "No SAR water/flood mask provided — cannot assess inundation extent. "
                "Flood delineation requires SAR Sentinel-1 data or a binary water mask."
            )
            return {
                "extent": {"flood_area_sqkm": 0.0, "inundation_percentage": 0.0, "severity": "UNASSESSED"},
                "infrastructure_cutoffs": [],
                "district": district,
                "status": "NO_INPUT",
            }, ComputationMode.RULE_BASED, 0.0, summary, None

        permanent_water_mask = kwargs.get("permanent_water_mask")
        extent = tool_delineate_flood_extent(current_mask, permanent_water_mask=permanent_water_mask)
        cutoffs = tool_intersect_with_infrastructure(extent["flood_area_sqkm"], district, bbox=bbox)

        baseline_note = ""
        if "estimated_conservative" in extent.get("baseline_source", ""):
            baseline_note = " (Conservative estimate — no permanent water baseline provided; results approximate.)"
        elif "shape_incompatible" in extent.get("baseline_source", ""):
            baseline_note = " (Permanent water baseline provided but shape-incompatible with current mask; ignored.)"

        infra_note = ""
        if cutoffs and cutoffs[0].get("corridor") is None:
            infra_note = f" Infrastructure impact: {cutoffs[0].get('note', 'unavailable')}"
            cutoffs_display = []
        else:
            cutoffs_display = cutoffs
            infra_note = f" Identified {len(cutoffs)} real road-corridor impact assessments (OpenStreetMap)."

        summary = (
            f"Flood inundation mapped: {extent['flood_area_sqkm']} sq. km submerged "
            f"({extent['inundation_percentage']}% of scene){baseline_note}.{infra_note}"
        )

        data = {
            "extent": extent,
            "infrastructure_cutoffs": cutoffs_display if cutoffs and cutoffs[0].get("corridor") is None else cutoffs,
            "district": district,
            "bbox_used_for_infrastructure": bbox is not None,
        }

        return data, ComputationMode.RULE_BASED, 0.82, summary, None