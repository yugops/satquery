"""
SatQuery AI — Agent 06: BiTemporalChangeAgent
Problem Statement ID: 26167 (ISRO / SAC)
Wing: Change_Detection
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from core.interfaces import BaseAgent
from core.models import ComputationMode
from .tools import tool_crop_spatial_aoi, tool_compute_spectral_change


class BiTemporalChangeAgent(BaseAgent):
    agent_id = "bitemporal_change"
    agent_name = "BiTemporalChangeAgent"
    mempalace_wing = "Wing: Change_Detection"

    async def _execute(self, **kwargs: Any) -> tuple[dict[str, Any], ComputationMode, float, Optional[str], Optional[str]]:
        t1_path = kwargs.get("image_path_t1") or kwargs.get("image_path", "")
        t2_path = kwargs.get("image_path_t2", "")
        bbox = kwargs.get("bbox")

        # CRITICAL HARD GUARD: If T2 is missing or null, DO NOT compute synthetic differences against blank squares
        if not t2_path or not os.path.exists(t2_path) or t1_path == t2_path:
            summary = (
                "Single-temporal acquisition provided. Bi-temporal change detection skipped "
                "(requires two distinct co-registered acquisitions: T1 pre-disaster and T2 post-disaster)."
            )
            data = {
                "status": "SKIPPED_SINGLE_TEMPORAL",
                "change_detected": False,
                "changed_percentage": 0.0,
                "severity": "None",
                "primary_class": "No Secondary Timestamp (T2) Provided",
                "classification": {
                    "changed_percentage": 0.0,
                    "severity": "None",
                    "primary_class": "No Secondary Timestamp (T2) Provided",
                },
                "t1_source": str(t1_path) if t1_path else None,
                "t2_source": None,
            }
            return data, ComputationMode.RULE_BASED, 1.0, summary, None

        # Execute localized high-resolution bi-temporal evaluation
        try:
            arr1, meta1 = tool_crop_spatial_aoi(t1_path, bbox)
            arr2, meta2 = tool_crop_spatial_aoi(t2_path, bbox)
            gsd = meta1.get("gsd", 10.0)

            change_stats = tool_compute_spectral_change(arr1, arr2, gsd_meters=gsd)

            if change_stats.get("affected_area_hectares") is not None:
                summary = (
                    f"Multispectral BOA Bi-Temporal Change ({Path(t1_path).name} -> {Path(t2_path).name}): "
                    f"detected {change_stats['affected_area_hectares']} ha ({change_stats['affected_area_m2']:,.0f} m²) alteration "
                    f"({change_stats['severity']}: {change_stats['primary_class']})."
                )
            elif change_stats.get("is_calibrated_reflectance"):
                summary = (
                    f"Single-Band Calibrated Radiance Delta ({Path(t1_path).name} -> {Path(t2_path).name}): "
                    f"{change_stats['changed_percentage']}% spectral shift ({change_stats['severity']}: {change_stats['primary_class']})."
                )
            else:
                summary = (
                    f"TCI True-Color Visual Divergence ({Path(t1_path).name} -> {Path(t2_path).name}): "
                    f"{change_stats['changed_percentage']}% visual contrast shift ({change_stats['primary_class']}). "
                    f"Quantitative physical area requires calibrated multispectral reflectance."
                )

            data = {
                "status": "COMPLETED",
                "is_calibrated_reflectance": change_stats.get("is_calibrated_reflectance", False),
                "change_detected": change_stats.get("is_calibrated_reflectance", False) and (change_stats.get("debris_pixel_count", 0) > 10),
                "changed_percentage": change_stats["changed_percentage"],
                "affected_area_hectares": change_stats.get("affected_area_hectares"),
                "affected_area_m2": change_stats.get("affected_area_m2"),
                "extent_mismatch_warning": change_stats.get("extent_mismatch_warning"),
                "classification": change_stats,
                "t1_source": str(t1_path),
                "t2_source": str(t2_path),
                "resolution_meters": gsd,
            }

            return data, ComputationMode.RULE_BASED, 0.95, summary, None

        except Exception as e:
            summary = f"Bi-temporal change detection execution error: {e}"
            data = {
                "status": "ERROR",
                "error": str(e),
                "changed_percentage": 0.0,
                "classification": {"changed_percentage": 0.0, "severity": "None", "primary_class": "Error"}
            }
            return data, ComputationMode.RULE_BASED, 0.0, summary, None
