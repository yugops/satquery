"""
SatQuery AI — Agent 13: GISSpatialMeasurementAgent
Problem Statement ID: 26167 (ISRO / SAC)
Wing: GIS_Calculations
"""

from __future__ import annotations

from typing import Any, Optional

from core.interfaces import BaseAgent
from core.models import ComputationMode
from .tools import (
    tool_calculate_pixel_area_sqkm,
    tool_calculate_euclidean_distance,
    tool_calculate_haversine_distance,
    tool_export_kml_document,
)


class GISSpatialMeasurementAgent(BaseAgent):
    agent_id = "gis_spatial_measurement"
    agent_name = "GISSpatialMeasurementAgent"
    mempalace_wing = "Wing: GIS_Calculations"

    async def _execute(self, **kwargs: Any) -> tuple[dict[str, Any], ComputationMode, float, Optional[str], Optional[str]]:
        mask = kwargs.get("mask")
        gsd = kwargs.get("gsd_meters", 10.0)
        features = kwargs.get("features", [])
        coords_pair = kwargs.get("coords_pair")

        if mask is None and coords_pair is None:
            return {}, ComputationMode.RULE_BASED, 0.0, (
                "No mask or coordinates provided — cannot measure spatial features. "
                "No default area was substituted."
            ), None

        data: dict[str, Any] = {}
        if mask is not None:
            area_sqkm = tool_calculate_pixel_area_sqkm(mask, gsd)
            data["area_sqkm"] = area_sqkm
            data["area_hectares"] = round(area_sqkm * 100.0, 2)
            data["gsd_meters"] = gsd

        if coords_pair is not None and len(coords_pair) == 2:
            dist_km = tool_calculate_haversine_distance(coords_pair[0], coords_pair[1])
            data["haversine_distance_km"] = dist_km

        kml_result = tool_export_kml_document(features)
        data["kml_export_available"] = True
        data["kml_snippet"] = kml_result["kml"][:300] + "..."
        data["kml_placemark_count"] = kml_result["placemark_count"]
        if kml_result["skipped_features_missing_coordinates"]:
            data["kml_skipped_features"] = kml_result["skipped_features_missing_coordinates"]

        summary = "Geospatial SI computation: "
        if "area_sqkm" in data:
            summary += f"Surface area {data['area_sqkm']} sq. km (GSD {gsd}m). "
        if "haversine_distance_km" in data:
            summary += f"Distance {data['haversine_distance_km']} km. "
        summary += f"KML layer synthesized ({kml_result['placemark_count']} placemark(s))."
        if kml_result["skipped_features_missing_coordinates"]:
            summary += (
                f" Skipped {len(kml_result['skipped_features_missing_coordinates'])} feature(s) "
                f"missing coordinates rather than fabricating placeholder geometry."
            )

        return data, ComputationMode.RULE_BASED, 1.0, summary, None