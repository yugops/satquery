"""
SatQuery AI — Agent 10: CycloneAndCoastalRiskAgent
Problem Statement ID: 26167 (ISRO / SAC)
Wing: Cyclone_Storm_Surge
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np

from core.interfaces import BaseAgent
from core.models import ComputationMode
from .tools import (
    tool_measure_seawater_intrusion_distance,
    tool_estimate_cyclone_surge_hydrodynamics,
    tool_assess_soil_salinity_hazard,
)


class CycloneAndCoastalRiskAgent(BaseAgent):
    agent_id = "cyclone_coastal_risk"
    agent_name = "CycloneAndCoastalRiskAgent"
    mempalace_wing = "Wing: Cyclone_Storm_Surge"

    async def _execute(self, **kwargs: Any) -> tuple[dict[str, Any], ComputationMode, float, Optional[str], Optional[str]]:
        district = kwargs.get("district", "Unknown Region")
        district_resolved = district != "Unknown Region"

        surge_mask = kwargs.get("surge_mask") or kwargs.get("mask")
        pressure_drop = kwargs.get("pressure_drop_mbar")
        wind_speed = kwargs.get("wind_speed_kmh")
        coastal_slope_deg = kwargs.get("coastal_slope_deg")
        fetch_m = kwargs.get("fetch_m")
        water_depth_m = kwargs.get("water_depth_m")

        if surge_mask is not None and isinstance(surge_mask, np.ndarray):
            intrusion_km = tool_measure_seawater_intrusion_distance(surge_mask)
            salinity_report = tool_assess_soil_salinity_hazard(intrusion_km, district)
            summary = (
                f"Cyclone coastal impact in {district} (from spatial surge mask): Measured {intrusion_km} km seawater intrusion. "
                f"Soil salinity risk: {salinity_report['hazard_category']} (Recovery ~{salinity_report['estimated_soil_recovery_months']} months)."
            )
            if salinity_report["advisory_is_generic"]:
                summary += " (Generic mitigation advisory — district not matched to a known coastal agro-zone.)"

            data = {
                "salinity_assessment": salinity_report,
                "intrusion_km": intrusion_km,
                "source": "spatial_raster_mask",
                "district_resolved": district_resolved,
            }
            confidence = 0.91 if district_resolved else 0.80
            return data, ComputationMode.RULE_BASED, confidence, summary, None

        elif pressure_drop is not None or wind_speed is not None:
            p_drop = float(pressure_drop or 35.0)
            w_spd = float(wind_speed or 120.0)
            partial_input_note = None
            if pressure_drop is None or wind_speed is None:
                partial_input_note = (
                    "Only one of pressure_drop_mbar/wind_speed_kmh was supplied; the other was "
                    "defaulted. Real cyclones have correlated pressure/wind, so a defaulted value "
                    "can be physically inconsistent with the supplied one — treat this as lower-confidence."
                )

            hydro = tool_estimate_cyclone_surge_hydrodynamics(
                p_drop, w_spd,
                coastal_slope_deg=coastal_slope_deg,
                fetch_m=fetch_m,
                water_depth_m=water_depth_m,
            )
            intrusion_km = hydro["estimated_inland_penetration_km"]
            salinity_report = tool_assess_soil_salinity_hazard(intrusion_km, district)

            summary = (
                f"Cyclone surge hydrodynamics in {district} (\u0394P={p_drop} mbar, Wind={w_spd} km/h): "
                f"Estimated surge height {hydro['estimated_surge_height_m']}m, inland penetration {intrusion_km} km ({hydro['cyclone_intensity_scale']}). "
                f"Soil salinity risk: {salinity_report['hazard_category']}. {hydro['caveat']}"
            )
            if partial_input_note:
                summary += f" {partial_input_note}"

            confidence = 0.88
            if hydro["assumptions_used"]["slope_was_assumed_generic"]:
                confidence -= 0.08
            if partial_input_note:
                confidence -= 0.10
            if not district_resolved:
                confidence -= 0.05
            confidence = round(max(0.5, confidence), 2)

            data = {
                "hydrodynamics": hydro,
                "salinity_assessment": salinity_report,
                "intrusion_km": intrusion_km,
                "source": "inverted_barometer_wind_shear_physics",
                "district_resolved": district_resolved,
                "partial_meteorological_input": partial_input_note is not None,
            }
            return data, ComputationMode.RULE_BASED, confidence, summary, None

        else:
            summary = (
                "No storm surge spatial mask or meteorological parameters (wind/pressure drop) provided — "
                "cannot assess coastal inundation. Provide SAR coastal mask or wind/pressure values."
            )
            return {
                "status": "NO_INPUT",
                "district": district,
            }, ComputationMode.RULE_BASED, 0.0, summary, None