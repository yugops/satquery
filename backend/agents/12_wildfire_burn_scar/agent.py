"""
SatQuery AI — Agent 12: WildfireAndBurnScarAgent
Problem Statement ID: 26167 (ISRO / SAC)
Wing: Wildfire_Burn_Scars
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np

from core.interfaces import BaseAgent
from core.models import ComputationMode
from .tools import tool_delineate_burn_scar_nbr, tool_simplified_wind_spread_estimate, verify_band_order


class WildfireAndBurnScarAgent(BaseAgent):
    agent_id = "wildfire_burn_scar"
    agent_name = "WildfireAndBurnScarAgent"
    mempalace_wing = "Wing: Wildfire_Burn_Scars"

    async def _execute(self, **kwargs: Any) -> tuple[dict[str, Any], ComputationMode, float, Optional[str], Optional[str]]:
        image_path = kwargs.get("image_path") or kwargs.get("multiband_tif", "")
        district = kwargs.get("district", "Unknown Region")
        district_resolved = district != "Unknown Region"
        wind_speed_kmh = kwargs.get("wind_speed_kmh")
        wind_direction_deg = kwargs.get("wind_direction_deg")

        if not image_path:
            return {}, ComputationMode.RULE_BASED, 0.0, "No multiband raster provided for wildfire analysis.", None

        import rasterio
        NIR_BAND_IDX, SWIR2_BAND_IDX = 2, 4
        try:
            with rasterio.open(image_path) as src:
                if src.count < 4:
                    return {}, ComputationMode.RULE_BASED, 0.0, (
                        f"Input has only {src.count} band(s) — NBR requires real NIR (Band 2) and SWIR-2 (Band 4) bands."
                    ), None
                band_check = verify_band_order(list(src.descriptions), NIR_BAND_IDX, SWIR2_BAND_IDX)
                nir = src.read(NIR_BAND_IDX).astype(np.float64) / 10000.0
                swir2 = src.read(SWIR2_BAND_IDX).astype(np.float64) / 10000.0
                res_m = abs(src.res[0])
                gsd_m = res_m if res_m > 0 else 10.0
        except Exception as e:
            return {}, ComputationMode.RULE_BASED, 0.0, f"Error reading multiband raster: {e}", None

        if band_check["status"] == "MISMATCH_SUSPECTED":
            return {"band_verification": band_check}, ComputationMode.RULE_BASED, 0.0, (
                f"Refusing to compute NBR — raster band descriptions don't match expected "
                f"NIR/SWIR2 positions ({band_check['note']}). Re-check the input band ordering."
            ), None

        scar_result = tool_delineate_burn_scar_nbr(nir, swir2)
        confident_burn = scar_result["confident_burn_mask"]
        scar_pixels = int(np.sum(confident_burn))
        scar_ha = round((scar_pixels * (gsd_m ** 2)) / 10000.0, 1)
        excluded_pixels = int(np.sum(scar_result["likely_water_or_shadow_mask"]))

        wind_is_real = wind_speed_kmh is not None
        spread = tool_simplified_wind_spread_estimate(
            scar_ha,
            wind_speed_kmh=wind_speed_kmh if wind_speed_kmh is not None else 24.0,
            wind_direction_deg=wind_direction_deg if wind_direction_deg is not None else 45.0,
            wind_is_real_observation=wind_is_real,
        )

        summary = (
            f"Wildfire analysis in {district}: Delineated {scar_ha} hectares confident burn scar (NBR). "
            f"Wind spread estimate: {spread['forward_spread_6hr_km']} km forward threat in 6 hrs ({spread['threat_status']})."
        )
        if excluded_pixels > 0:
            summary += f" Excluded {excluded_pixels} extreme-negative-NBR pixels as likely water/shadow, not fire damage."
        if band_check["status"] == "UNVERIFIABLE_NO_BAND_METADATA":
            summary += " Band identity could not be verified from file metadata — NIR/SWIR2 assignment is assumed."
        if not wind_is_real:
            summary += " Wind spread uses a generic default, not a real observation for this location."
        if not district_resolved:
            summary += " District not resolved."

        data = {
            "burn_scar_hectares": scar_ha,
            "burn_scar_pixels": scar_pixels,
            "likely_water_or_shadow_pixels_excluded": excluded_pixels,
            "gsd_meters": gsd_m,
            "wind_spread_estimate": spread,
            "raster_file": image_path,
            "band_verification": band_check,
            "district_resolved": district_resolved,
        }

        confidence = 0.91
        if band_check["status"] == "UNVERIFIABLE_NO_BAND_METADATA":
            confidence -= 0.10
        if not wind_is_real:
            confidence -= 0.10
        confidence = round(max(0.5, confidence), 2)

        return data, ComputationMode.RULE_BASED, confidence, summary, None