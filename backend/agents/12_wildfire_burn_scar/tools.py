"""
SatQuery AI — Agent 12: WildfireAndBurnScarAgent Tools
Problem Statement ID: 26167 (ISRO / SAC)

Implements NBR SWIR burn scar delineation and a simplified wind-driven
forward wildfire spread heuristic.
"""

from __future__ import annotations

from typing import Any
import numpy as np


def tool_delineate_burn_scar_nbr(nir_b8: np.ndarray, swir_b12: np.ndarray) -> dict[str, Any]:
    """
    Normalized Burn Ratio (NBR) = (NIR - SWIR2) / (NIR + SWIR2).

    Single-date NBR thresholding cannot distinguish burn scar from other
    low-NBR surfaces (open water, wet soil, deep shadow, dark urban surfaces)
    as reliably as bi-temporal dNBR can. This returns a conservative
    "confident_burn" mask (moderate-negative NBR, the typical burn-scar
    range) and separately flags extreme-negative pixels as
    "likely_water_or_shadow" so they are excluded from the fire count instead
    of silently inflating it.
    """
    nir = nir_b8.astype(np.float32)
    swir = swir_b12.astype(np.float32)
    denom = nir + swir
    with np.errstate(divide='ignore', invalid='ignore'):
        nbr = np.where(denom > 1e-4, (nir - swir) / denom, np.nan)
    valid = (nir > 0.01) | (swir > 0.01)

    confident_burn = ((nbr < -0.1) & (nbr > -0.8) & valid).astype(np.uint8)
    likely_water_or_shadow = ((nbr <= -0.8) & valid).astype(np.uint8)

    return {
        "confident_burn_mask": confident_burn,
        "likely_water_or_shadow_mask": likely_water_or_shadow,
        "caveat": (
            "Single-date NBR thresholding, not bi-temporal dNBR. Extreme-negative "
            "pixels (likely water/shadow) are excluded from the burn scar count, "
            "but moderate-negative false positives from wet soil or dark urban "
            "surfaces are still possible without a dedicated water/urban mask."
        ),
    }


_EXPECTED_BAND_DESCRIPTIONS = {
    "nir": ["nir", "b08", "band 8", "near infrared", "near-infrared"],
    "swir2": ["swir2", "swir-2", "b12", "band 12", "shortwave infrared 2"],
}


def verify_band_order(band_descriptions: list, nir_index: int, swir2_index: int) -> dict[str, Any]:
    """
    Best-effort check that the raster's band descriptions/tags (if present)
    match the NIR/SWIR2 band indices this tool is about to index into. Many
    rasters carry no band descriptions at all, in which case this honestly
    reports 'unverifiable' rather than silently assuming the hardcoded
    indices (band 2 = NIR, band 4 = SWIR2, matching this pipeline's custom
    4-band ingestion convention) are correct for whatever file was actually
    handed in.
    """
    def _matches(desc, keywords: list[str]):
        if not desc:
            return None
        d = str(desc).lower()
        return any(k in d for k in keywords)

    nir_desc = band_descriptions[nir_index - 1] if 0 <= nir_index - 1 < len(band_descriptions) else None
    swir_desc = band_descriptions[swir2_index - 1] if 0 <= swir2_index - 1 < len(band_descriptions) else None

    nir_match = _matches(nir_desc, _EXPECTED_BAND_DESCRIPTIONS["nir"])
    swir_match = _matches(swir_desc, _EXPECTED_BAND_DESCRIPTIONS["swir2"])

    if nir_match is None and swir_match is None:
        status = "UNVERIFIABLE_NO_BAND_METADATA"
        note = (
            "Raster carries no band descriptions — NIR=band2/SWIR2=band4 ordering is "
            "ASSUMED based on this pipeline's expected 4-band stack convention, not "
            "verified against the file itself."
        )
    elif nir_match is False or swir_match is False:
        status = "MISMATCH_SUSPECTED"
        note = (
            "Band descriptions in the file do not match the expected NIR/SWIR2 "
            "positions — results may be computed from the wrong bands."
        )
    else:
        status = "VERIFIED"
        note = "Band descriptions in the file confirm expected NIR/SWIR2 positions."

    return {
        "status": status,
        "nir_band_description": nir_desc,
        "swir2_band_description": swir_desc,
        "note": note,
    }


def tool_simplified_wind_spread_estimate(
    burn_area_ha: float,
    wind_speed_kmh: float = 24.0,
    wind_direction_deg: float = 45.0,
    fuel_model: str = "Indian Pine / Dry Deciduous",
    wind_is_real_observation: bool = False,
) -> dict[str, Any]:
    """Simplified wind-driven heuristic. NOT the full Rothermel (1972) model."""
    base_ros = 3.2
    wind_factor = 1.0 + 0.04 * (wind_speed_kmh ** 1.3)
    ros_m_min = round(base_ros * wind_factor, 1)

    forward_distance_6hr_km = round((ros_m_min * 60 * 6) / 1000.0, 1)

    caveat = "Simplified empirical wind estimate; full 11-parameter Rothermel model not evaluated."
    if not wind_is_real_observation:
        caveat += (
            " Wind speed/direction are GENERIC DEFAULTS, not a real observation for "
            "this location/time — spread distance should be treated as illustrative only."
        )

    return {
        "model": "Simplified wind-driven heuristic (NOT full Rothermel 1972)",
        "fuel_type": fuel_model,
        "wind_speed_kmh": wind_speed_kmh,
        "wind_direction_deg": wind_direction_deg,
        "wind_is_real_observation": wind_is_real_observation,
        "rate_of_spread_m_min": ros_m_min,
        "forward_spread_6hr_km": forward_distance_6hr_km,
        "threat_status": "HIGH FORWARD VELOCITY" if ros_m_min > 8.0 else "CONTAINED PROPAGATION",
        "simulation_caveat": caveat,
    }