"""
SatQuery AI — Agent 10: CycloneAndCoastalRiskAgent Tools
Problem Statement ID: 26167 (ISRO / SAC)
"""

from __future__ import annotations

from typing import Any
import numpy as np


def tool_measure_seawater_intrusion_distance(coastal_flood_mask: np.ndarray, gsd_meters: float = 10.0) -> float:
    """Calculates perpendicular inland penetration distance of storm surge in kilometers from binary mask.

    LIMITATION: assumes the mask's row axis is roughly aligned with the inland
    direction (coastline running along image columns). If the AOI isn't
    oriented that way this will over/under-estimate penetration distance —
    not fixed here since it requires real coastline-normal geometry, flagged
    for visibility.
    """
    if coastal_flood_mask is None or coastal_flood_mask.size == 0:
        return 0.0
    active_rows = np.where(np.any(coastal_flood_mask > 0, axis=1))[0]
    if len(active_rows) == 0:
        return 0.0
    pixel_depth = float(len(active_rows))
    return round((pixel_depth * gsd_meters) / 1000.0, 2)


# Deltaic / low-lying coastal plains (Bay of Bengal, Gujarat delta systems)
# typically run < 0.15 degrees of slope. The previous default (1.2 degrees)
# was roughly an order of magnitude too steep for the coastlines this project
# actually targets, and silently produced inland-penetration estimates ~40-50x
# smaller than the project's own historical precedent record for Cyclone Fani
# (2.8km real vs ~0.06km predicted with the old default, same wind/pressure
# inputs). A single default can never be right everywhere — so instead of
# just picking a different constant, this now ALWAYS reports what it assumed.
DEFAULT_COASTAL_SLOPE_DEG = 0.12
DEFAULT_FETCH_M = 50000.0
DEFAULT_WATER_DEPTH_M = 20.0


def tool_estimate_cyclone_surge_hydrodynamics(
    pressure_drop_mbar: float = 45.0,
    wind_speed_kmh: float = 130.0,
    coastal_slope_deg: float | None = None,
    fetch_m: float | None = None,
    water_depth_m: float | None = None,
) -> dict[str, Any]:
    """
    Estimates storm surge height and inland runup using inverted barometer + wind shear physics.
    Inverted Barometer Effect: ~1 cm sea level rise per 1 mbar pressure drop.
    Wind Set-Up: Proportional to (Wind Speed)^2 over shallow continental shelf.

    coastal_slope_deg / fetch_m / water_depth_m should come from real
    bathymetry/DEM data when available. When not supplied, this falls back to
    generic deltaic-coast assumptions and explicitly flags that in the return
    value (`assumptions_used`) — never silently presented as site-calibrated.
    """
    used_default_slope = coastal_slope_deg is None
    used_default_fetch = fetch_m is None
    used_default_depth = water_depth_m is None

    slope = coastal_slope_deg if coastal_slope_deg is not None else DEFAULT_COASTAL_SLOPE_DEG
    fetch = fetch_m if fetch_m is not None else DEFAULT_FETCH_M
    depth = water_depth_m if water_depth_m is not None else DEFAULT_WATER_DEPTH_M

    # Barometric component (m)
    baro_surge_m = (pressure_drop_mbar * 100.0) / (1025.0 * 9.81)

    # Wind stress component (m) (wind speed converted to m/s)
    u_ms = wind_speed_kmh / 3.6
    wind_surge_m = (1.225 * 0.0026 * (u_ms ** 2) * fetch) / (1025.0 * 9.81 * depth)

    total_surge_height_m = round(float(baro_surge_m + wind_surge_m), 2)

    # Inland penetration distance (km) based on coastal slope
    slope_rad = np.radians(max(0.02, slope))
    inland_runup_km = round(float((total_surge_height_m / np.sin(slope_rad)) / 1000.0), 2)

    return {
        "estimated_surge_height_m": total_surge_height_m,
        "barometric_component_m": round(float(baro_surge_m), 2),
        "wind_stress_component_m": round(float(wind_surge_m), 2),
        "estimated_inland_penetration_km": inland_runup_km,
        "cyclone_intensity_scale": (
            "Extremely Severe Cyclonic Storm" if wind_speed_kmh > 165
            else "Very Severe Cyclonic Storm" if wind_speed_kmh > 118
            else "Cyclonic Storm"
        ),
        "assumptions_used": {
            "coastal_slope_deg": slope,
            "fetch_m": fetch,
            "water_depth_m": depth,
            "slope_was_assumed_generic": used_default_slope,
            "fetch_was_assumed_generic": used_default_fetch,
            "depth_was_assumed_generic": used_default_depth,
        },
        "caveat": (
            "Wind-setup and inland-penetration figures use generic deltaic-coast "
            "bathymetry assumptions unless real coastal_slope_deg/fetch_m/water_depth_m "
            "are supplied from DEM/bathymetry data. Treat as order-of-magnitude, not survey-grade."
        ) if (used_default_slope or used_default_fetch or used_default_depth) else (
            "Computed using site-supplied bathymetry parameters."
        ),
    }


# Generic salt-tolerant crop guidance by broad coastal agro-zone. Still a
# coarse lookup, not a real agronomy model — but it now actually uses the
# district it's given instead of accepting the parameter and ignoring it
# (the previous version took `district` as an argument and never referenced
# it in the function body, so every location got the same Odisha-specific
# paddy recommendation regardless of where the query was about).
_REGIONAL_SALT_TOLERANT_ADVISORY: dict[str, str] = {
    "odisha": "Apply gypsum treatment; salt-tolerant paddy varieties (CR Dhan 407 / Luna Sankhi).",
    "west bengal": "Apply gypsum treatment; salt-tolerant paddy varieties (Nonabokra, Getu) for Sundarbans belt.",
    "andhra pradesh": "Apply gypsum treatment; salt-tolerant paddy (CSR-36) and aquaculture conversion where salinity is severe.",
    "tamil nadu": "Apply gypsum treatment; salt-tolerant paddy (CO 43) and casuarina shelterbelts.",
    "gujarat": "Apply gypsum/pressmud treatment; salt-tolerant cotton/Bajra varieties for Kutch/Saurashtra coast.",
    "kerala": "Apply organic matter + gypsum; Pokkali salt-tolerant rice varieties for backwater-adjacent paddy.",
    "maharashtra": "Apply gypsum treatment; salt-tolerant paddy and Suru sugarcane rotation for Konkan coast.",
}
_GENERIC_ADVISORY = (
    "Apply gypsum or equivalent calcium amendment to displace sodium; verify locally "
    "appropriate salt-tolerant cultivar with district agricultural extension office — "
    "no region-specific crop guidance available for this location."
)


def tool_assess_soil_salinity_hazard(intrusion_km: float, district: str = "Unknown Region") -> dict[str, Any]:
    """Estimates soil electrical conductivity (EC) increase and agricultural remediation timeframe."""
    estimated_ec_dsm = round(min(16.0, max(2.5, intrusion_km * 4.2)), 1)

    d_clean = district.lower()
    mitigation = _GENERIC_ADVISORY
    matched_region = None
    for region_key, advisory in _REGIONAL_SALT_TOLERANT_ADVISORY.items():
        if region_key in d_clean:
            mitigation = advisory
            matched_region = region_key
            break

    return {
        "seawater_intrusion_km": intrusion_km,
        "soil_salinity_ec_dsm": estimated_ec_dsm,
        "hazard_category": "SEVERE SALINIZATION" if estimated_ec_dsm > 8.0 else "MODERATE SALINITY",
        "groundwater_contamination_risk": "HIGH" if intrusion_km > 2.0 else "LOW",
        "estimated_soil_recovery_months": int(intrusion_km * 5.0) + 6,
        "mitigation_recommendation": mitigation,
        "advisory_region_matched": matched_region,
        "advisory_is_generic": matched_region is None,
    }