"""
SatQuery AI — Agent 11: LandslideAndGLOFAgent Tools
Problem Statement ID: 26167 (ISRO / SAC)

Implements Montgomery & Dietrich (1994) SHALSTAB infinite-slope
stability physics and glacial lake area acceleration.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import rasterio


def tool_compute_dem_slope_aspect(dem_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Computes real topographic slope in degrees from a DEM raster.

    Returns (slope_deg, valid_mask). valid_mask is False wherever the source
    pixel was nodata. slope_deg is still numerically 0.0 at those locations
    (to keep array shapes intact for downstream tools), but callers computing
    statistics MUST exclude them via valid_mask — otherwise nodata voids
    (cloud edges, water bodies, tile edge artifacts) silently read as "flat,
    stable ground" and bias aggregate landslide-risk stats optimistic. This
    was previously happening unconditionally via `np.nan_to_num(..., nan=0.0)`
    feeding straight into SHALSTAB with no mask.
    """
    with rasterio.open(dem_path) as src:
        elevation = src.read(1).astype(np.float64)
        if src.nodata is not None:
            valid_mask = elevation != src.nodata
        else:
            valid_mask = np.ones_like(elevation, dtype=bool)
        elevation_for_grad = np.where(valid_mask, elevation, np.nan)

        res_x = abs(src.res[0])
        if src.crs and src.crs.is_geographic:
            bounds = src.bounds
            mid_lat = (bounds.bottom + bounds.top) / 2.0
            gsd_m = res_x * 111320.0 * np.cos(np.radians(mid_lat))
        else:
            gsd_m = res_x if res_x > 0 else 30.0

    dz_dx = np.gradient(elevation_for_grad, axis=1) / max(0.1, gsd_m)
    dz_dy = np.gradient(elevation_for_grad, axis=0) / max(0.1, gsd_m)
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)

    # np.gradient smears a NaN into its immediate neighbors, so widen the
    # invalid mask to match before zero-filling for shape safety.
    invalid_from_grad = np.isnan(slope_deg)
    valid_mask = valid_mask & ~invalid_from_grad
    slope_deg = np.nan_to_num(slope_deg, nan=0.0)

    return slope_deg, valid_mask


def tool_calculate_shalstab_safety_factor(
    slope_deg: np.ndarray,
    valid_mask: np.ndarray | None = None,
    friction_angle_deg: float = 38.0,
    soil_density_kg_m3: float = 1900.0,
    water_density_kg_m3: float = 1000.0,
    soil_cohesion_kpa: float = 4.5,
    soil_depth_m: float = 1.8
) -> dict[str, Any]:
    """
    Computes Factor of Safety (Fs) using Montgomery-Dietrich SHALSTAB model:
    Fs = (c / (rho_s * g * z * sin(theta) * cos(theta))) + (1 - (rho_w / rho_s)) * (tan(phi) / tan(theta))

    friction_angle_deg / soil_density_kg_m3 / soil_cohesion_kpa / soil_depth_m
    are generic soil-mechanics literature defaults, NOT measured for the
    specific DEM's terrain. Different lithologies (Himalayan gneiss vs.
    Western Ghats laterite vs. coastal alluvium) have materially different
    real values, so these are echoed back in the output (`soil_params_used`)
    rather than presented as site-calibrated.

    valid_mask excludes nodata/void pixels from all statistics.
    """
    if valid_mask is None:
        valid_mask = np.ones_like(slope_deg, dtype=bool)

    theta = np.radians(np.clip(slope_deg, 1.0, 80.0))
    phi = np.radians(friction_angle_deg)
    g = 9.81
    z = soil_depth_m

    c_term = (soil_cohesion_kpa * 1000.0) / (soil_density_kg_m3 * g * z * np.sin(theta) * np.cos(theta) + 1e-6)
    fric_term = (1.0 - (water_density_kg_m3 / soil_density_kg_m3)) * (np.tan(phi) / (np.tan(theta) + 1e-6))

    fs_matrix = c_term + fric_term

    valid_count = int(np.sum(valid_mask))
    total_count = int(valid_mask.size)
    coverage_pct = round((valid_count / total_count) * 100.0, 1) if total_count else 0.0

    if valid_count == 0:
        return {
            "model_name": "Montgomery-Dietrich SHALSTAB (1994)",
            "status": "NO_VALID_PIXELS",
            "note": "DEM contained no valid (non-nodata) pixels — cannot compute SHALSTAB statistics.",
            "valid_pixel_coverage_pct": 0.0,
        }

    fs_valid = fs_matrix[valid_mask]

    critical_unstable_pct = float(np.mean(fs_valid < 1.0) * 100.0)
    moderate_risk_pct = float(np.mean((fs_valid >= 1.0) & (fs_valid < 1.3)) * 100.0)
    stable_pct = float(np.mean(fs_valid >= 1.3) * 100.0)

    return {
        "model_name": "Montgomery-Dietrich SHALSTAB (1994)",
        "mean_factor_of_safety": round(float(np.mean(fs_valid)), 2),
        "critical_unstable_slope_pct": round(critical_unstable_pct, 2),
        "moderate_risk_pct": round(moderate_risk_pct, 2),
        "stable_pct": round(stable_pct, 2),
        "hazard_level": "IMMINENT DEBRIS FLOW RISK" if critical_unstable_pct > 15.0 else "LOCALIZED SLOPE VULNERABILITY",
        "valid_pixel_coverage_pct": coverage_pct,
        "soil_params_used": {
            "friction_angle_deg": friction_angle_deg,
            "soil_density_kg_m3": soil_density_kg_m3,
            "soil_cohesion_kpa": soil_cohesion_kpa,
            "soil_depth_m": soil_depth_m,
            "note": "Generic literature defaults unless overridden — not site-measured soil data.",
        },
    }


def tool_monitor_glacial_lake_expansion(
    lake_area_growth_rate_pct_month: float | None = None
) -> dict[str, Any]:
    """
    Evaluates moraine-dammed glacial lake expansion acceleration to flag GLOF threat.

    Requires a real measured growth rate from multi-temporal SAR or optical imagery.
    When no real measurement is available, returns NO_TEMPORAL_MONITORING_DATA status
    rather than fabricating a growth rate.
    """
    if lake_area_growth_rate_pct_month is None:
        return {
            "status": "NO_TEMPORAL_MONITORING_DATA",
            "glof_breach_risk": "UNMONITORED",
            "note": (
                "GLOF threat assessment requires multi-temporal SAR or optical lake area measurements. "
                "No temporal data was provided. Provide T1/T2 acquisitions over the glacial lake to compute real growth rate."
            ),
        }

    glof_critical = lake_area_growth_rate_pct_month > 10.0
    return {
        "status": "MONITORED",
        "lake_area_growth_rate_pct_month": lake_area_growth_rate_pct_month,
        "glof_breach_risk": "CRITICAL" if glof_critical else "MONITORED_STABLE",
        "downstream_warning_time_hours": 3.5 if glof_critical else 18.0,
        "moraine_dam_status": "Vulnerable to piping / seismic overtopping" if glof_critical else "Structurally Sound",
    }