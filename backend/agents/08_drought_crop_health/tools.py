"""
SatQuery AI — Agent 08: DroughtAndCropHealthAgent Tools
Problem Statement ID: 26167 (ISRO / SAC)
"""

from __future__ import annotations

from typing import Any
import numpy as np


def tool_calculate_ndvi(nir_band: np.ndarray, red_band: np.ndarray) -> np.ndarray:
    """Calculates Normalized Difference Vegetation Index (NDVI) = (NIR - Red) / (NIR + Red)."""
    nir = nir_band.astype(np.float32)
    red = red_band.astype(np.float32)
    denom = nir + red
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = np.where(denom > 1e-4, (nir - red) / denom, np.nan)
    return np.clip(ndvi, -1.0, 1.0)


def tool_calculate_ndmi_moisture(nir_band: np.ndarray, swir_band: np.ndarray) -> np.ndarray:
    """Calculates Normalized Difference Moisture Index (NDMI) = (NIR - SWIR) / (NIR + SWIR)."""
    nir = nir_band.astype(np.float32)
    swir = swir_band.astype(np.float32)
    denom = nir + swir
    with np.errstate(divide='ignore', invalid='ignore'):
        ndmi = np.where(denom > 1e-4, (nir - swir) / denom, np.nan)
    return np.clip(ndmi, -1.0, 1.0)


def tool_forecast_crop_loss_risk(ndvi_delta: float, district: str = "Unknown Region") -> dict[str, Any]:
    """Projects agricultural yield loss risk based on negative NDVI anomalies."""
    severity = "SEVERE DROUGHT STRESS" if ndvi_delta < -0.25 else "MODERATE MOISTURE DEFICIT" if ndvi_delta < -0.05 else "NORMAL VEGETATION / HEALTHY CANOPY"
    loss_pct = round(min(85.0, max(0.0, abs(ndvi_delta) * 140.0)), 1) if ndvi_delta < 0 else 0.0

    d_lower = district.lower()
    if "maharashtra" in d_lower or "beed" in d_lower or "marathwada" in d_lower:
        crops = ["Soybean", "Cotton", "Pulses (Tur/Moong)"]
    elif "kerala" in d_lower or "wayanad" in d_lower:
        crops = ["Tea", "Coffee", "Cardamom", "Paddy", "Rubber"]
    elif "assam" in d_lower or "barpeta" in d_lower:
        crops = ["Rice (Paddy)", "Tea", "Jute", "Mustard"]
    elif "odisha" in d_lower or "puri" in d_lower:
        crops = ["Paddy", "Groundnut", "Pulses", "Coconut"]
    else:
        crops = ["Local Cropland / Native Canopy"]

    return {
        "ndvi_anomaly_delta": round(ndvi_delta, 3),
        "drought_severity": severity,
        "projected_yield_loss_pct": loss_pct,
        "primary_impacted_crops": crops,
        "advisory": (
            "Trigger emergency micro-irrigation and PM Fasal Bima Yojana crop assessment."
            if loss_pct > 20.0 else
            f"Canopy health stable in {district}; normal seasonal monitoring active."
        )
    }
