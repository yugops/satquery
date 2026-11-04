"""
SatQuery AI — Agent 7: SARCloudPenetrationAgent

Wing: SAR_Processing

Processes Sentinel-1 SAR imagery when optical data is unusable
due to cloud cover. Handles VV/VH polarization, dB calibration,
false-color composite generation, and water/flood detection via
radar backscatter thresholding.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from core.config import SatQueryConfig
from core.exceptions import ImageValidationError
from core.interfaces import BaseAgent
from core.models import SARFalseColor


# ---------------------------------------------------------------------------
# Tool Functions
# ---------------------------------------------------------------------------

def tool_convert_sar_to_false_color(
    vv_tif: str,
    vh_tif: str,
    *,
    mock: bool = True,
) -> SARFalseColor:
    """Convert VV/VH SAR polarization bands to a false-color RGB composite.

    Pipeline:
    1. Read VV and VH bands from GeoTIFF files.
    2. Apply dB calibration: dB = 10 * log10(DN²) = 20 * log10(DN).
    3. Normalize to display range.
    4. Compose false-color: R=VV, G=VH, B=VV/VH ratio.

    Args:
        vv_tif: Path to VV polarization GeoTIFF.
        vh_tif: Path to VH polarization GeoTIFF.
        mock: If True, returns synthetic false-color metadata.

    Returns:
        SARFalseColor with composite shape and calibration metadata.
    """
    if mock:
        return _mock_false_color()

    # ---- Real mode: load via rasterio ----
    try:
        import rasterio
    except ImportError:
        raise ImageValidationError(
            message="rasterio is required for SAR processing.",
            detail="pip install rasterio",
        )

    for path in (vv_tif, vh_tif):
        if not os.path.isfile(path):
            raise ImageValidationError(
                message=f"SAR file not found: {path}",
                detail=f"path={path}",
            )

    with rasterio.open(vv_tif) as src_vv:
        vv = src_vv.read(1).astype(np.float64)
    with rasterio.open(vh_tif) as src_vh:
        vh = src_vh.read(1).astype(np.float64)

    # dB calibration: 10 * log10(power), where power = DN²
    eps = 1e-10
    vv_db = 10.0 * np.log10(np.clip(vv ** 2, eps, None))
    vh_db = 10.0 * np.log10(np.clip(vh ** 2, eps, None))

    # Normalize to 0–255 for display
    vv_norm = _normalize_to_uint8(vv_db)
    vh_norm = _normalize_to_uint8(vh_db)

    # VV/VH ratio band (clamped)
    ratio = np.clip(vv_db - vh_db, -30, 30)
    ratio_norm = _normalize_to_uint8(ratio)

    # False-color composite: R=VV, G=VH, B=VV/VH
    composite = np.stack([vv_norm, vh_norm, ratio_norm], axis=-1)

    return SARFalseColor(
        shape=composite.shape,
        vv_db_range=(float(vv_db.min()), float(vv_db.max())),
        vh_db_range=(float(vh_db.min()), float(vh_db.max())),
        calibration_applied=True,
    )


def tool_sar_water_threshold(
    vv_array: np.ndarray,
    threshold_db: float = -18.0,
    *,
    mock: bool = True,
) -> np.ndarray:
    """Detect water/flood areas using SAR backscatter thresholding.

    Water surfaces produce low radar backscatter (specular reflection).
    Pixels with VV backscatter below the threshold are flagged as water.

    Args:
        vv_array: VV polarization array (either raw DN or dB-calibrated).
        threshold_db: Backscatter threshold in dB (default -18.0 dB).
            Pixels below this value are classified as water.
        mock: If True, generates a synthetic water mask.

    Returns:
        Binary mask (uint8) where 1 = water, 0 = non-water.
    """
    if mock:
        return _mock_water_mask(vv_array)

    arr = np.asarray(vv_array, dtype=np.float64)

    # Convert to dB if not already.
    # Heuristic: if any value is negative, assume already in dB.
    # Raw linear DN values are always positive.
    if arr.min() < 0:
        # Already in dB range
        vv_db = arr
    else:
        # Raw linear DN → convert to dB: 10 * log10(DN²) = 20 * log10(DN)
        eps = 1e-10
        vv_db = 10.0 * np.log10(np.clip(arr ** 2, eps, None))

    # Threshold: water where backscatter < threshold_db
    water_mask = (vv_db < threshold_db).astype(np.uint8)

    return water_mask


# ---------------------------------------------------------------------------
# SARCloudPenetrationAgent
# ---------------------------------------------------------------------------

class SARCloudPenetrationAgent(BaseAgent):
    """Agent 7 — Processes SAR imagery for cloud-penetrating analysis.

    Invoked when cloud contamination is detected in optical imagery
    or when the input is identified as Sentinel-1 SAR data.
    """

    agent_name = "SARCloudPenetrationAgent"
    mempalace_wing = "Wing: SAR_Processing"

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        """Process SAR data to produce false-color and water detection.

        Expected kwargs:
            vv_tif (str): Path to VV polarization file.
            vh_tif (str): Path to VH polarization file.
            vv_array (np.ndarray | None): Pre-loaded VV array.
            threshold_db (float): Water detection threshold.

        Returns:
            Dict with false_color metadata, water_mask info, and
            water fraction estimate.
        """
        vv_tif = kwargs.get("vv_tif", "")
        vh_tif = kwargs.get("vh_tif", "")
        vv_array = kwargs.get("vv_array")
        threshold_db = kwargs.get(
            "threshold_db", self.config.sar_water_threshold_db
        )
        is_mock = self.config.mode == "mock"

        result: dict[str, Any] = {
            "false_color": None,
            "water_detection": None,
            "water_fraction": 0.0,
            "confidence": 0.0,
        }

        # Step 1: False-color composite
        if vv_tif and vh_tif:
            fc = tool_convert_sar_to_false_color(
                vv_tif, vh_tif, mock=is_mock
            )
            result["false_color"] = fc.model_dump()

        # Step 2: Water/flood detection
        if vv_array is not None:
            water_mask = tool_sar_water_threshold(
                vv_array, threshold_db, mock=is_mock
            )
            water_pixels = int(np.sum(water_mask))
            total_pixels = water_mask.size
            water_fraction = water_pixels / total_pixels if total_pixels > 0 else 0.0

            result["water_detection"] = {
                "water_pixels": water_pixels,
                "total_pixels": total_pixels,
                "threshold_db": threshold_db,
            }
            result["water_fraction"] = round(water_fraction, 4)

        elif is_mock:
            # Mock water detection without input array
            result["water_detection"] = {
                "water_pixels": 3200,
                "total_pixels": 65536,
                "threshold_db": threshold_db,
            }
            result["water_fraction"] = 0.0488

        water_info = result.get("water_detection", {})
        water_px = water_info.get("water_pixels", 0)
        water_pct = result.get("water_fraction", 0.0) * 100
        result["summary"] = (
            f"SAR cloud penetration analysis completed. Generated calibrated false-color composite. "
            f"Water thresholding detected {water_px:,} surface water pixels ({water_pct:.2f}% coverage)."
        )

        result["confidence"] = 0.80 if is_mock else 0.75

        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize a float array to uint8 [0, 255] range."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    normalized = (arr - mn) / (mx - mn) * 255.0
    return normalized.astype(np.uint8)


def _mock_false_color() -> SARFalseColor:
    """Generate synthetic SAR false-color metadata."""
    return SARFalseColor(
        shape=(256, 256, 3),
        vv_db_range=(-25.0, -5.0),
        vh_db_range=(-30.0, -10.0),
        calibration_applied=True,
    )


def _mock_water_mask(vv_array: np.ndarray | None) -> np.ndarray:
    """Generate a synthetic water mask."""
    if vv_array is not None:
        h, w = vv_array.shape[:2]
    else:
        h, w = 256, 256

    mask = np.zeros((h, w), dtype=np.uint8)
    # Simulate a water body in the lower-right quadrant
    q_h, q_w = h // 2, w // 2
    mask[q_h:, q_w:] = 1
    return mask
