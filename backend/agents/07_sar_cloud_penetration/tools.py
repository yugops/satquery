"""
SatQuery AI — Agent 07: SARCloudPenetrationAgent Tools
Problem Statement ID: 26167 (ISRO / SAC)
Dataset: BigEarthNet-S1
Model: ConvNeXt-Tiny (sar_convnext_shard0.pt)
"""

from __future__ import annotations

import os
from typing import Any
import numpy as np
from PIL import Image

from scipy.ndimage import uniform_filter

from models_registry.sar_convnext import SARModelEngine


_SAR_ENGINE: SARModelEngine | None = None


def tool_apply_lee_filter(image: np.ndarray, window_size: int = 7) -> np.ndarray:
    """Real Lee filter for SAR speckle reduction — adaptive, preserves edges
    better than a plain mean/median filter."""
    img = image.astype(np.float64)
    mean = uniform_filter(img, size=window_size)
    mean_sq = uniform_filter(img ** 2, size=window_size)
    variance = mean_sq - mean ** 2
    overall_variance = variance.mean()
    weights = variance / (variance + overall_variance + 1e-10)
    filtered = mean + weights * (img - mean)
    return filtered.astype(image.dtype)


def get_sar_engine(checkpoint_path: str = "sar_convnext_shard0.pt") -> SARModelEngine:
    global _SAR_ENGINE
    if _SAR_ENGINE is None:
        _SAR_ENGINE = SARModelEngine(checkpoint_path)
    return _SAR_ENGINE


def tool_convert_sar_to_false_color(vv_array: np.ndarray, vh_array: np.ndarray) -> np.ndarray:
    """Compose false-color RGB composite: R=VV_norm, G=VH_norm, B=|VV-VH|_ratio."""
    eps = 1e-10
    vv_db = 10.0 * np.log10(np.clip(vv_array ** 2 if vv_array.max() > 2.0 else vv_array, eps, None))
    vh_db = 10.0 * np.log10(np.clip(vh_array ** 2 if vh_array.max() > 2.0 else vh_array, eps, None))

    r = np.clip((vv_db - (-25.0)) / 25.0 * 255.0, 0, 255).astype(np.uint8)
    g = np.clip((vh_db - (-32.0)) / 27.0 * 255.0, 0, 255).astype(np.uint8)
    b = np.clip((np.abs(vv_db - vh_db)) / 20.0 * 255.0, 0, 255).astype(np.uint8)

    return np.stack([r, g, b], axis=-1)


def tool_sar_water_threshold(vv_array: np.ndarray, threshold_db: float = -18.0) -> np.ndarray:
    """Detects water pixels where radar backscatter < threshold_db."""
    eps = 1e-10
    if vv_array.min() >= 0:
        vv_db = 10.0 * np.log10(np.clip(vv_array ** 2 if vv_array.max() > 2.0 else vv_array, eps, None))
    else:
        vv_db = vv_array
    return (vv_db < threshold_db).astype(np.uint8)


def tool_run_sar_convnext_inference(vv_array: np.ndarray, vh_array: np.ndarray, checkpoint_path: str = "sar_convnext_shard0.pt") -> dict[str, Any]:
    """Runs forward pass through trained ConvNeXt-Tiny SAR model."""
    engine = get_sar_engine(checkpoint_path)
    return engine.predict(vv_array, vh_array)
