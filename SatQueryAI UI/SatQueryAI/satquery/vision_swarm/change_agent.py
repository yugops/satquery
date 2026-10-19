"""
SatQuery AI — Agent 6: BiTemporalChangeAgent

Wing: Temporal_Analysis

Compares two temporally separated, co-registered satellite images
to detect and classify changes. Uses pixel-wise differencing and
(optionally) a BIT/ChangeFormer model for semantic classification.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.config import SatQueryConfig
from core.exceptions import PairRequiredError
from core.interfaces import BaseAgent, ModelAdapter
from core.models import ChangeClassification, HeatmapResult


# ---------------------------------------------------------------------------
# Change Detection Model Adapter
# ---------------------------------------------------------------------------

class ChangeDetectionAdapter(ModelAdapter):
    """Wraps BIT / ChangeFormer for semantic change classification."""

    def __init__(self, model_id: str = "custom-bit-cd"):
        super().__init__(model_id)

    def load(self, config: SatQueryConfig) -> None:
        """Load change detection model weights."""
        try:
            # Placeholder — actual BIT/ChangeFormer loading logic
            self._loaded = True
        except Exception as exc:
            from core.exceptions import ModelUnavailableError
            raise ModelUnavailableError(
                self.model_id,
                message=f"Failed to load change detection model: {exc}",
            )

    def predict(self, img_t1: np.ndarray, img_t2: np.ndarray) -> dict:
        """Run change classification on an aligned image pair.

        Returns:
            Dict with change_mask (ndarray), categories (list[str]),
            and confidence (float).
        """
        self.ensure_loaded()
        # Real implementation would run BIT/ChangeFormer inference
        # Returning synthetic output as placeholder
        diff = np.abs(img_t1.astype(float) - img_t2.astype(float))
        mask = np.mean(diff, axis=-1) if diff.ndim == 3 else diff
        mask = (mask / (mask.max() + 1e-8))
        return {
            "change_mask": (mask > 0.3).astype(np.uint8),
            "categories": ["general_change"],
            "confidence": 0.7,
        }


# ---------------------------------------------------------------------------
# Tool Functions
# ---------------------------------------------------------------------------

def tool_generate_difference_heatmap(
    img_t1: Any,
    img_t2: Any,
    *,
    mock: bool = True,
) -> HeatmapResult:
    """Compute a pixel-wise difference heatmap between two images.

    Calculates the absolute difference normalized to [0, 1].
    In mock mode, generates a synthetic gradient heatmap.

    Args:
        img_t1: First (earlier) image as np.ndarray.
        img_t2: Second (later) image as np.ndarray.
        mock: If True, returns synthetic heatmap.

    Returns:
        HeatmapResult with shape, value range, and mean change.
    """
    if mock:
        return _mock_heatmap(img_t1)

    arr1 = _to_float_array(img_t1)
    arr2 = _to_float_array(img_t2)

    # Compute per-channel absolute difference, then average across channels
    if arr1.ndim == 3:
        diff = np.mean(np.abs(arr1 - arr2), axis=-1)
    else:
        diff = np.abs(arr1 - arr2)

    # Normalize to [0, 1]
    max_val = diff.max()
    if max_val > 0:
        diff = diff / max_val

    return HeatmapResult(
        heatmap_shape=(diff.shape[0], diff.shape[1]),
        min_val=float(diff.min()),
        max_val=float(diff.max()),
        mean_change=float(diff.mean()),
    )


def tool_classify_change_type(
    heatmap: HeatmapResult | None,
    t1: Any,
    t2: Any,
    *,
    adapter: ChangeDetectionAdapter | None = None,
    mock: bool = True,
) -> ChangeClassification:
    """Classify the semantic type of change detected between two images.

    In mock mode, returns rule-based classification from heatmap
    statistics. In real mode, runs a BIT/ChangeFormer model.

    Args:
        heatmap: Pre-computed HeatmapResult (from tool_generate_difference_heatmap).
        t1: First image (np.ndarray).
        t2: Second image (np.ndarray).
        adapter: Optional pre-loaded ChangeDetectionAdapter.
        mock: If True, returns rule-based classification.

    Returns:
        ChangeClassification with categories, area fraction, and summary.
    """
    if mock or adapter is None:
        return _mock_classification(heatmap)

    result = adapter.predict(
        _to_float_array(t1),
        _to_float_array(t2),
    )
    return ChangeClassification(
        categories=result.get("categories", []),
        changed_area_fraction=float(
            np.mean(result.get("change_mask", np.array(0)))
        ),
        summary="Model-based change classification.",
        confidence=result.get("confidence", 0.5),
    )


# ---------------------------------------------------------------------------
# BiTemporalChangeAgent
# ---------------------------------------------------------------------------

class BiTemporalChangeAgent(BaseAgent):
    """Agent 6 — Compares two satellite images for temporal changes.

    Pre-validates that an image pair is provided. Computes a difference
    heatmap and classifies change types.
    """

    agent_name = "BiTemporalChangeAgent"
    mempalace_wing = "Wing: Temporal_Analysis"

    def __init__(self, config: SatQueryConfig | None = None):
        super().__init__(config)
        self._adapter: ChangeDetectionAdapter | None = None

        if self.config.mode == "real":
            self._adapter = ChangeDetectionAdapter(self.config.change_model_id)
            self._adapter.load(self.config)

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        """Detect and classify changes between two satellite images.

        Expected kwargs:
            image_t1: np.ndarray — first (earlier) image.
            image_t2: np.ndarray — second (later) image.
            query (str): Optional user query for context.

        Returns:
            Dict with heatmap, classification, and change summary.

        Raises:
            PairRequiredError: If image_t2 is not provided.
        """
        img_t1 = kwargs.get("image_t1")
        img_t2 = kwargs.get("image_t2")

        if img_t2 is None:
            raise PairRequiredError()

        is_mock = self.config.mode == "mock"

        # Step 1: Generate difference heatmap
        heatmap = tool_generate_difference_heatmap(
            img_t1, img_t2, mock=is_mock
        )

        # Step 2: Classify change types
        classification = tool_classify_change_type(
            heatmap, img_t1, img_t2,
            adapter=self._adapter,
            mock=is_mock,
        )

        return {
            "heatmap": heatmap.model_dump(),
            "classification": classification.model_dump(),
            "changed_area_fraction": classification.changed_area_fraction,
            "categories": classification.categories,
            "summary": classification.summary,
            "confidence": classification.confidence,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float_array(img: Any) -> np.ndarray:
    """Normalize an image to float64 in [0, 1]."""
    if img is None:
        return np.zeros((64, 64), dtype=np.float64)
    arr = np.asarray(img, dtype=np.float64)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr


def _mock_heatmap(img: Any) -> HeatmapResult:
    """Generate a synthetic heatmap result."""
    if isinstance(img, np.ndarray):
        h, w = img.shape[:2]
    else:
        h, w = 256, 256

    return HeatmapResult(
        heatmap_shape=(h, w),
        min_val=0.0,
        max_val=1.0,
        mean_change=0.23,
    )


def _mock_classification(heatmap: HeatmapResult | None) -> ChangeClassification:
    """Generate a rule-based mock classification from heatmap stats."""
    mean = heatmap.mean_change if heatmap else 0.2

    categories = []
    if mean > 0.3:
        categories.append("new_construction")
    if mean > 0.15:
        categories.append("vegetation_change")
    if mean > 0.1:
        categories.append("surface_modification")
    if not categories:
        categories.append("minimal_change")

    return ChangeClassification(
        categories=categories,
        changed_area_fraction=round(mean * 0.6, 3),
        summary=(
            f"[MOCK] Detected {len(categories)} change categories. "
            f"Mean change intensity: {mean:.2f}."
        ),
        confidence=0.82,
    )
