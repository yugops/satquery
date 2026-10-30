"""Tests for BiTemporalChangeAgent (Agent 6)."""

import numpy as np
import pytest

from core.config import SatQueryConfig
from core.exceptions import PairRequiredError
from core.models import (
    AgentStatus,
    ChangeClassification,
    HeatmapResult,
    OutputType,
)
from vision_swarm.change_agent import (
    BiTemporalChangeAgent,
    tool_classify_change_type,
    tool_generate_difference_heatmap,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_config():
    return SatQueryConfig(mode="mock", db_path=":memory:")


@pytest.fixture
def change_agent(mock_config):
    return BiTemporalChangeAgent(config=mock_config)


@pytest.fixture
def img_t1():
    return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def img_t2():
    return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Tool Function Tests
# ---------------------------------------------------------------------------

class TestDifferenceHeatmap:
    def test_mock_returns_heatmap_result(self):
        result = tool_generate_difference_heatmap(None, None, mock=True)
        assert isinstance(result, HeatmapResult)
        assert result.heatmap_shape is not None
        assert result.min_val <= result.max_val

    def test_real_computation(self, img_t1, img_t2):
        result = tool_generate_difference_heatmap(img_t1, img_t2, mock=False)
        assert isinstance(result, HeatmapResult)
        assert result.heatmap_shape == (64, 64)
        assert 0.0 <= result.min_val <= result.max_val <= 1.0
        assert result.mean_change >= 0.0

    def test_identical_images_no_change(self):
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        result = tool_generate_difference_heatmap(img, img, mock=False)
        assert result.mean_change == 0.0

    def test_max_change(self):
        black = np.zeros((32, 32, 3), dtype=np.uint8)
        white = np.full((32, 32, 3), 255, dtype=np.uint8)
        result = tool_generate_difference_heatmap(black, white, mock=False)
        assert result.mean_change > 0.9


class TestClassifyChangeType:
    def test_mock_returns_classification(self):
        heatmap = HeatmapResult(
            heatmap_shape=(64, 64), mean_change=0.25
        )
        result = tool_classify_change_type(heatmap, None, None, mock=True)
        assert isinstance(result, ChangeClassification)
        assert len(result.categories) > 0
        assert result.confidence > 0.0

    def test_high_change_includes_construction(self):
        heatmap = HeatmapResult(
            heatmap_shape=(64, 64), mean_change=0.5
        )
        result = tool_classify_change_type(heatmap, None, None, mock=True)
        assert "new_construction" in result.categories

    def test_low_change_minimal(self):
        heatmap = HeatmapResult(
            heatmap_shape=(64, 64), mean_change=0.05
        )
        result = tool_classify_change_type(heatmap, None, None, mock=True)
        assert "minimal_change" in result.categories


# ---------------------------------------------------------------------------
# Agent Run Tests
# ---------------------------------------------------------------------------

class TestChangeAgent:
    @pytest.mark.asyncio
    async def test_run_with_pair(self, change_agent, img_t1, img_t2):
        response = await change_agent.run(
            image_t1=img_t1,
            image_t2=img_t2,
            query="What changed?",
        )
        assert response.status == AgentStatus.SUCCESS
        assert response.output_type == OutputType.MOCK
        result = response.result
        assert "heatmap" in result
        assert "classification" in result
        assert "categories" in result
        assert len(result["categories"]) > 0

    @pytest.mark.asyncio
    async def test_pair_required_error(self, change_agent, img_t1):
        """Single image should trigger PairRequiredError → FAILED status."""
        response = await change_agent.run(
            image_t1=img_t1,
            image_t2=None,
        )
        assert response.status == AgentStatus.FAILED
        assert "error" in response.metadata

    @pytest.mark.asyncio
    async def test_result_has_confidence(self, change_agent, img_t1, img_t2):
        response = await change_agent.run(
            image_t1=img_t1,
            image_t2=img_t2,
        )
        assert response.confidence > 0.0
