"""Tests for SARCloudPenetrationAgent (Agent 7)."""

import numpy as np
import pytest

from core.config import SatQueryConfig
from core.models import AgentStatus, OutputType, SARFalseColor
from vision_swarm.sar_agent import (
    SARCloudPenetrationAgent,
    tool_convert_sar_to_false_color,
    tool_sar_water_threshold,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_config():
    return SatQueryConfig(mode="mock", db_path=":memory:")


@pytest.fixture
def sar_agent(mock_config):
    return SARCloudPenetrationAgent(config=mock_config)


# ---------------------------------------------------------------------------
# False Color Tests
# ---------------------------------------------------------------------------

class TestFalseColor:
    def test_mock_returns_sar_false_color(self):
        result = tool_convert_sar_to_false_color("vv.tif", "vh.tif", mock=True)
        assert isinstance(result, SARFalseColor)
        assert result.shape is not None
        assert len(result.shape) == 3
        assert result.shape[2] == 3  # RGB channels
        assert result.calibration_applied is True

    def test_mock_db_ranges(self):
        result = tool_convert_sar_to_false_color("vv.tif", "vh.tif", mock=True)
        assert result.vv_db_range is not None
        assert result.vh_db_range is not None
        # VV should typically be higher than VH
        assert result.vv_db_range[0] > result.vh_db_range[0]


# ---------------------------------------------------------------------------
# Water Threshold Tests
# ---------------------------------------------------------------------------

class TestWaterThreshold:
    def test_mock_returns_mask(self):
        vv = np.random.randn(64, 64).astype(np.float64)
        mask = tool_sar_water_threshold(vv, threshold_db=-18.0, mock=True)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 1})

    def test_mock_mask_has_water(self):
        vv = np.random.randn(64, 64).astype(np.float64)
        mask = tool_sar_water_threshold(vv, mock=True)
        # Mock places water in lower-right quadrant
        assert np.sum(mask) > 0

    def test_real_threshold_all_below(self):
        """Array all below threshold → all water."""
        vv = np.full((32, 32), -25.0)  # Well below -18 dB
        mask = tool_sar_water_threshold(vv, threshold_db=-18.0, mock=False)
        assert np.all(mask == 1)

    def test_real_threshold_all_above(self):
        """Array all above threshold → no water."""
        vv = np.full((32, 32), -5.0)  # Well above -18 dB
        mask = tool_sar_water_threshold(vv, threshold_db=-18.0, mock=False)
        assert np.all(mask == 0)

    def test_db_calibration_correctness(self):
        """Verify dB conversion: for values in 'likely dB' range,
        no double-conversion should occur."""
        # Values already in dB range (negative, < 100)
        vv_db = np.array([[-20.0, -15.0], [-10.0, -5.0]])
        mask = tool_sar_water_threshold(vv_db, threshold_db=-18.0, mock=False)
        # Only -20.0 should be below threshold
        expected = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(mask, expected)


# ---------------------------------------------------------------------------
# Agent Run Tests
# ---------------------------------------------------------------------------

class TestSARAgent:
    @pytest.mark.asyncio
    async def test_run_mock_with_array(self, sar_agent):
        vv = np.random.randn(64, 64).astype(np.float64) * 10 - 15
        response = await sar_agent.run(
            vv_tif="mock_vv.tif",
            vh_tif="mock_vh.tif",
            vv_array=vv,
        )
        assert response.status == AgentStatus.SUCCESS
        assert response.output_type == OutputType.MOCK
        result = response.result
        assert "false_color" in result
        assert "water_detection" in result
        assert "water_fraction" in result

    @pytest.mark.asyncio
    async def test_run_mock_without_array(self, sar_agent):
        """Should still succeed with mock water detection."""
        response = await sar_agent.run(
            vv_tif="mock_vv.tif",
            vh_tif="mock_vh.tif",
        )
        assert response.status == AgentStatus.SUCCESS
        result = response.result
        assert result["water_fraction"] >= 0.0

    @pytest.mark.asyncio
    async def test_confidence_present(self, sar_agent):
        response = await sar_agent.run(
            vv_tif="mock_vv.tif",
            vh_tif="mock_vh.tif",
        )
        assert response.confidence > 0.0
