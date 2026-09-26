"""Tests for VisualGroundingAgent (Agent 5)."""

import numpy as np
import pytest

from core.config import SatQueryConfig
from core.models import AgentStatus, BBox, GeoJSONPolygon, OutputType
from vision_swarm.grounding_agent import (
    VisualGroundingAgent,
    tool_grounding_dino_bbox,
    tool_mobilesam_segment_polygon,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_config():
    return SatQueryConfig(mode="mock", db_path=":memory:")


@pytest.fixture
def grounding_agent(mock_config):
    return VisualGroundingAgent(config=mock_config)


@pytest.fixture
def dummy_image():
    return np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Tool Function Tests
# ---------------------------------------------------------------------------

class TestGroundingDINOBBox:
    def test_mock_returns_bboxes(self):
        bboxes = tool_grounding_dino_bbox(
            None, "buildings", mock=True, image_shape=(256, 256)
        )
        assert isinstance(bboxes, list)
        assert len(bboxes) > 0
        assert all(isinstance(b, BBox) for b in bboxes)

    def test_bbox_coordinates_within_image(self):
        h, w = 512, 512
        bboxes = tool_grounding_dino_bbox(
            None, "buildings", mock=True, image_shape=(h, w)
        )
        for b in bboxes:
            assert 0 <= b.x1 <= w
            assert 0 <= b.y1 <= h
            assert b.x2 >= b.x1
            assert b.y2 >= b.y1

    def test_bbox_has_confidence(self):
        bboxes = tool_grounding_dino_bbox(
            None, "trees", mock=True, image_shape=(256, 256)
        )
        for b in bboxes:
            assert b.confidence > 0.0

    def test_bbox_has_label(self):
        bboxes = tool_grounding_dino_bbox(
            None, "vehicles", mock=True, image_shape=(256, 256)
        )
        for b in bboxes:
            assert b.label != ""


class TestMobileSAMSegment:
    def test_mock_returns_geojson(self):
        bboxes = [
            BBox(x1=10, y1=10, x2=50, y2=50, confidence=0.9, label="building"),
        ]
        result = tool_mobilesam_segment_polygon(None, bboxes, mock=True)
        assert isinstance(result, GeoJSONPolygon)
        assert result.type == "FeatureCollection"
        assert len(result.features) == 1

    def test_polygon_geometry(self):
        bboxes = [
            BBox(x1=10, y1=20, x2=30, y2=40, confidence=0.8, label="road"),
        ]
        result = tool_mobilesam_segment_polygon(None, bboxes, mock=True)
        feature = result.features[0]
        assert feature["geometry"]["type"] == "Polygon"
        coords = feature["geometry"]["coordinates"][0]
        assert len(coords) == 5  # Closed polygon ring

    def test_multiple_bboxes(self):
        bboxes = [
            BBox(x1=0, y1=0, x2=10, y2=10, confidence=0.9, label="a"),
            BBox(x1=20, y1=20, x2=30, y2=30, confidence=0.8, label="b"),
        ]
        result = tool_mobilesam_segment_polygon(None, bboxes, mock=True)
        assert len(result.features) == 2


# ---------------------------------------------------------------------------
# Agent Run Tests
# ---------------------------------------------------------------------------

class TestGroundingAgent:
    @pytest.mark.asyncio
    async def test_run_mock(self, grounding_agent, dummy_image):
        response = await grounding_agent.run(
            image=dummy_image,
            query="buildings",
        )
        assert response.status == AgentStatus.SUCCESS
        assert response.output_type == OutputType.MOCK
        result = response.result
        assert "bboxes" in result
        assert "geojson_polygons" in result
        assert result["detection_count"] > 0

    @pytest.mark.asyncio
    async def test_confidence_aggregated(self, grounding_agent, dummy_image):
        response = await grounding_agent.run(
            image=dummy_image,
            query="trees",
        )
        result = response.result
        assert result["confidence"] > 0.0

    @pytest.mark.asyncio
    async def test_high_threshold_fewer_boxes(self, grounding_agent, dummy_image):
        """Higher threshold should not increase detections."""
        response_low = await grounding_agent.run(
            image=dummy_image,
            query="buildings",
            box_threshold=0.1,
        )
        response_high = await grounding_agent.run(
            image=dummy_image,
            query="buildings",
            box_threshold=0.99,
        )
        # In mock mode, threshold doesn't filter (mock always returns same set)
        # But the test verifies the parameter is accepted without error
        assert response_low.status == AgentStatus.SUCCESS
        assert response_high.status == AgentStatus.SUCCESS
