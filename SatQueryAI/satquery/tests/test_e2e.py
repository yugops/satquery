"""
SatQuery AI — End-to-End Integration Test

Tests the full pipeline:
    query → orchestrator → validator → specialist → audit → FinalResponse

Covers three scenarios:
    1. VQA query (single image)
    2. Change detection query (image pair)
    3. SAR fallback query (cloud contamination → SAR)
"""

import numpy as np
import pytest

from core.config import SatQueryConfig
from core.models import (
    AgentStatus,
    FinalResponse,
    IntentType,
    OutputType,
)
from core.orchestrator import LeadOrchestratorAgent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_config():
    return SatQueryConfig(mode="mock", db_path=":memory:")


@pytest.fixture
def orchestrator(mock_config):
    return LeadOrchestratorAgent(config=mock_config)


@pytest.fixture
def sample_image():
    """A non-cloudy satellite image (dark enough to pass cloud check)."""
    return np.random.randint(0, 150, (128, 128, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_pair():
    """Two temporally different images."""
    t1 = np.random.randint(0, 150, (128, 128, 3), dtype=np.uint8)
    t2 = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
    return t1, t2


# ---------------------------------------------------------------------------
# Scenario 1: VQA Query
# ---------------------------------------------------------------------------

class TestE2EVQA:
    """User asks: 'How many buildings are visible in this satellite image?'
    Expected flow: Orchestrator → Validator → VQA → Audit → FinalResponse
    """

    @pytest.mark.asyncio
    async def test_vqa_pipeline(self, orchestrator, sample_image):
        response = await orchestrator.run(
            query="How many buildings are visible in this satellite image?",
            image_array=sample_image,
        )

        # ---- Verify FinalResponse structure ----
        assert isinstance(response, FinalResponse)
        assert response.session_id != ""
        assert response.query == "How many buildings are visible in this satellite image?"
        assert response.total_execution_ms > 0

        # ---- Verify intent classification ----
        assert response.intent.intent_type == IntentType.VQA
        assert "SingleSceneVQAAgent" in response.intent.requires_agents

        # ---- Verify agent responses ----
        agent_names = [r.agent for r in response.agent_responses]
        assert "GeoValidatorAgent" in agent_names
        assert "SingleSceneVQAAgent" in agent_names

        # All agents should have succeeded
        for r in response.agent_responses:
            if r.status != AgentStatus.SKIPPED:
                assert r.status == AgentStatus.SUCCESS
                assert r.execution_time_ms >= 0

        # ---- Verify synthesized answer ----
        assert response.synthesized_answer != ""
        assert "MOCK" in response.synthesized_answer  # Output source label

    @pytest.mark.asyncio
    async def test_vqa_output_type_is_mock(self, orchestrator, sample_image):
        response = await orchestrator.run(
            query="Describe this scene",
            image_array=sample_image,
        )
        vqa_responses = [
            r for r in response.agent_responses
            if r.agent == "SingleSceneVQAAgent"
        ]
        assert len(vqa_responses) == 1
        assert vqa_responses[0].output_type == OutputType.MOCK


# ---------------------------------------------------------------------------
# Scenario 2: Change Detection Query
# ---------------------------------------------------------------------------

class TestE2EChangeDetection:
    """User asks: 'What changed between these two satellite images?'
    Expected flow: Orchestrator → Validator → BiTemporalChange → Audit → Final
    """

    @pytest.mark.asyncio
    async def test_change_pipeline(self, orchestrator, sample_image_pair):
        t1, t2 = sample_image_pair

        response = await orchestrator.run(
            query="What changed between these two satellite images?",
            image_array=t1,
            image_array_t2=t2,
        )

        assert isinstance(response, FinalResponse)
        assert response.intent.intent_type == IntentType.CHANGE
        assert response.intent.has_pair is True

        agent_names = [r.agent for r in response.agent_responses]
        assert "GeoValidatorAgent" in agent_names
        assert "BiTemporalChangeAgent" in agent_names

        # Change agent should succeed
        change_response = [
            r for r in response.agent_responses
            if r.agent == "BiTemporalChangeAgent"
        ]
        assert len(change_response) == 1
        assert change_response[0].status == AgentStatus.SUCCESS

        result = change_response[0].result
        assert "categories" in result
        assert len(result["categories"]) > 0


# ---------------------------------------------------------------------------
# Scenario 3: SAR Fallback Query
# ---------------------------------------------------------------------------

class TestE2ESARFallback:
    """User provides SAR data paths.
    Expected flow: Orchestrator → Validator → SAR Agent → Audit → Final
    """

    @pytest.mark.asyncio
    async def test_sar_pipeline(self, orchestrator):
        response = await orchestrator.run(
            query="Analyze this SAR imagery for flood detection",
            vv_tif="sentinel1_vv.tif",
            vh_tif="sentinel1_vh.tif",
        )

        assert isinstance(response, FinalResponse)
        assert response.intent.intent_type == IntentType.SAR
        assert response.intent.is_sar is True

        agent_names = [r.agent for r in response.agent_responses]
        assert "SARCloudPenetrationAgent" in agent_names

        sar_response = [
            r for r in response.agent_responses
            if r.agent == "SARCloudPenetrationAgent"
        ]
        assert len(sar_response) == 1
        assert sar_response[0].status == AgentStatus.SUCCESS


# ---------------------------------------------------------------------------
# Cross-Cutting Concerns
# ---------------------------------------------------------------------------

class TestE2ECrossCutting:
    """Verify rules that apply to all scenarios."""

    @pytest.mark.asyncio
    async def test_all_responses_have_required_fields(
        self, orchestrator, sample_image
    ):
        """Rule #9: Every agent result must contain agent, status, result,
        confidence, execution_time_ms, metadata."""
        response = await orchestrator.run(
            query="What type of land cover is this?",
            image_array=sample_image,
        )
        for ar in response.agent_responses:
            assert ar.agent != ""
            assert ar.status in (
                AgentStatus.SUCCESS,
                AgentStatus.FAILED,
                AgentStatus.SKIPPED,
            )
            assert isinstance(ar.execution_time_ms, float)
            assert isinstance(ar.confidence, float)
            assert isinstance(ar.metadata, dict)

    @pytest.mark.asyncio
    async def test_output_type_distinguishes_mock(
        self, orchestrator, sample_image
    ):
        """Rule #12: Clearly distinguish mock output from real model output."""
        response = await orchestrator.run(
            query="Describe the scene",
            image_array=sample_image,
        )
        for ar in response.agent_responses:
            if ar.status == AgentStatus.SUCCESS:
                assert ar.output_type in (
                    OutputType.MOCK,
                    OutputType.MODEL,
                    OutputType.RULE_BASED,
                )

    @pytest.mark.asyncio
    async def test_no_crash_without_image(self, orchestrator):
        """Rule #6: Never crash because an optional resource is unavailable."""
        response = await orchestrator.run(
            query="What do you see?",
        )
        # Should not raise — may have failed agents, but pipeline completes
        assert isinstance(response, FinalResponse)

    @pytest.mark.asyncio
    async def test_grounding_query(self, orchestrator, sample_image):
        """Verify grounding queries route correctly."""
        response = await orchestrator.run(
            query="Locate all buildings near the river",
            image_array=sample_image,
        )
        assert response.intent.intent_type == IntentType.GROUNDING
        agent_names = [r.agent for r in response.agent_responses]
        assert "VisualGroundingAgent" in agent_names
