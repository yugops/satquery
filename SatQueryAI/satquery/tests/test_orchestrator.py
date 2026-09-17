"""Tests for LeadOrchestratorAgent (Agent 1)."""

import pytest
import pytest_asyncio
import numpy as np

from core.config import SatQueryConfig
from core.models import (
    AgentStatus,
    FinalResponse,
    IntentPlan,
    IntentType,
    OutputType,
)
from core.orchestrator import (
    LeadOrchestratorAgent,
    tool_build_dag_plan,
    tool_classify_intent,
    tool_synthesize_evidence,
)


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
def dummy_image():
    return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Intent Classification Tests
# ---------------------------------------------------------------------------

class TestClassifyIntent:
    def test_vqa_intent(self):
        plan = tool_classify_intent("How many buildings are visible?")
        assert plan.intent_type == IntentType.VQA
        assert "SingleSceneVQAAgent" in plan.requires_agents
        assert plan.confidence > 0.0

    def test_change_intent_with_pair(self):
        plan = tool_classify_intent(
            "What changed between these images?", has_pair=True
        )
        assert plan.intent_type == IntentType.CHANGE
        assert "BiTemporalChangeAgent" in plan.requires_agents
        assert plan.has_pair is True

    def test_grounding_intent(self):
        plan = tool_classify_intent("Locate all buildings near the river")
        assert plan.intent_type == IntentType.GROUNDING
        assert "VisualGroundingAgent" in plan.requires_agents

    def test_sar_intent(self):
        plan = tool_classify_intent("Analyze SAR backscatter", is_sar=True)
        assert plan.intent_type == IntentType.SAR
        assert "SARCloudPenetrationAgent" in plan.requires_agents

    def test_disaster_intent(self):
        plan = tool_classify_intent("Assess flood damage in this area")
        assert plan.intent_type == IntentType.DISASTER

    def test_unknown_defaults_to_vqa(self):
        plan = tool_classify_intent("asdfghjkl random text")
        assert plan.intent_type == IntentType.VQA
        assert plan.confidence < 0.5

    def test_sar_context_boost(self):
        plan = tool_classify_intent("Analyze this image", is_sar=True)
        assert plan.intent_type == IntentType.SAR

    def test_raw_query_preserved(self):
        q = "How many buildings?"
        plan = tool_classify_intent(q)
        assert plan.raw_query == q


# ---------------------------------------------------------------------------
# DAG Plan Tests
# ---------------------------------------------------------------------------

class TestBuildDAGPlan:
    def test_validator_always_first(self):
        intent = IntentPlan(
            intent_type=IntentType.VQA,
            requires_agents=["SingleSceneVQAAgent"],
        )
        dag = tool_build_dag_plan(intent)
        assert dag[0].agent_name == "GeoValidatorAgent"
        assert dag[0].depends_on == []

    def test_specialist_depends_on_validator(self):
        intent = IntentPlan(
            intent_type=IntentType.VQA,
            requires_agents=["SingleSceneVQAAgent"],
        )
        dag = tool_build_dag_plan(intent)
        specialist = dag[1]
        assert specialist.agent_name == "SingleSceneVQAAgent"
        assert "GeoValidatorAgent" in specialist.depends_on

    def test_dag_length_matches_agents(self):
        intent = IntentPlan(
            intent_type=IntentType.CHANGE,
            requires_agents=["BiTemporalChangeAgent"],
        )
        dag = tool_build_dag_plan(intent)
        # Validator + 1 specialist
        assert len(dag) == 2


# ---------------------------------------------------------------------------
# Evidence Synthesis Tests
# ---------------------------------------------------------------------------

class TestSynthesizeEvidence:
    def test_synthesize_success(self):
        from core.models import AgentResponse

        results = [
            AgentResponse(
                agent="TestAgent",
                status=AgentStatus.SUCCESS,
                result={"answer": "There are 5 buildings."},
                confidence=0.9,
            )
        ]
        answer = tool_synthesize_evidence(results, "How many buildings?")
        assert "5 buildings" in answer

    def test_synthesize_all_failed(self):
        from core.models import AgentResponse

        results = [
            AgentResponse(
                agent="TestAgent",
                status=AgentStatus.FAILED,
                result=None,
                confidence=0.0,
            )
        ]
        answer = tool_synthesize_evidence(results)
        assert "unable" in answer.lower() or "failed" in answer.lower()

    def test_output_source_label(self):
        from core.models import AgentResponse

        results = [
            AgentResponse(
                agent="TestAgent",
                status=AgentStatus.SUCCESS,
                result={"answer": "test"},
                confidence=0.8,
                output_type=OutputType.MOCK,
            )
        ]
        answer = tool_synthesize_evidence(results)
        assert "MOCK" in answer


# ---------------------------------------------------------------------------
# End-to-End Orchestrator Tests
# ---------------------------------------------------------------------------

class TestOrchestratorRun:
    @pytest.mark.asyncio
    async def test_run_vqa_mock(self, orchestrator, dummy_image):
        response = await orchestrator.run(
            query="What do you see in this image?",
            image_array=dummy_image,
        )
        assert isinstance(response, FinalResponse)
        assert response.query == "What do you see in this image?"
        assert response.intent.intent_type == IntentType.VQA
        assert len(response.agent_responses) > 0
        assert response.synthesized_answer != ""
        assert response.total_execution_ms > 0

    @pytest.mark.asyncio
    async def test_run_change_mock(self, orchestrator, dummy_image):
        img_t2 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        response = await orchestrator.run(
            query="What changed between these two images?",
            image_array=dummy_image,
            image_array_t2=img_t2,
        )
        assert response.intent.intent_type == IntentType.CHANGE
        assert response.intent.has_pair is True

    @pytest.mark.asyncio
    async def test_run_returns_session_id(self, orchestrator, dummy_image):
        response = await orchestrator.run(
            query="Describe this scene",
            image_array=dummy_image,
        )
        assert response.session_id != ""

    @pytest.mark.asyncio
    async def test_agent_responses_have_required_fields(
        self, orchestrator, dummy_image
    ):
        response = await orchestrator.run(
            query="How many buildings?",
            image_array=dummy_image,
        )
        for ar in response.agent_responses:
            assert ar.agent != ""
            assert ar.status in (
                AgentStatus.SUCCESS,
                AgentStatus.FAILED,
                AgentStatus.SKIPPED,
            )
            assert ar.execution_time_ms >= 0
