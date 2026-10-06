"""
SatQuery AI — Tests for GroqAdapter and LLM-based Intent Classification
"""

from unittest.mock import MagicMock, patch
import pytest

from core.adapters.groq_adapter import GroqAdapter
from core.config import SatQueryConfig
from core.exceptions import ModelUnavailableError
from core.models import IntentType, OutputType
from core.orchestrator import LeadOrchestratorAgent


@pytest.fixture
def mock_config():
    return SatQueryConfig(mode="mock", db_path=":memory:")


@pytest.fixture
def groq_config():
    return SatQueryConfig(
        mode="real",
        llm_backend="groq",
        groq_api_key="gsk_mock_test_key",
        db_path=":memory:",
    )


class TestGroqAdapter:
    def test_load_raises_without_api_key(self):
        cfg = SatQueryConfig(mode="real", groq_api_key="")
        adapter = GroqAdapter(api_key="")
        with pytest.raises(ModelUnavailableError) as exc_info:
            adapter.load(cfg)
        assert "Groq API key is missing" in str(exc_info.value)

    def test_load_and_predict_with_mock_client(self, groq_config):
        adapter = GroqAdapter(api_key="gsk_mock_test_key")

        mock_groq_client = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "change_detection"
        mock_completion.choices = [mock_choice]
        mock_groq_client.chat.completions.create.return_value = mock_completion

        with patch("groq.Groq", return_value=mock_groq_client):
            adapter.load(groq_config)
            assert adapter.is_loaded

            intent = adapter.predict("Compare the before and after satellite scenes.")
            assert intent == IntentType.CHANGE

    def test_parse_intent_various_categories(self, groq_config):
        adapter = GroqAdapter(api_key="gsk_mock_test_key")
        adapter._loaded = True
        adapter._client = MagicMock()

        # Test helper parsing logic directly
        assert adapter._parse_intent_response("vqa") == IntentType.VQA
        assert adapter._parse_intent_response("change_detection") == IntentType.CHANGE
        assert adapter._parse_intent_response("grounding") == IntentType.GROUNDING
        assert adapter._parse_intent_response("sar_analysis") == IntentType.SAR
        assert adapter._parse_intent_response("disaster") == IntentType.DISASTER
        assert adapter._parse_intent_response("disaster_correlation") == IntentType.DISASTER_CORRELATION
        assert adapter._parse_intent_response("gis") == IntentType.GIS
        assert adapter._parse_intent_response("unknown random garbage") == IntentType.UNKNOWN

    def test_predict_error_falls_back_to_unknown(self, groq_config):
        adapter = GroqAdapter(api_key="gsk_mock_test_key")
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Groq Rate Limit 429")
        adapter._client = mock_client
        adapter._loaded = True

        intent = adapter.predict("Detect buildings in this scene.")
        assert intent == IntentType.UNKNOWN


class TestOrchestratorGroqIntegration:
    @pytest.mark.asyncio
    async def test_orchestrator_uses_groq_when_configured(self, groq_config):
        with patch("core.adapters.groq_adapter.GroqAdapter.load"), \
             patch("vision_swarm.grounding_agent.GroundingDINOAdapter.load"), \
             patch("vision_swarm.grounding_agent.MobileSAMAdapter.load"), \
             patch("vision_swarm.change_agent.ChangeDetectionAdapter.load"), \
             patch("vision_swarm.vqa_agent.VLMAdapter.load"):
            orchestrator = LeadOrchestratorAgent(config=groq_config)

            # Mock groq adapter prediction
            assert orchestrator._groq_adapter is not None
            orchestrator._groq_adapter._loaded = True
            orchestrator._groq_adapter.predict = MagicMock(return_value=IntentType.GROUNDING)

            response = await orchestrator.run(query="Locate all storage tanks in the region.")
            assert response.intent.intent_type == IntentType.GROUNDING
            assert "VisualGroundingAgent" in response.intent.requires_agents


