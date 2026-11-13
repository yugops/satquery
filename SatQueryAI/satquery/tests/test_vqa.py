"""Tests for SingleSceneVQAAgent (Agent 4)."""

import numpy as np
import pytest

from core.config import SatQueryConfig
from core.models import AgentStatus, CountResult, OutputType, VQAResult
from vision_swarm.vqa_agent import (
    SingleSceneVQAAgent,
    tool_object_count_reasoner,
    tool_vlm_generate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_config():
    return SatQueryConfig(mode="mock", db_path=":memory:")


@pytest.fixture
def vqa_agent(mock_config):
    return SingleSceneVQAAgent(config=mock_config)


@pytest.fixture
def dummy_image():
    return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Tool Function Tests
# ---------------------------------------------------------------------------

class TestVLMGenerate:
    def test_mock_returns_string(self):
        result = tool_vlm_generate(None, "Describe this scene", mock=True)
        assert isinstance(result, str)
        assert len(result) > 10

    def test_mock_building_response(self):
        result = tool_vlm_generate(None, "What buildings are visible?", mock=True)
        assert "building" in result.lower() or "structure" in result.lower()

    def test_mock_water_response(self):
        result = tool_vlm_generate(None, "Is there a river?", mock=True)
        assert "water" in result.lower() or "lake" in result.lower()

    def test_mock_generic_response(self):
        result = tool_vlm_generate(None, "xyz random", mock=True)
        assert isinstance(result, str)


class TestObjectCountReasoner:
    def test_mock_returns_count_result(self):
        result = tool_object_count_reasoner(None, "buildings", mock=True)
        assert isinstance(result, CountResult)
        assert result.count > 0
        assert result.target_class == "buildings"
        assert result.confidence > 0.0

    def test_mock_different_classes(self):
        buildings = tool_object_count_reasoner(None, "buildings", mock=True)
        vehicles = tool_object_count_reasoner(None, "vehicles", mock=True)
        # Should return different mock counts
        assert buildings.count != vehicles.count

    def test_unknown_class_returns_default(self):
        result = tool_object_count_reasoner(None, "alien_spacecraft", mock=True)
        assert isinstance(result, CountResult)
        assert result.count > 0  # default mock count


# ---------------------------------------------------------------------------
# Agent Run Tests
# ---------------------------------------------------------------------------

class TestVQAAgent:
    @pytest.mark.asyncio
    async def test_run_vqa_query(self, vqa_agent, dummy_image):
        response = await vqa_agent.run(
            image=dummy_image,
            query="What do you see in this image?",
        )
        assert response.status == AgentStatus.SUCCESS
        assert response.agent == "SingleSceneVQAAgent"
        assert response.output_type == OutputType.MOCK
        assert isinstance(response.result, VQAResult)
        assert response.result.answer != ""

    @pytest.mark.asyncio
    async def test_run_counting_query(self, vqa_agent, dummy_image):
        response = await vqa_agent.run(
            image=dummy_image,
            query="How many buildings are visible?",
        )
        assert response.status == AgentStatus.SUCCESS
        result = response.result
        assert isinstance(result, VQAResult)
        assert "buildings" in result.answer.lower() or "12" in result.answer

    @pytest.mark.asyncio
    async def test_response_structure(self, vqa_agent, dummy_image):
        """Verify AgentResponse has all required fields (Rule #9)."""
        response = await vqa_agent.run(
            image=dummy_image,
            query="Describe the scene",
        )
        assert response.agent != ""
        assert response.status in (
            AgentStatus.SUCCESS, AgentStatus.FAILED, AgentStatus.SKIPPED
        )
        assert response.execution_time_ms >= 0
        assert response.confidence >= 0.0
        assert "mempalace_wing" in response.metadata

    @pytest.mark.asyncio
    async def test_no_crash_on_none_image(self, vqa_agent):
        """Agent should handle None image gracefully (Rule #6)."""
        response = await vqa_agent.run(
            image=None,
            query="What is visible?",
        )
        # Should succeed in mock mode regardless
        assert response.status == AgentStatus.SUCCESS


# ---------------------------------------------------------------------------
# Hugging Face InferenceClient Tests
# ---------------------------------------------------------------------------

class TestHFInferenceClient:
    def test_build_chat_messages_text_only(self):
        from vision_swarm.vqa_agent import VLMAdapter

        msgs = VLMAdapter.build_chat_messages(None, "Describe this scene.")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Describe this scene."

    def test_build_chat_messages_with_image(self, dummy_image):
        from vision_swarm.vqa_agent import VLMAdapter

        msgs = VLMAdapter.build_chat_messages(dummy_image, "Describe this satellite image.")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        content = msgs[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Describe this satellite image."
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_tool_hf_vlm_chat_completion_with_mock_client(self, dummy_image):
        from unittest.mock import MagicMock
        from vision_swarm.vqa_agent import tool_hf_vlm_chat_completion

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "The satellite image shows an agricultural area with distinct crop fields."
        mock_response.choices = [mock_choice]
        mock_client.chat_completion.return_value = mock_response

        answer = tool_hf_vlm_chat_completion(
            prompt="Describe this satellite image.",
            image=dummy_image,
            client=mock_client,
        )

        assert answer == "The satellite image shows an agricultural area with distinct crop fields."
        mock_client.chat_completion.assert_called_once()
        call_kwargs = mock_client.chat_completion.call_args.kwargs
        assert "messages" in call_kwargs
        assert call_kwargs["max_tokens"] == 256

    def test_vlm_adapter_hf_inference_predict(self, dummy_image):
        from unittest.mock import MagicMock
        from vision_swarm.vqa_agent import VLMAdapter

        adapter = VLMAdapter(
            model_id="Qwen/Qwen2-VL-2B-Instruct",
            hf_token="test_token",
            provider="hf_inference",
        )
        cfg = SatQueryConfig(mode="real", hf_api_token="test_token", vlm_provider="hf_inference")
        adapter.load(cfg)

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Identified 4 cargo vessels in the harbor."
        mock_response.choices = [mock_choice]

        adapter._client.chat_completion = MagicMock(return_value=mock_response)

        ans = adapter.predict(dummy_image, "How many vessels are present?")
        assert ans == "Identified 4 cargo vessels in the harbor."

