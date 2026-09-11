"""
SatQuery AI — Agent 4: SingleSceneVQAAgent

Wing: Visual_Reasoning

Answers open-ended natural language questions about a single satellite
image by routing through a configurable VLM (Qwen2-VL by default).
Supports object counting with structured CountResult output.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from core.config import SatQueryConfig
from core.exceptions import ModelUnavailableError
from core.interfaces import BaseAgent, ModelAdapter
from core.models import CountResult, OutputType, VQAResult


# ---------------------------------------------------------------------------
# VLM Model Adapter
# ---------------------------------------------------------------------------

class VLMAdapter(ModelAdapter):
    """Wraps Qwen2-VL (or any HuggingFace VLM) behind a uniform interface.

    Supports both Hugging Face InferenceClient API (hosted inference)
    and local transformers pipeline. In mock mode, returns plausible
    template responses without network calls or loading model weights.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        hf_token: str = "",
        provider: str = "hf_inference",
    ):
        super().__init__(model_id)
        self.hf_token = hf_token
        self.provider = provider
        self._client: Any = None
        self._processor: Any = None

    def load(self, config: SatQueryConfig) -> None:
        """Initialize InferenceClient or local transformers model."""
        self.hf_token = self.hf_token or config.hf_api_token
        self.provider = self.provider or config.vlm_provider

        if self.provider == "hf_inference":
            try:
                from huggingface_hub import InferenceClient

                self._client = InferenceClient(
                    model=self.model_id,
                    token=self.hf_token or None,
                )
                self._loaded = True
            except Exception as exc:
                raise ModelUnavailableError(
                    self.model_id,
                    message=f"Failed to initialize Hugging Face InferenceClient: {exc}",
                )
        else:
            try:
                from transformers import AutoModelForVision2Seq, AutoProcessor

                self._processor = AutoProcessor.from_pretrained(self.model_id)
                self._model = AutoModelForVision2Seq.from_pretrained(
                    self.model_id,
                    device_map="auto",
                    torch_dtype="auto",
                )
                self._loaded = True
            except Exception as exc:
                raise ModelUnavailableError(
                    self.model_id,
                    message=f"Failed to load VLM: {exc}",
                )

    def predict(
        self,
        image: Any,
        prompt: str,
        max_tokens: int = 256,
    ) -> str:
        """Run VLM inference on a single image + prompt.

        Args:
            image: PIL.Image, np.ndarray, file path, or None.
            prompt: Natural language question.
            max_tokens: Maximum output tokens.

        Returns:
            Generated text answer.
        """
        self.ensure_loaded()

        if self.provider == "hf_inference" and self._client is not None:
            return self._predict_hf_inference(image, prompt, max_tokens)
        else:
            return self._predict_transformers(image, prompt, max_tokens)

    def _predict_hf_inference(
        self,
        image: Any,
        prompt: str,
        max_tokens: int = 256,
    ) -> str:
        messages = self.build_chat_messages(image, prompt)
        try:
            response = self._client.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
            )
            if response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    return choice.message.content or ""
            return ""
        except Exception as exc:
            raise ModelUnavailableError(
                self.model_id,
                message=f"HuggingFace InferenceClient request failed: {exc}",
            )

    def _predict_transformers(
        self,
        image: Any,
        prompt: str,
        max_tokens: int = 256,
    ) -> str:
        from PIL import Image as PILImage
        import torch

        # Normalize image to PIL
        if isinstance(image, (str, Path)):
            pil_img = PILImage.open(str(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = PILImage.fromarray(image.astype(np.uint8)).convert("RGB")
        elif hasattr(image, "convert"):
            pil_img = image.convert("RGB")
        else:
            pil_img = image

        inputs = self._processor(
            text=prompt,
            images=pil_img,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=max_tokens)

        return self._processor.decode(output_ids[0], skip_special_tokens=True)

    @staticmethod
    def build_chat_messages(image: Any, prompt: str) -> list[dict[str, Any]]:
        """Build OpenAI-compatible chat messages structure for InferenceClient."""
        if image is None:
            return [{"role": "user", "content": prompt}]

        data_uri = VLMAdapter.image_to_data_uri(image)
        if not data_uri:
            return [{"role": "user", "content": prompt}]

        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ]

    @staticmethod
    def image_to_data_uri(image: Any) -> str | None:
        """Convert image (file path, ndarray, or PIL) to base64 Data URI."""
        import base64
        import io
        from PIL import Image as PILImage

        try:
            if isinstance(image, (str, Path)):
                pil_img = PILImage.open(str(image)).convert("RGB")
            elif isinstance(image, np.ndarray):
                if image.size == 0:
                    return None
                pil_img = PILImage.fromarray(image.astype(np.uint8)).convert("RGB")
            elif hasattr(image, "convert"):
                pil_img = image.convert("RGB")
            else:
                return None

            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG")
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64_str}"
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Tool Functions
# ---------------------------------------------------------------------------

def tool_hf_vlm_chat_completion(
    prompt: str,
    image: Any = None,
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
    hf_token: str = "",
    max_tokens: int = 256,
    client: Any | None = None,
) -> str:
    """Run VLM inference using Hugging Face InferenceClient chat_completion.

    Example:
        >>> from vision_swarm.vqa_agent import tool_hf_vlm_chat_completion
        >>> answer = tool_hf_vlm_chat_completion("Describe this satellite image.")
        >>> print(answer)

    Args:
        prompt: Natural language query (e.g. 'Describe this satellite image.').
        image: Optional image as file path, numpy ndarray, or PIL Image.
        model_id: HuggingFace model ID (default: 'Qwen/Qwen2-VL-2B-Instruct').
        hf_token: HuggingFace API token (falls back to SATQUERY_HF_API_TOKEN).
        max_tokens: Maximum output tokens.
        client: Optional pre-configured InferenceClient instance.

    Returns:
        Generated natural language answer string.
    """
    from huggingface_hub import InferenceClient

    if client is None:
        client = InferenceClient(
            model=model_id,
            token=hf_token or None,
        )

    messages = VLMAdapter.build_chat_messages(image, prompt)
    response = client.chat_completion(
        messages=messages,
        max_tokens=max_tokens,
    )
    if response.choices and len(response.choices) > 0:
        choice = response.choices[0]
        if hasattr(choice, "message") and hasattr(choice.message, "content"):
            return choice.message.content or ""
    return ""


def tool_vlm_generate(
    image: Any,
    prompt: str,
    max_tokens: int = 256,
    *,
    adapter: VLMAdapter | None = None,
    mock: bool = True,
) -> str:
    """Generate a natural language answer to a visual question.

    In mock mode, returns a plausible template response based on
    prompt keywords. In real mode, delegates to the VLMAdapter.

    Args:
        image: Image as ndarray, file path, or PIL Image.
        prompt: Natural language question about the image.
        max_tokens: Maximum tokens for the response.
        adapter: Optional pre-loaded VLMAdapter (real mode).
        mock: If True, returns synthetic response.

    Returns:
        Generated text answer.
    """
    if mock or adapter is None:
        return _mock_vlm_response(prompt)

    return adapter.predict(image, prompt, max_tokens)


def tool_object_count_reasoner(
    image: Any,
    target_class: str,
    *,
    adapter: VLMAdapter | None = None,
    mock: bool = True,
) -> CountResult:
    """Count instances of a specific object class in a satellite image.

    Uses a counting-specific prompt template and extracts the numeric
    count from the VLM response via regex.

    Args:
        image: Image as ndarray, file path, or PIL Image.
        target_class: Object class to count (e.g., "buildings", "vehicles").
        adapter: Optional pre-loaded VLMAdapter (real mode).
        mock: If True, returns synthetic count.

    Returns:
        CountResult with count, reasoning, and confidence.
    """
    if mock or adapter is None:
        return _mock_count_result(target_class)

    count_prompt = (
        f"Carefully examine this satellite image. "
        f"Count the number of {target_class} visible in the image. "
        f"Provide the count as a number and explain your reasoning."
    )
    raw_answer = adapter.predict(image, count_prompt, max_tokens=200)

    # Extract numeric count from response
    numbers = re.findall(r"\b(\d+)\b", raw_answer)
    count = int(numbers[0]) if numbers else 0

    return CountResult(
        count=count,
        target_class=target_class,
        reasoning=raw_answer,
        confidence=0.7,
    )


# ---------------------------------------------------------------------------
# SingleSceneVQAAgent
# ---------------------------------------------------------------------------

class SingleSceneVQAAgent(BaseAgent):
    """Agent 4 — Answers questions about a single satellite image.

    Invoked whenever the DAG plan classifies the query intent as VQA.
    Supports both free-form questions and structured counting queries.
    """

    agent_name = "SingleSceneVQAAgent"
    mempalace_wing = "Wing: Visual_Reasoning"

    def __init__(self, config: SatQueryConfig | None = None):
        super().__init__(config)
        self._adapter: VLMAdapter | None = None

        if self.config.mode == "real":
            self._adapter = VLMAdapter(
                model_id=self.config.vlm_model_id,
                hf_token=self.config.hf_api_token,
                provider=self.config.vlm_provider,
            )
            self._adapter.load(self.config)

    async def _execute(self, **kwargs: Any) -> VQAResult:
        """Answer a visual question about a satellite image.

        Expected kwargs:
            image: np.ndarray, file path, or PIL Image.
            query (str): Natural language question.
            max_tokens (int): Optional max tokens.

        Returns:
            VQAResult with answer, reasoning, and confidence.
        """
        image = kwargs.get("image")
        query = kwargs.get("query", "")
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        is_mock = self.config.mode == "mock"

        # Detect counting queries
        counting_keywords = ["how many", "count", "number of"]
        is_counting = any(kw in query.lower() for kw in counting_keywords)

        if is_counting:
            # Extract target class from query
            target = _extract_target_class(query)
            count_result = tool_object_count_reasoner(
                image, target, adapter=self._adapter, mock=is_mock
            )
            return VQAResult(
                answer=f"There are approximately {count_result.count} {target}.",
                reasoning=count_result.reasoning,
                confidence=count_result.confidence,
            )
        else:
            answer = tool_vlm_generate(
                image, query, max_tokens,
                adapter=self._adapter, mock=is_mock,
            )
            return VQAResult(
                answer=answer,
                reasoning="Generated by VLM scene analysis.",
                confidence=0.85 if is_mock else 0.75,
            )


# ---------------------------------------------------------------------------
# Mock / Helpers
# ---------------------------------------------------------------------------

def _mock_vlm_response(prompt: str) -> str:
    """Generate a plausible mock response based on prompt keywords."""
    lower = prompt.lower()

    if any(w in lower for w in ["building", "structure", "house"]):
        return (
            "The satellite image shows several built-up structures consistent "
            "with residential or commercial buildings. The area appears to be "
            "a suburban settlement with moderate density."
        )
    if any(w in lower for w in ["road", "highway", "path"]):
        return (
            "A paved road is visible running through the scene, approximately "
            "oriented north-south. It appears to be a two-lane road with "
            "light vehicle traffic."
        )
    if any(w in lower for w in ["water", "river", "lake", "ocean"]):
        return (
            "A water body is visible in the image. Based on its shape and "
            "surrounding vegetation, it appears to be a small inland lake "
            "or reservoir."
        )
    if any(w in lower for w in ["vegetation", "forest", "tree", "green"]):
        return (
            "The scene contains dense vegetation cover, likely a mix of "
            "deciduous and evergreen species. The NDVI values would suggest "
            "healthy, active vegetation."
        )
    if any(w in lower for w in ["describe", "scene", "what do you see"]):
        return (
            "This satellite image shows a mixed land-use area with patches "
            "of vegetation, some built-up regions, and open terrain. The "
            "overall scene suggests a peri-urban or rural landscape."
        )
    return (
        "Based on the satellite imagery analysis, the scene contains "
        "identifiable features including land cover variations and "
        "possible anthropogenic structures."
    )


def _mock_count_result(target_class: str) -> CountResult:
    """Generate a synthetic count result."""
    # Deterministic-ish mock counts per class
    mock_counts = {
        "buildings": 12,
        "building": 12,
        "vehicles": 5,
        "vehicle": 5,
        "cars": 3,
        "trees": 25,
        "roads": 2,
    }
    count = mock_counts.get(target_class.lower(), 7)

    return CountResult(
        count=count,
        target_class=target_class,
        reasoning=(
            f"[MOCK] Estimated {count} {target_class} based on synthetic "
            f"analysis of the satellite scene."
        ),
        confidence=0.85,
    )


def _extract_target_class(query: str) -> str:
    """Extract the target object class from a counting query."""
    # Simple heuristic: look for common patterns
    patterns = [
        r"how many (\w+)",
        r"count (?:the )?(\w+)",
        r"number of (\w+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            return match.group(1)
    return "objects"
