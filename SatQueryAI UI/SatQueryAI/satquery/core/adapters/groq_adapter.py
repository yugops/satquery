"""
SatQuery AI — Groq LLM Adapter

Implements ModelAdapter for fast hosted LLM inference via Groq API.
Used by LeadOrchestratorAgent for intelligent query intent classification.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from core.config import SatQueryConfig
from core.exceptions import ModelUnavailableError
from core.interfaces import ModelAdapter
from core.models import IntentType

logger = logging.getLogger(__name__)


class GroqAdapter(ModelAdapter):
    """Adapter for Groq Cloud LLM inference.

    Supports mock/real modes. In real mode, communicates with Groq API
    using the official groq SDK or OpenAI-compatible client.
    """

    def __init__(
        self,
        model_id: str = "llama-3.3-70b-versatile",
        api_key: str = "",
    ):
        super().__init__(model_id)
        self.api_key = api_key
        self._client: Any = None

    def load(self, config: SatQueryConfig) -> None:
        """Initialize Groq client using configured API key."""
        self.api_key = self.api_key or config.groq_api_key
        self.model_id = self.model_id or config.groq_model_id

        if not self.api_key:
            raise ModelUnavailableError(
                self.model_id,
                message="Groq API key is missing. Set SATQUERY_GROQ_API_KEY.",
            )

        try:
            from groq import Groq

            self._client = Groq(api_key=self.api_key)
            self._loaded = True
        except Exception as exc:
            raise ModelUnavailableError(
                self.model_id,
                message=f"Failed to initialize Groq client: {exc}",
            )

    def predict(
        self,
        query: str,
        system_prompt: str | None = None,
        max_tokens: int = 50,
    ) -> IntentType:
        """Classify user query into an IntentType enum using Groq LLM.

        Args:
            query: Natural language query.
            system_prompt: Optional custom system prompt.
            max_tokens: Maximum tokens in response.

        Returns:
            Validated IntentType enum member.
        """
        self.ensure_loaded()

        sys_msg = system_prompt or (
            "You are an intent classification system for a satellite imagery intelligence platform.\n"
            "Classify the user's query into EXACTLY ONE of the following category keys:\n"
            "- vqa\n"
            "- change_detection\n"
            "- grounding\n"
            "- sar_analysis\n"
            "- gis\n"
            "- disaster\n"
            "- disaster_correlation\n"
            "- unknown\n\n"
            "Output ONLY the category name. Do not output anything else."
        )

        try:
            chat_completion = self._client.chat.completions.create(
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": query},
                ],
                model=self.model_id,
                max_tokens=max_tokens,
                temperature=0.0,
            )

            raw_answer = (
                chat_completion.choices[0].message.content.strip().lower()
                if chat_completion.choices
                else ""
            )

            return self._parse_intent_response(raw_answer)
        except Exception as exc:
            logger.warning("Groq intent classification failed: %s. Falling back to unknown.", exc)
            return IntentType.UNKNOWN

    def _parse_intent_response(self, text: str) -> IntentType:
        """Parse and validate LLM output text into a valid IntentType."""
        cleaned = text.strip().lower()

        # Direct enum value match
        for member in IntentType:
            if member.value == cleaned:
                return member

        # Regex search in case of extra whitespace or punctuation
        for member in IntentType:
            pattern = rf"\b{re.escape(member.value)}\b"
            if re.search(pattern, cleaned):
                return member

        # Common synonym fallbacks
        if "change" in cleaned:
            return IntentType.CHANGE
        if "ground" in cleaned or "box" in cleaned or "locate" in cleaned:
            return IntentType.GROUNDING
        if "sar" in cleaned or "radar" in cleaned:
            return IntentType.SAR
        if "wildfire" in cleaned or "fire" in cleaned:
            return IntentType.DISASTER_CORRELATION
        if "disaster" in cleaned or "flood" in cleaned:
            return IntentType.DISASTER
        if "gis" in cleaned or "ndvi" in cleaned:
            return IntentType.GIS
        if "vqa" in cleaned or "question" in cleaned or "count" in cleaned:
            return IntentType.VQA

        return IntentType.UNKNOWN
