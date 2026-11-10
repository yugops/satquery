"""
SatQuery AI — Agent 19: BhashiniVoiceAgent
Problem Statement ID: 26167 (ISRO / SAC)
Wing: Tactical_Voice_Dispatch
"""

from __future__ import annotations

from typing import Any, Optional
from core.interfaces import BaseAgent
from core.models import ComputationMode
from .tools import tool_bhashini_dispatch_sitrep


class BhashiniVoiceAgent(BaseAgent):
    agent_id = "bhashini_voice"
    agent_name = "BhashiniVoiceAgent"
    mempalace_wing = "Wing: Tactical_Voice_Dispatch"

    async def _execute(self, **kwargs: Any) -> tuple[dict[str, Any], ComputationMode, float, Optional[str], Optional[str]]:
        sitrep_text = kwargs.get("sitrep_text") or kwargs.get("query", "")
        target_lang = kwargs.get("language", "hi")

        if not sitrep_text:
            return {}, ComputationMode.RULE_BASED, 0.0, (
                "No SitRep text or query provided for translation."
            ), None

        res = tool_bhashini_dispatch_sitrep(sitrep_text, target_lang)

        summary = f"Multilingual translation generated in {res['language_name']} (Bhashini IndicTTS pending)."
        data = {
            "dispatch": res,
            "translated_text": res["localized_sitrep_text"],
            "target_language": res["target_language"],
            "language_name": res["language_name"],
            "tts_audio_ready": False,
        }

        return data, ComputationMode.API_INFERENCE, 0.70, summary, None
