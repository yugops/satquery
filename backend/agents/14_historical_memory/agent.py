"""
SatQuery AI — Agent 14: HistoricalMemoryAgent
Problem Statement ID: 26167 (ISRO / SAC)
Wing: Global_Historical_Memory
"""

from __future__ import annotations

from typing import Any, Optional
from core.interfaces import BaseAgent
from core.models import ComputationMode
from .tools import tool_mempalace_recall_precedent


class HistoricalMemoryAgent(BaseAgent):
    agent_id = "historical_memory"
    agent_name = "HistoricalMemoryAgent"
    mempalace_wing = "Wing: Global_Historical_Memory"

    async def _execute(self, **kwargs: Any) -> tuple[dict[str, Any], ComputationMode, float, Optional[str], Optional[str]]:
        district = kwargs.get("district") or kwargs.get("region", "")
        disaster_type = kwargs.get("disaster_type", "")

        precedents = tool_mempalace_recall_precedent(district, disaster_type)

        if not precedents:
            summary = f"No historical disaster precedent found in MemPalace database for district='{district}'."
            data = {
                "precedents": [],
                "matched_count": 0,
                "primary_precedent": None,
            }
            return data, ComputationMode.RULE_BASED, 0.0, summary, None

        top = precedents[0]
        summary = (
            f"MemPalace recalled historical precedent for {top['district']}, {top['state']} ({top['date']}): "
            f"{top['impact_summary']}"
        )

        data = {
            "precedents": precedents,
            "matched_count": len(precedents),
            "primary_precedent": top,
        }

        return data, ComputationMode.RULE_BASED, 0.95, summary, None
