"""
SatQuery AI — Base Interfaces & Agent Contracts
Problem Statement ID: 26167 (ISRO / SAC)

Defines BaseAgent abstract class enforcing computation_mode tracking,
timing, exception wrapping, and MemPalace integration.
"""

from __future__ import annotations

import abc
import hashlib
import time
from typing import Any, Optional
from pathlib import Path

from core.config import SatQueryConfig
from core.models import AgentResponse, AgentStatus, ComputationMode


class BaseAgent(abc.ABC):
    """Abstract base class for all 22 SatQuery AI specialist agents."""

    agent_id: str = "base_agent"
    agent_name: str = "BaseAgent"
    mempalace_wing: str = "Wing: General"

    def __init__(self, config: SatQueryConfig) -> None:
        self.config = config

    @abc.abstractmethod
    async def _execute(self, **kwargs: Any) -> tuple[dict[str, Any], ComputationMode, float, Optional[str], Optional[str]]:
        """
        Execute specialist agent logic.
        Must return a tuple: (data_dict, computation_mode, confidence, summary_text, model_weights_sha256).
        """
        raise NotImplementedError

    async def execute(self, **kwargs: Any) -> AgentResponse:
        """Executes the agent, tracks latency, catches errors, and formats AgentResponse."""
        start_time = time.perf_counter()
        try:
            data, comp_mode, conf, summary, weights_sha256 = await self._execute(**kwargs)
            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            
            status = AgentStatus.SUCCESS if comp_mode in (ComputationMode.REAL_INFERENCE, ComputationMode.API_INFERENCE, ComputationMode.RULE_BASED) else AgentStatus.FALLBACK
            
            return AgentResponse(
                agent_id=self.agent_id,
                agent_name=self.agent_name,
                mempalace_wing=self.mempalace_wing,
                status=status,
                computation_mode=comp_mode,
                model_weights_sha256=weights_sha256,
                execution_time_ms=elapsed_ms,
                confidence=conf,
                summary=summary or f"{self.agent_name} executed successfully.",
                data=data,
            )
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            return AgentResponse(
                agent_id=self.agent_id,
                agent_name=self.agent_name,
                mempalace_wing=self.mempalace_wing,
                status=AgentStatus.ERROR,
                computation_mode=ComputationMode.SYNTHETIC_FALLBACK,
                model_weights_sha256=None,
                execution_time_ms=elapsed_ms,
                confidence=0.0,
                summary=f"Error in {self.agent_name}: {str(exc)}",
                data={"error": str(exc)},
            )


def compute_file_sha256(file_path: str | Path) -> str:
    """Compute SHA-256 hash of a file for cryptographic integrity."""
    p = Path(file_path)
    if not p.is_file():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()
