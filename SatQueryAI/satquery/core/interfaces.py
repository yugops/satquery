"""
SatQuery AI — Abstract Interfaces

Defines BaseAgent and ModelAdapter — the two core abstractions that
every agent and every ML model wrapper must implement.

BaseAgent.run() provides automatic timing, error handling, structured
AgentResponse wrapping, and OutputType tagging. Subclasses only
implement _execute().
"""

from __future__ import annotations

import time
import traceback
from abc import ABC, abstractmethod
from typing import Any

from core.config import SatQueryConfig, config
from core.exceptions import AgentExecutionError, ModelUnavailableError
from core.logging import get_logger, log_agent_event
from core.models import AgentResponse, AgentStatus, OutputType


# ---------------------------------------------------------------------------
# ModelAdapter — interface for pluggable ML models
# ---------------------------------------------------------------------------

class ModelAdapter(ABC):
    """Abstract interface for external ML model loading and inference.

    Supports mock/real switching: in mock mode, adapters return
    plausible synthetic outputs without loading actual model weights.
    """

    def __init__(self, model_id: str = ""):
        self.model_id = model_id
        self._loaded = False
        self._model: Any = None

    @abstractmethod
    def load(self, config: SatQueryConfig) -> None:
        """Load the model weights. Called once during initialization
        when mode='real'. Should set self._loaded = True on success."""
        ...

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Run inference. Implementations should check self._loaded
        and raise ModelUnavailableError if not loaded."""
        ...

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def ensure_loaded(self) -> None:
        """Guard that raises if the model was never loaded."""
        if not self._loaded:
            raise ModelUnavailableError(self.model_id)


# ---------------------------------------------------------------------------
# BaseAgent — template for all SatQuery agents
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """Abstract base class for all SatQuery AI agents.

    Provides:
    - Automatic timing of _execute()
    - Structured AgentResponse wrapping
    - Error handling → AgentStatus.FAILED responses (never crashes)
    - OutputType tagging (mock vs model vs rule_based)
    - Logging of lifecycle events

    Subclasses must implement:
    - agent_name (class attribute)
    - mempalace_wing (class attribute)
    - _execute(**kwargs) → Any
    """

    agent_name: str = "BaseAgent"
    mempalace_wing: str = "Wing: Unassigned"

    def __init__(self, config: SatQueryConfig | None = None):
        self.config = config or SatQueryConfig()
        self.logger = get_logger(self.agent_name, self.config.log_level)

    async def run(self, **kwargs: Any) -> AgentResponse:
        """Execute the agent with full lifecycle management.

        Returns:
            AgentResponse with timing, status, output_type, and result.
        """
        log_agent_event(
            self.logger, "start",
            agent=self.agent_name,
            session_id=kwargs.get("session_id", ""),
        )

        start = time.perf_counter()
        try:
            result = await self._execute(**kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000

            output_type = (
                OutputType.MOCK if self.config.mode == "mock"
                else OutputType.MODEL
            )

            response = AgentResponse(
                agent=self.agent_name,
                status=AgentStatus.SUCCESS,
                result=result,
                confidence=self._extract_confidence(result),
                execution_time_ms=round(elapsed_ms, 2),
                output_type=output_type,
                metadata={
                    "mempalace_wing": self.mempalace_wing,
                    "mode": self.config.mode,
                },
            )

            log_agent_event(
                self.logger, "complete",
                agent=self.agent_name,
                session_id=kwargs.get("session_id", ""),
                extra={
                    "status": response.status,
                    "elapsed_ms": response.execution_time_ms,
                    "confidence": response.confidence,
                },
            )
            return response

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            error_detail = traceback.format_exc()

            log_agent_event(
                self.logger, "error",
                agent=self.agent_name,
                session_id=kwargs.get("session_id", ""),
                extra={"error": str(exc)},
            )

            return AgentResponse(
                agent=self.agent_name,
                status=AgentStatus.FAILED,
                result=None,
                confidence=0.0,
                execution_time_ms=round(elapsed_ms, 2),
                output_type=OutputType.UNAVAILABLE,
                metadata={
                    "mempalace_wing": self.mempalace_wing,
                    "error": str(exc),
                    "error_detail": error_detail,
                },
            )

    @abstractmethod
    async def _execute(self, **kwargs: Any) -> Any:
        """Core agent logic. Subclasses implement this.

        Should return a Pydantic model or serializable result.
        Exceptions are caught by run() and converted to FAILED responses.
        """
        ...

    def _extract_confidence(self, result: Any) -> float:
        """Attempt to pull a confidence score from the result object.
        Override in subclasses for custom extraction logic."""
        if hasattr(result, "confidence"):
            return float(result.confidence)
        if isinstance(result, dict) and "confidence" in result:
            return float(result["confidence"])
        # Default confidence for successful mock/rule-based results
        return 0.85 if self.config.mode == "mock" else 0.0
