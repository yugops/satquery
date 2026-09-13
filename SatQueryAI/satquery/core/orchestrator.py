"""
SatQuery AI — Agent 1: LeadOrchestratorAgent

Wing: Execution_Control

Central controller that accepts user queries and satellite imagery,
classifies intent, builds an execution DAG, dispatches specialist
agents (with async parallel support), collects results, audits
via AuditLedgerAgent, and synthesizes a final structured response.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import Any

import numpy as np
from core.adapters.groq_adapter import GroqAdapter
from core.config import SatQueryConfig
from core.exceptions import DAGExecutionError
from core.interfaces import BaseAgent
from core.ledger import AuditLedgerAgent
from core.logging import get_logger, log_agent_event
from core.models import (
    AgentExecutionNode,
    AgentResponse,
    AgentStatus,
    FinalResponse,
    GeoValidationResult,
    IntentPlan,
    IntentType,
    OutputType,
)
from core.validator import GeoValidatorAgent
from vision_swarm.change_agent import BiTemporalChangeAgent
from vision_swarm.disaster_correlation_agent import DisasterCorrelationAgent
from vision_swarm.grounding_agent import VisualGroundingAgent
from vision_swarm.sar_agent import SARCloudPenetrationAgent
from vision_swarm.vqa_agent import SingleSceneVQAAgent


# ---------------------------------------------------------------------------
# Intent Classification Keywords
# ---------------------------------------------------------------------------

_INTENT_KEYWORDS: dict[IntentType, list[str]] = {
    IntentType.VQA: [
        "what", "how many", "describe", "is there", "are there",
        "count", "identify", "scene", "visible", "tell me",
        "explain", "show me", "what type", "what kind",
    ],
    IntentType.CHANGE: [
        "change", "changed", "difference", "before and after",
        "compare", "temporal", "construction", "deforestation",
        "growth", "expansion", "between",
    ],
    IntentType.GROUNDING: [
        "locate", "find", "where is", "where are", "bounding box",
        "segment", "localize", "detect objects", "ground",
        "position of", "show location",
    ],
    IntentType.SAR: [
        "sar", "radar", "sentinel-1", "sentinel 1", "s1",
        "backscatter", "vv", "vh", "polarization",
        "synthetic aperture",
    ],
    IntentType.DISASTER: [
        "disaster", "flood", "earthquake", "cyclone",
        "damage", "hurricane", "tsunami", "landslide", "storm",
        "emergency", "relief",
    ],
    IntentType.DISASTER_CORRELATION: [
        "wildfire", "fire", "active fire", "thermal anomaly", "kanari",
        "burn scar", "fire perimeter", "smoke plume",
    ],
    IntentType.GIS: [
        "land cover", "land use", "urban", "agriculture", "forest",
        "classification", "gis", "ndvi", "vegetation index",
        "area calculation", "zonal",
    ],
}

# Maps intent types to the agent class names they require
_INTENT_TO_AGENTS: dict[IntentType, list[str]] = {
    IntentType.VQA: ["SingleSceneVQAAgent"],
    IntentType.CHANGE: ["BiTemporalChangeAgent"],
    IntentType.GROUNDING: ["VisualGroundingAgent"],
    IntentType.SAR: ["SARCloudPenetrationAgent"],
    IntentType.DISASTER: ["BiTemporalChangeAgent"],  # Part 1 fallback
    IntentType.DISASTER_CORRELATION: ["DisasterCorrelationAgent"],
    IntentType.GIS: ["SingleSceneVQAAgent"],          # Part 1 fallback
    IntentType.UNKNOWN: ["SingleSceneVQAAgent"],      # Default to VQA
}


# ---------------------------------------------------------------------------
# Tool Functions
# ---------------------------------------------------------------------------

def tool_classify_intent(
    query: str,
    has_pair: bool = False,
    is_sar: bool = False,
) -> IntentPlan:
    """Classify the user's intent from their query and image context.

    Uses keyword matching with weighted scoring. Detects compound
    intents and returns the list of required agents.

    Args:
        query: Natural language user query.
        has_pair: Whether an image pair (t1, t2) is available.
        is_sar: Whether the input is identified as SAR data.

    Returns:
        IntentPlan with intent_type, requires_agents, and confidence.
    """
    lower_query = query.lower()
    scores: dict[IntentType, float] = {}

    for intent_type, keywords in _INTENT_KEYWORDS.items():
        score = sum(1.0 for kw in keywords if kw in lower_query)
        if score > 0:
            scores[intent_type] = score

    # Contextual boosts
    if is_sar:
        scores[IntentType.SAR] = scores.get(IntentType.SAR, 0) + 3.0
    if has_pair:
        scores[IntentType.CHANGE] = scores.get(IntentType.CHANGE, 0) + 2.0

    if not scores:
        # Default to VQA for unrecognized queries
        return IntentPlan(
            intent_type=IntentType.VQA,
            has_pair=has_pair,
            is_sar=is_sar,
            requires_agents=_INTENT_TO_AGENTS[IntentType.VQA],
            confidence=0.4,
            raw_query=query,
        )

    # Pick the highest-scoring intent
    best_intent = max(scores, key=lambda k: scores[k])
    best_score = scores[best_intent]
    max_possible = max(len(kws) for kws in _INTENT_KEYWORDS.values())
    confidence = min(0.95, 0.5 + (best_score / max_possible) * 0.5)

    requires = list(_INTENT_TO_AGENTS.get(best_intent, []))

    return IntentPlan(
        intent_type=best_intent,
        has_pair=has_pair,
        is_sar=is_sar,
        requires_agents=requires,
        confidence=round(confidence, 3),
        raw_query=query,
    )


def tool_build_dag_plan(intent: IntentPlan) -> list[AgentExecutionNode]:
    """Convert an IntentPlan into an ordered execution DAG.

    Always starts with GeoValidatorAgent. Specialist agents depend
    on the validator. Parallelizable agents share the same dependency.

    Args:
        intent: Classified intent plan.

    Returns:
        Ordered list of AgentExecutionNode objects.
    """
    nodes: list[AgentExecutionNode] = []

    # Node 0: Validator always runs first
    nodes.append(AgentExecutionNode(
        agent_name="GeoValidatorAgent",
        depends_on=[],
        priority=0,
    ))

    # Specialist nodes depend on validator
    for i, agent_name in enumerate(intent.requires_agents):
        nodes.append(AgentExecutionNode(
            agent_name=agent_name,
            depends_on=["GeoValidatorAgent"],
            priority=1,
        ))

    return nodes


def tool_synthesize_evidence(
    agent_results: list[AgentResponse],
    query: str = "",
) -> str:
    """Merge multiple agent results into a synthesized text answer.

    Picks the primary answer from the highest-confidence successful
    agent, and appends supplementary information from others.

    Args:
        agent_results: List of AgentResponse objects from specialist agents.
        query: Original user query for context.

    Returns:
        Synthesized natural language answer string.
    """
    successful = [
        r for r in agent_results
        if r.status == AgentStatus.SUCCESS and r.result is not None
    ]

    if not successful:
        return "Unable to produce an analysis. All specialist agents failed or were skipped."

    # Sort by confidence descending
    successful.sort(key=lambda r: r.confidence, reverse=True)
    primary = successful[0]

    parts = []

    # Extract the main answer
    result = primary.result
    if isinstance(result, dict):
        # Look for common answer fields
        for key in ("answer", "summary", "synthesized_answer"):
            if key in result:
                parts.append(str(result[key]))
                break
        else:
            # Fallback: use the first meaningful string value
            for k, v in result.items():
                if isinstance(v, str) and len(v) > 10:
                    parts.append(v)
                    break
    elif hasattr(result, "answer"):
        parts.append(result.answer)
    elif hasattr(result, "summary"):
        parts.append(result.summary)
    elif isinstance(result, dict):
        summary_items = [f"{k}: {v}" for k, v in result.items() if not isinstance(v, (dict, list))]
        if summary_items:
            parts.append("; ".join(summary_items))
        else:
            parts.append(str(result))
    else:
        parts.append(str(result))

    # Append supplementary from other agents
    for resp in successful[1:]:
        r = resp.result
        supplement = ""
        if isinstance(r, dict):
            supplement = r.get("summary", r.get("answer", ""))
        elif hasattr(r, "summary"):
            supplement = r.summary
        if supplement:
            parts.append(f"[{resp.agent}]: {supplement}")

    # Add output type annotation
    output_label = primary.output_type.value.upper()
    parts.append(f"\n[Output Source: {output_label}]")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LeadOrchestratorAgent
# ---------------------------------------------------------------------------

class LeadOrchestratorAgent(BaseAgent):
    """Agent 1 — Central orchestrator for the SatQuery AI pipeline.

    Manages the full lifecycle: intent classification → DAG construction
    → async agent dispatch → audit logging → response synthesis.

    Usage:
        agent = LeadOrchestratorAgent()
        result = agent.run(query="How many buildings?", image_path="scene.tif")
    """

    agent_name = "LeadOrchestratorAgent"
    mempalace_wing = "Wing: Execution_Control"

    def __init__(self, config: SatQueryConfig | None = None):
        super().__init__(config)
        self._session_id = str(uuid.uuid4())

        # ---- Build agent registry ----
        self._agents: dict[str, BaseAgent] = {
            "GeoValidatorAgent": GeoValidatorAgent(self.config),
            "SingleSceneVQAAgent": SingleSceneVQAAgent(self.config),
            "VisualGroundingAgent": VisualGroundingAgent(self.config),
            "BiTemporalChangeAgent": BiTemporalChangeAgent(self.config),
            "SARCloudPenetrationAgent": SARCloudPenetrationAgent(self.config),
            "DisasterCorrelationAgent": DisasterCorrelationAgent(self.config),
        }
        self._auditor = AuditLedgerAgent(self.config)

        # ---- Optional LLM Backend (Groq) ----
        self._groq_adapter: GroqAdapter | None = None
        if self.config.mode == "real" and self.config.llm_backend == "groq":
            self._groq_adapter = GroqAdapter(
                model_id=self.config.groq_model_id,
                api_key=self.config.groq_api_key,
            )
            try:
                self._groq_adapter.load(self.config)
            except Exception as exc:
                self.logger.warning("Failed to initialize GroqAdapter: %s. Falling back to rule-based.", exc)
                self._groq_adapter = None

    def run_sync(
        self,
        query: str,
        image_path: str | None = None,
        image_path_t2: str | None = None,
        image_array: np.ndarray | None = None,
        image_array_t2: np.ndarray | None = None,
        vv_tif: str | None = None,
        vh_tif: str | None = None,
    ) -> FinalResponse:
        """Synchronous entry point — wraps the async run() for convenience.

        Args:
            query: Natural language user query.
            image_path: Path to the primary image.
            image_path_t2: Path to the second image (for change detection).
            image_array: Pre-loaded primary image array.
            image_array_t2: Pre-loaded second image array.
            vv_tif: Path to VV SAR polarization file.
            vh_tif: Path to VH SAR polarization file.

        Returns:
            FinalResponse with all agent results and synthesized answer.
        """
        return asyncio.run(self.run(
            query=query,
            image_path=image_path,
            image_path_t2=image_path_t2,
            image_array=image_array,
            image_array_t2=image_array_t2,
            vv_tif=vv_tif,
            vh_tif=vh_tif,
        ))

    async def run(self, **kwargs: Any) -> FinalResponse:
        """Override BaseAgent.run() to return FinalResponse directly.

        The orchestrator is the top-level entry point, so callers
        expect FinalResponse, not AgentResponse.
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
            result.total_execution_ms = round(elapsed_ms, 2)

            log_agent_event(
                self.logger, "complete",
                agent=self.agent_name,
                extra={
                    "status": "success",
                    "elapsed_ms": result.total_execution_ms,
                    "confidence": 0.85,
                },
            )
            return result

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            log_agent_event(
                self.logger, "error",
                agent=self.agent_name,
                extra={"error": str(exc)},
            )
            # Return a minimal FinalResponse on catastrophic failure
            return FinalResponse(
                session_id=self._session_id,
                query=kwargs.get("query", ""),
                synthesized_answer=f"Pipeline error: {exc}",
                total_execution_ms=round(elapsed_ms, 2),
            )

    async def _execute(self, **kwargs: Any) -> FinalResponse:
        """Full orchestration pipeline.

        Expected kwargs:
            query (str): User query.
            image_path (str | None): Primary image path.
            image_path_t2 (str | None): Second image path.
            image_array (np.ndarray | None): Primary image array.
            image_array_t2 (np.ndarray | None): Second image array.
            vv_tif (str | None): VV SAR file path.
            vh_tif (str | None): VH SAR file path.

        Returns:
            FinalResponse.
        """
        start_time = time.perf_counter()
        session_id = self._session_id
        query = kwargs.get("query", "")
        image_path = kwargs.get("image_path")
        image_path_t2 = kwargs.get("image_path_t2")
        image_array = kwargs.get("image_array")
        image_array_t2 = kwargs.get("image_array_t2")
        vv_tif = kwargs.get("vv_tif")
        vh_tif = kwargs.get("vh_tif")

        has_pair = (image_path_t2 is not None) or (image_array_t2 is not None)

        # Detect SAR from file paths
        is_sar = False
        if vv_tif or vh_tif:
            is_sar = True
        elif image_path:
            lower = image_path.lower()
            is_sar = any(tag in lower for tag in ["_vv", "_vh", "sar", "s1"])

        # ---- Step 1: Classify intent ----
        classification_output_type = OutputType.RULE_BASED
        if self._groq_adapter is not None and self._groq_adapter.is_loaded:
            try:
                classified_intent_type = self._groq_adapter.predict(query)
                requires = list(_INTENT_TO_AGENTS.get(classified_intent_type, ["SingleSceneVQAAgent"]))
                intent = IntentPlan(
                    intent_type=classified_intent_type,
                    has_pair=has_pair,
                    is_sar=is_sar,
                    requires_agents=requires,
                    confidence=0.9,
                    raw_query=query,
                )
                classification_output_type = OutputType.MODEL
            except Exception as exc:
                self.logger.warning("Groq classification failed: %s. Falling back to rule-based.", exc)
                intent = tool_classify_intent(query, has_pair, is_sar)
        else:
            intent = tool_classify_intent(query, has_pair, is_sar)
            if self.config.mode == "mock":
                classification_output_type = OutputType.MOCK

        log_agent_event(
            self.logger, "intent_classified",
            agent=self.agent_name,
            session_id=session_id,
            extra={
                "intent": intent.intent_type,
                "confidence": intent.confidence,
                "output_type": classification_output_type.value,
            },
        )

        # ---- Step 2: Build DAG ----
        dag = tool_build_dag_plan(intent)

        # ---- Step 3: Execute DAG ----
        agent_responses: list[AgentResponse] = []

        # Phase A: Run GeoValidatorAgent
        validator_response = await self._run_agent(
            "GeoValidatorAgent",
            session_id=session_id,
            image_path=image_path,
            image_array=image_array,
        )
        agent_responses.append(validator_response)

        # Check for SAR fallback from cloud detection
        validation_result = validator_response.result or {}
        if isinstance(validation_result, dict):
            cloud = validation_result.get("cloud_result")
            if cloud and hasattr(cloud, "is_contaminated") and cloud.is_contaminated:
                # Override to SAR if cloud-contaminated and SAR data available
                if vv_tif or vh_tif:
                    intent.intent_type = IntentType.SAR
                    intent.requires_agents = ["SARCloudPenetrationAgent"]
                    log_agent_event(
                        self.logger, "sar_fallback",
                        agent=self.agent_name,
                        session_id=session_id,
                    )
            elif isinstance(cloud, dict) and cloud.get("is_contaminated"):
                if vv_tif or vh_tif:
                    intent.intent_type = IntentType.SAR
                    intent.requires_agents = ["SARCloudPenetrationAgent"]

            # Update is_sar from validator
            if validation_result.get("is_sar") and not is_sar:
                is_sar = True
                intent.is_sar = True
                if intent.intent_type not in (IntentType.SAR, IntentType.CHANGE):
                    intent.intent_type = IntentType.SAR
                    intent.requires_agents = ["SARCloudPenetrationAgent"]

        # Phase B: Run specialist agent(s) — parallel where possible
        specialist_nodes = [n for n in dag if n.agent_name != "GeoValidatorAgent"]

        if specialist_nodes:
            specialist_tasks = []
            for node in specialist_nodes:
                agent_kwargs = self._build_agent_kwargs(
                    node.agent_name,
                    query=query,
                    session_id=session_id,
                    image_path=image_path,
                    image_path_t2=image_path_t2,
                    image_array=image_array,
                    image_array_t2=image_array_t2,
                    vv_tif=vv_tif,
                    vh_tif=vh_tif,
                    validation_result=validation_result,
                )
                specialist_tasks.append(
                    self._run_agent(node.agent_name, **agent_kwargs)
                )

            # Execute specialists concurrently
            specialist_results = await asyncio.gather(
                *specialist_tasks, return_exceptions=True
            )

            for res in specialist_results:
                if isinstance(res, Exception):
                    agent_responses.append(AgentResponse(
                        agent="unknown",
                        status=AgentStatus.FAILED,
                        result=None,
                        confidence=0.0,
                        metadata={"error": str(res)},
                        output_type=OutputType.UNAVAILABLE,
                    ))
                else:
                    agent_responses.append(res)

        # ---- Step 4: Audit ----
        audit_response = await self._auditor.run(
            session_id=session_id,
            query=query,
            intent=intent.intent_type.value,
            agent_responses=agent_responses,
        )

        # ---- Step 5: Synthesize ----
        # Only synthesize from specialist agents (not validator/auditor)
        specialist_responses = [
            r for r in agent_responses
            if r.agent not in ("GeoValidatorAgent", "AuditLedgerAgent")
        ]
        synthesized = tool_synthesize_evidence(specialist_responses, query)

        # Extract location and satellite_tle from GeoValidatorAgent if available
        location = None
        satellite_tle = None
        for r in agent_responses:
            if r.agent == "GeoValidatorAgent" and r.result is not None:
                if isinstance(r.result, dict):
                    location = r.result.get("location")
                    satellite_tle = r.result.get("satellite_tle")
                elif hasattr(r.result, "location"):
                    location = r.result.location
                    satellite_tle = getattr(r.result, "satellite_tle", None)
                if location is not None or satellite_tle is not None:
                    break

        total_ms = (time.perf_counter() - start_time) * 1000

        return FinalResponse(
            session_id=session_id,
            query=query,
            intent=intent,
            agent_responses=agent_responses,
            synthesized_answer=synthesized,
            total_execution_ms=round(total_ms, 2),
            location=location,
            satellite_tle=satellite_tle,
        )

    async def _run_agent(
        self, agent_name: str, **kwargs: Any
    ) -> AgentResponse:
        """Look up and run an agent from the registry.

        Returns a FAILED AgentResponse if the agent is not registered
        (never raises).
        """
        agent = self._agents.get(agent_name)
        if agent is None:
            self.logger.warning(f"Agent '{agent_name}' not registered — skipping.")
            return AgentResponse(
                agent=agent_name,
                status=AgentStatus.SKIPPED,
                result=None,
                confidence=0.0,
                metadata={"reason": "Agent not registered in this Part."},
                output_type=OutputType.UNAVAILABLE,
            )
        return await agent.run(**kwargs)

    def _build_agent_kwargs(
        self,
        agent_name: str,
        *,
        query: str,
        session_id: str,
        image_path: str | None,
        image_path_t2: str | None,
        image_array: np.ndarray | None,
        image_array_t2: np.ndarray | None,
        vv_tif: str | None,
        vh_tif: str | None,
        validation_result: dict | None,
    ) -> dict[str, Any]:
        """Build the kwargs dict for a specific specialist agent."""
        base = {"session_id": session_id, "query": query}

        if agent_name == "SingleSceneVQAAgent":
            base["image"] = image_array if image_array is not None else image_path

        elif agent_name == "VisualGroundingAgent":
            base["image"] = image_array if image_array is not None else image_path

        elif agent_name == "BiTemporalChangeAgent":
            base["image_t1"] = image_array
            base["image_t2"] = image_array_t2
            # If arrays not provided, try loading from paths, falling back to raw path string
            if image_array is None and image_path:
                base["image_t1"] = self._safe_load_image(image_path) or image_path
            if image_array_t2 is None and image_path_t2:
                base["image_t2"] = self._safe_load_image(image_path_t2) or image_path_t2

        elif agent_name == "SARCloudPenetrationAgent":
            base["vv_tif"] = vv_tif or ""
            base["vh_tif"] = vh_tif or ""
            if image_array is not None:
                base["vv_array"] = image_array

        elif agent_name == "DisasterCorrelationAgent":
            loc = None
            if isinstance(validation_result, dict):
                loc = validation_result.get("location")
            elif hasattr(validation_result, "location"):
                loc = validation_result.location
            if loc is not None:
                base["location"] = loc
                if hasattr(loc, "bounding_box") and loc.bounding_box:
                    base["bounding_box"] = loc.bounding_box

        return base

    @staticmethod
    def _safe_load_image(path: str) -> np.ndarray | None:
        """Attempt to load an image file into a numpy array."""
        try:
            from PIL import Image

            return np.array(Image.open(path).convert("RGB"))
        except Exception:
            return None
