"""
SatQuery AI — Agent 3: AuditLedgerAgent

Wing: Audit_Compliance

Logs every agent execution to a local SQLite database for full
traceability. Provides calibrated confidence scoring via Bayesian
weighting and sensor SNR degradation.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from core.config import SatQueryConfig
from core.interfaces import BaseAgent
from core.models import LedgerEntry


# ---------------------------------------------------------------------------
# SQLite Schema
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS audit_ledger (
    entry_id           TEXT PRIMARY KEY,
    session_id         TEXT NOT NULL,
    timestamp          TEXT NOT NULL,
    agent_name         TEXT NOT NULL,
    query              TEXT,
    intent             TEXT,
    execution_duration_ms REAL,
    model_version      TEXT,
    input_summary      TEXT,
    output_summary     TEXT,
    confidence         REAL,
    status             TEXT,
    output_type        TEXT,
    error_detail       TEXT
);
"""

_INSERT_SQL = """
INSERT INTO audit_ledger (
    entry_id, session_id, timestamp, agent_name, query, intent,
    execution_duration_ms, model_version, input_summary, output_summary,
    confidence, status, output_type, error_detail
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""


# ---------------------------------------------------------------------------
# Tool Functions (standalone, usable without the agent wrapper)
# ---------------------------------------------------------------------------

def tool_calculate_calibrated_confidence(
    evidence_scores: list[float],
    sensor_snr: float | None = None,
) -> float:
    """Compute a Bayesian-weighted average of evidence scores, degraded
    by sensor signal-to-noise ratio when available.

    Args:
        evidence_scores: List of per-agent confidence scores in [0, 1].
        sensor_snr: Optional sensor SNR in dB. Lower SNR degrades confidence.

    Returns:
        Calibrated confidence clamped to [0.0, 1.0].
    """
    if not evidence_scores:
        return 0.0

    # Bayesian-weighted average: higher scores get more weight
    weights = [s ** 2 for s in evidence_scores]
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0

    weighted_avg = sum(s * w for s, w in zip(evidence_scores, weights)) / total_weight

    # Apply SNR degradation factor
    if sensor_snr is not None and sensor_snr > 0:
        # Typical satellite SNR: 20–40 dB.  Below 10 dB → heavy degradation.
        snr_factor = min(1.0, sensor_snr / 30.0)
        weighted_avg *= snr_factor

    return max(0.0, min(1.0, round(weighted_avg, 4)))


def tool_commit_ledger_entry(
    session_id: str,
    trace_data: dict[str, Any],
    db_path: str = "satquery.db",
) -> str:
    """Insert an audit record into the SQLite ledger.

    Args:
        session_id: Current session/trace ID.
        trace_data: Dict with keys matching LedgerEntry fields.
        db_path: Path to the SQLite database file.

    Returns:
        The generated entry_id (UUID).
    """
    entry_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(_CREATE_TABLE_SQL)

        conn.execute(_INSERT_SQL, (
            entry_id,
            session_id,
            now,
            trace_data.get("agent_name", ""),
            trace_data.get("query", ""),
            trace_data.get("intent", ""),
            trace_data.get("execution_duration_ms", 0.0),
            trace_data.get("model_version", ""),
            _safe_json(trace_data.get("input_summary", "")),
            _safe_json(trace_data.get("output_summary", "")),
            trace_data.get("confidence", 0.0),
            trace_data.get("status", "success"),
            trace_data.get("output_type", "mock"),
            trace_data.get("error_detail", ""),
        ))
        conn.commit()
    finally:
        conn.close()

    return entry_id


# ---------------------------------------------------------------------------
# AuditLedgerAgent
# ---------------------------------------------------------------------------

class AuditLedgerAgent(BaseAgent):
    """Agent 3 — Logs execution traces to SQLite for auditability.

    This agent is invoked by the orchestrator after specialist agents
    complete. It receives the list of AgentResponses and commits each
    to the audit_ledger table.
    """

    agent_name = "AuditLedgerAgent"
    mempalace_wing = "Wing: Audit_Compliance"

    def __init__(self, config: SatQueryConfig | None = None):
        super().__init__(config)
        self._db_path = self.config.db_path

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        """Commit audit entries for the current session.

        Expected kwargs:
            session_id (str): Session identifier.
            query (str): Original user query.
            intent (str): Classified intent type.
            agent_responses (list[AgentResponse]): Results to audit.

        Returns:
            Dict with entry_ids and calibrated confidence.
        """
        session_id = kwargs.get("session_id", str(uuid.uuid4()))
        query = kwargs.get("query", "")
        intent = kwargs.get("intent", "")
        agent_responses = kwargs.get("agent_responses", [])

        entry_ids = []
        evidence_scores = []

        for resp in agent_responses:
            res = getattr(resp, "result", None)
            answer_text = ""
            if isinstance(res, dict):
                answer_text = res.get("answer") or res.get("summary") or res.get("synthesized_answer") or str(res)
            elif hasattr(res, "answer"):
                answer_text = res.answer
            elif hasattr(res, "summary"):
                answer_text = res.summary
            elif res is not None:
                answer_text = str(res)

            model_name = getattr(resp, "metadata", {}).get("model", "") or (
                getattr(res, "model", "") if hasattr(res, "model") else (res.get("model", "") if isinstance(res, dict) else "")
            ) or getattr(resp, "metadata", {}).get("mode", "")

            source_val = getattr(res, "source", None) or (res.get("source") if isinstance(res, dict) else None) or (
                resp.output_type.value if hasattr(resp.output_type, "value") else str(resp.output_type)
            )

            status_val = resp.status.value if hasattr(resp.status, "value") else str(resp.status)
            conf_val = getattr(resp, "confidence", None)

            # Build trace data from AgentResponse
            trace = {
                "agent_name": resp.agent if hasattr(resp, "agent") else str(resp),
                "query": query,
                "intent": intent,
                "execution_duration_ms": getattr(resp, "execution_time_ms", 0.0),
                "model_version": model_name,
                "input_summary": query,
                "output_summary": answer_text[:1000] if answer_text else "",
                "confidence": conf_val if conf_val is not None else 0.0,
                "status": status_val,
                "output_type": str(source_val),
                "error_detail": getattr(resp, "metadata", {}).get("error", ""),
            }

            eid = tool_commit_ledger_entry(session_id, trace, self._db_path)
            entry_ids.append(eid)
            if conf_val is not None and conf_val > 0:
                evidence_scores.append(conf_val)

        calibrated = tool_calculate_calibrated_confidence(
            evidence_scores,
            sensor_snr=kwargs.get("sensor_snr"),
        )

        return {
            "entry_ids": entry_ids,
            "entries_committed": len(entry_ids),
            "calibrated_confidence": calibrated if evidence_scores else None,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json(value: Any) -> str:
    """Convert a value to a JSON string, falling back to str()."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)
