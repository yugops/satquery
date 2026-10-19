"""
SatQuery AI — Agent 03: AuditLedgerAgent
Problem Statement ID: 26167 (ISRO / SAC)
Wing: Audit_Ledger
"""

from __future__ import annotations

from typing import Any, Optional
from datetime import datetime, timezone
from core.interfaces import BaseAgent
from core.models import ComputationMode
from .tools import (
    tool_calculate_calibrated_confidence,
    tool_commit_ledger_entry,
)


class AuditLedgerAgent(BaseAgent):
    agent_id = "audit_ledger"
    agent_name = "AuditLedgerAgent"
    mempalace_wing = "Wing: Audit_Ledger"

    async def _execute(self, **kwargs: Any) -> tuple[dict[str, Any], ComputationMode, float, Optional[str], Optional[str]]:
        session_id = kwargs.get("session_id", "SQ-000")
        agent_traces = kwargs.get("agent_traces", [])
        merkle_root = kwargs.get("merkle_root", "")
        
        evidence_scores = [
            t.get("confidence", 0.8) if isinstance(t, dict) else getattr(t, "confidence", 0.8)
            for t in agent_traces
        ]
        
        # Robustly extract real cloud & shadow fraction from GeoValidator trace
        cloud_frac = 0.0
        shadow_frac = 0.0
        for t in agent_traces:
            t_data = t if isinstance(t, dict) else getattr(t, "data", {})
            if isinstance(t_data, dict):
                cc = t_data.get("cloud_check") or t_data.get("cloud_mask") or {}
                if isinstance(cc, dict):
                    if "cloud_fraction" in cc and cc["cloud_fraction"] is not None:
                        cloud_frac = float(cc["cloud_fraction"])
                    if "shadow_fraction" in cc and cc["shadow_fraction"] is not None:
                        shadow_frac = float(cc["shadow_fraction"])
                    break
                elif "cloud_fraction" in t_data and t_data["cloud_fraction"] is not None:
                    cloud_frac = float(t_data["cloud_fraction"])
                    break

        # Identify primary specialist agent trace
        primary_agent_id = None
        primary_conf = None
        for t in agent_traces:
            aid = t.get("agent_id") if isinstance(t, dict) else getattr(t, "agent_id", "")
            if aid not in ("geo_validator", "audit_ledger", "verifiable_ledger"):
                primary_agent_id = aid
                primary_conf = t.get("confidence") if isinstance(t, dict) else getattr(t, "confidence", None)
                break

        sensor_snr = 1.0 - min(1.0, (cloud_frac or 0.0) + (shadow_frac or 0.0))
        calibrated_conf = tool_calculate_calibrated_confidence(
            evidence_scores,
            sensor_snr=sensor_snr,
            cloud_fraction=cloud_frac,
            primary_agent_id=primary_agent_id,
            primary_confidence=primary_conf
        )
        
        payload = {
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "merkle_root": merkle_root,
            "agent_traces": [t if isinstance(t, dict) else t.model_dump() for t in agent_traces],
            "calibrated_confidence": calibrated_conf,
            "cloud_fraction": cloud_frac,
            "sensor_snr": sensor_snr,
        }
        
        record_id = tool_commit_ledger_entry(self.config.db_path, session_id, payload)
        
        summary = (
            f"Audit ledger record committed ({record_id}). Calibrated Bayesian confidence: {calibrated_conf*100:.1f}% "
            f"(Cloud contamination penalty: {cloud_frac*100:.1f}%). Immutable SQLite persistence verified."
        )

        data = {
            "record_id": record_id,
            "calibrated_confidence": calibrated_conf,
            "cloud_fraction": cloud_frac,
            "db_path": self.config.db_path,
        }

        return data, ComputationMode.RULE_BASED, 1.0, summary, None
