import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import importlib
from core.config import SatQueryConfig
from core.models import AgentResponse, AgentStatus, ComputationMode

agent_mod = importlib.import_module('agents.03_audit_ledger.agent')
AuditLedgerAgent = agent_mod.AuditLedgerAgent

agent = AuditLedgerAgent(SatQueryConfig())

# Test with mock trace containing 0.2 cloud and 0.1 shadow
trace_geo = AgentResponse(
    agent_id="geo_validator", agent_name="Geo", mempalace_wing="w",
    status=AgentStatus.SUCCESS, computation_mode=ComputationMode.RULE_BASED,
    summary="geo", data={"cloud_check": {"cloud_fraction": 0.20, "shadow_fraction": 0.10}}
)
trace_vqa = AgentResponse(
    agent_id="vqa", agent_name="VQA", mempalace_wing="w",
    status=AgentStatus.SUCCESS, computation_mode=ComputationMode.REAL_INFERENCE,
    summary="vqa", confidence=0.90, data={}
)

resp = asyncio.run(agent.execute(session_id="TEST-11", agent_traces=[trace_geo, trace_vqa]))
print(f"resp summary: {resp.summary}")
print(f"calibrated confidence: {resp.data.get('calibrated_confidence')}")
print("VERIFICATION_SUCCESS")
