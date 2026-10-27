import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import importlib
from core.config import SatQueryConfig
from core.models import AgentResponse, AgentStatus, ComputationMode

agent_mod = importlib.import_module('agents.20_verifiable_ledger.agent')
VerifiableLedgerAgent = agent_mod.VerifiableLedgerAgent

agent = VerifiableLedgerAgent(SatQueryConfig())

trace = AgentResponse(
    agent_id="test_agent", agent_name="Test", mempalace_wing="w",
    status=AgentStatus.SUCCESS, computation_mode=ComputationMode.REAL_INFERENCE,
    summary="Analysis succeeded.", data={}
)

resp = asyncio.run(agent.execute(agent_traces=[trace]))
status = resp.data.get("admissibility_status")
print(f"admissibility_status: {status}")

assert "TAMPER-EVIDENT:" in status
assert "does NOT certify" in status
assert "CERTIFIED: Full cryptographic provenance" not in status

print("VERIFICATION_SUCCESS")
