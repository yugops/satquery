import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import importlib
from core.config import SatQueryConfig
from core.models import AgentResponse, AgentStatus, ComputationMode

mod = importlib.import_module('agents.01_lead_orchestrator.agent')
LeadOrchestratorAgent = mod.LeadOrchestratorAgent

orchestrator = LeadOrchestratorAgent(SatQueryConfig(), agent_registry={})

# Call orchestrator with no geo_validator trace present
res = asyncio.run(orchestrator.run(query='evaluate this satellite scene'))
print(f"primary_answer: {res.primary_answer}")

assert "unavailable" in res.primary_answer.lower()
assert "chesapeake" not in res.primary_answer.lower()
assert "791" not in res.primary_answer
assert "718" not in res.primary_answer

print("VERIFICATION_SUCCESS")
