import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import importlib
from core.config import SatQueryConfig
from core.models import ComputationMode

agent_mod = importlib.import_module('agents.07_sar_cloud_penetration.agent')
SARCloudPenetrationAgent = agent_mod.SARCloudPenetrationAgent

agent = SARCloudPenetrationAgent(SatQueryConfig())

# Call with empty input
resp = asyncio.run(agent.execute(vv_tif="", vh_tif="", image_path=""))
print(f"computation_mode: {resp.computation_mode.value}")
print(f"confidence: {resp.confidence}")
print(f"summary: {resp.summary}")

assert resp.computation_mode == ComputationMode.RULE_BASED
assert resp.confidence == 0.0
assert "No SAR VV/VH input provided" in resp.summary
assert resp.computation_mode != ComputationMode.REAL_INFERENCE

print("VERIFICATION_SUCCESS")
