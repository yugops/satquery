import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import importlib
import pytest
from core.config import SatQueryConfig

tools_mod = importlib.import_module('agents.02_geo_validator.tools')
agent_mod = importlib.import_module('agents.02_geo_validator.agent')

# 1. Test direct tool call
raised_fnf = False
try:
    tools_mod.tool_extract_geotiff_metadata("nonexistent.tif")
except FileNotFoundError as e:
    raised_fnf = True
    print(f"Tool raised expected FileNotFoundError: {e}")

assert raised_fnf, "tool_extract_geotiff_metadata did not raise FileNotFoundError"

# 2. Test agent response
agent = agent_mod.GeoValidatorAgent(SatQueryConfig())
resp = asyncio.run(agent.execute(image_path="nonexistent.tif"))
print(f"agent status: {resp.status.value}")
print(f"agent summary: {resp.summary}")
print(f"agent confidence: {resp.confidence}")

assert resp.status.value == "error"
assert "Error in GeoValidatorAgent" in resp.summary
assert resp.confidence == 0.0

print("VERIFICATION_SUCCESS")
