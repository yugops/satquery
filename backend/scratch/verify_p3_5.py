import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import importlib
from core.config import SatQueryConfig

agent_mod = importlib.import_module('agents.12_wildfire_burn_scar.agent')
WildfireAndBurnScarAgent = agent_mod.WildfireAndBurnScarAgent
agent = WildfireAndBurnScarAgent(SatQueryConfig())

multiband_file = "d:/satquery/test/wayanad_pre_multiband_t1.tif"

# Run 1
resp1 = asyncio.run(agent.execute(image_path=multiband_file, district="Wayanad, Kerala"))
# Run 2
resp2 = asyncio.run(agent.execute(image_path=multiband_file, district="Wayanad, Kerala"))

print("=== RUN 1 OUTPUT ===")
print(f"Burn scar hectares: {resp1.data['burn_scar_hectares']} ha, pixels: {resp1.data['burn_scar_pixels']}")
print(f"Spread estimate: {resp1.data['wind_spread_estimate']}")
print(f"Summary: {resp1.summary}")

print("\n=== RUN 2 OUTPUT ===")
print(f"Burn scar hectares: {resp2.data['burn_scar_hectares']} ha, pixels: {resp2.data['burn_scar_pixels']}")
print(f"Spread estimate: {resp2.data['wind_spread_estimate']}")
print(f"Summary: {resp2.summary}")

# Verify exact deterministic identity
assert resp1.data['burn_scar_hectares'] == resp2.data['burn_scar_hectares'], "Burn scar area must be identical across runs!"
assert resp1.data['burn_scar_pixels'] == resp2.data['burn_scar_pixels']

# Check Rothermel claim
assert "Rothermel Surface Fire Spread (1972)" not in resp1.summary
assert "NOT full Rothermel 1972" in resp1.data['wind_spread_estimate']['model']

print("\nRuns are 100% IDENTICAL and deterministic.")
print("VERIFICATION_SUCCESS")
