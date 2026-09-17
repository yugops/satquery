import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import importlib
from core.config import SatQueryConfig
from core.models import ComputationMode

# 1. Flood agent
flood_mod = importlib.import_module('agents.09_flood_inundation.agent')
flood_agent = flood_mod.FloodAndInundationAgent(SatQueryConfig())
resp_flood = asyncio.run(flood_agent.execute(water_mask=None))
print(f"Flood: summary='{resp_flood.summary}', conf={resp_flood.confidence}, mode={resp_flood.computation_mode.value}")
assert resp_flood.confidence == 0.0
assert "No water/SAR mask provided" in resp_flood.summary

# 2. Cyclone agent
cyclone_mod = importlib.import_module('agents.10_cyclone_coastal_risk.agent')
cyclone_agent = cyclone_mod.CycloneAndCoastalRiskAgent(SatQueryConfig())
resp_cyclone = asyncio.run(cyclone_agent.execute(surge_mask=None))
print(f"Cyclone: summary='{resp_cyclone.summary}', conf={resp_cyclone.confidence}, mode={resp_cyclone.computation_mode.value}")
assert resp_cyclone.confidence == 0.0
assert "No storm surge mask provided" in resp_cyclone.summary

# 3. GIS agent
gis_mod = importlib.import_module('agents.13_gis_spatial_measurement.agent')
gis_agent = gis_mod.GISSpatialMeasurementAgent(SatQueryConfig())
resp_gis = asyncio.run(gis_agent.execute(mask=None))
print(f"GIS: summary='{resp_gis.summary}', conf={resp_gis.confidence}, mode={resp_gis.computation_mode.value}")
assert resp_gis.confidence == 0.0
assert "No mask/geometry provided" in resp_gis.summary

print("VERIFICATION_SUCCESS")
