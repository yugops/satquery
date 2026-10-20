import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import importlib
from core.config import SatQueryConfig

# 1. Verify HistoricalMemoryAgent P4-2
hist_mod = importlib.import_module('agents.14_historical_memory.agent')
hist_agent = hist_mod.HistoricalMemoryAgent(SatQueryConfig())

resp_unmatched = asyncio.run(hist_agent.execute(district="Kanyakumari"))
print("=== HISTORICAL MEMORY (UNMATCHED) ===")
print(f"Confidence: {resp_unmatched.confidence}, Summary: {resp_unmatched.summary}")
print(f"Matched count: {resp_unmatched.data['matched_count']}")
assert resp_unmatched.confidence == 0.0
assert resp_unmatched.data['matched_count'] == 0
assert "No historical disaster precedent found" in resp_unmatched.summary

resp_matched = asyncio.run(hist_agent.execute(district="Wayanad"))
print("\n=== HISTORICAL MEMORY (MATCHED) ===")
print(f"Confidence: {resp_matched.confidence}, Summary: {resp_matched.summary}")
assert resp_matched.confidence == 0.95
assert resp_matched.data['matched_count'] > 0

# 2. Verify GIS Haversine P4-4
gis_mod = importlib.import_module('agents.13_gis_spatial_measurement.agent')
gis_agent = gis_mod.GISSpatialMeasurementAgent(SatQueryConfig())

mumbai = (18.9220, 72.8347)
delhi = (28.6139, 77.2090)

resp_gis = asyncio.run(gis_agent.execute(coords_pair=(mumbai, delhi)))
dist_km = resp_gis.data['haversine_distance_km']
print(f"\n=== GIS HAVERSINE DISTANCE (Mumbai -> Delhi) ===")
print(f"Computed Haversine Distance: {dist_km} km")
print(f"Summary: {resp_gis.summary}")

# Great-circle distance between Mumbai and Delhi is ~1160 km
assert 1140.0 <= dist_km <= 1180.0, f"Distance should be ~1160 km, got {dist_km} km"

print("\nVERIFICATION_SUCCESS")
