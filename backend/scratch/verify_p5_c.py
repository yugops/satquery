import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import importlib
import numpy as np
from core.config import SatQueryConfig

evac_mod = importlib.import_module('agents.18_evac_routing.agent')
EvacRoutingAgent = evac_mod.EvacRoutingAgent
agent = EvacRoutingAgent(SatQueryConfig())

# Test Pair 1: (2, 3) to (18, 14) with an obstacle blocking the direct diagonal
hazard_mask1 = np.zeros((20, 20), dtype=np.uint8)
hazard_mask1[5:15, 8:12] = 1 # Block center

resp1 = asyncio.run(agent.execute(start=(2, 3), goal=(18, 14), hazard_mask=hazard_mask1))
route1 = resp1.data['evacuation_plan']['route_coordinates']
print("=== ROUTE 1 ===")
print(f"Start: {resp1.data['start']}, Goal: {resp1.data['goal']}")
print(f"Route Start: {route1[0]}, Route End: {route1[-1]}")
print(f"Route Length: {len(route1)}, Safe Found: {resp1.data['evacuation_plan']['safe_route_found']}")
print(f"Hazards Bypassed: {resp1.data['evacuation_plan']['hazards_bypassed']}")

# Ensure no hazard cell is traversed
for node in route1:
    assert hazard_mask1[node[0], node[1]] == 0, f"Route crossed hazard cell at {node}!"

# Test Pair 2: (0, 15) to (12, 1) with different obstacle
hazard_mask2 = np.zeros((20, 20), dtype=np.uint8)
hazard_mask2[3:8, 5:10] = 1

resp2 = asyncio.run(agent.execute(start=(0, 15), goal=(12, 1), hazard_mask=hazard_mask2))
route2 = resp2.data['evacuation_plan']['route_coordinates']
print("\n=== ROUTE 2 ===")
print(f"Start: {resp2.data['start']}, Goal: {resp2.data['goal']}")
print(f"Route Start: {route2[0]}, Route End: {route2[-1]}")
print(f"Route Length: {len(route2)}, Safe Found: {resp2.data['evacuation_plan']['safe_route_found']}")

assert route1[0] == (2, 3) and route1[-1] == (18, 14), "Route 1 must start/end at custom endpoints"
assert route2[0] == (0, 15) and route2[-1] == (12, 1), "Route 2 must start/end at custom endpoints"
assert route1 != route2, "Different endpoints and hazard configurations must yield distinct routes"

print("\nVERIFICATION_SUCCESS")
