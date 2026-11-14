import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib
tools_mod = importlib.import_module('agents.02_geo_validator.tools')

res = tools_mod.tool_reverse_geocode_gps(48.8566, 2.3522)
print(f"res for (48.8566, 2.3522): {res}")

assert res.get("country") == "FR"
assert "paris" in str(res.get("district", "")).lower() or "paris" in str(res.get("state", "")).lower()
assert res.get("country") != "Global"

# Also test Indian coordinate (Wayanad, Kerala: 11.6854, 76.1320)
res_in = tools_mod.tool_reverse_geocode_gps(11.6854, 76.1320)
print(f"res for (11.6854, 76.1320): {res_in}")
assert res_in.get("country") == "IN"
assert res_in.get("state") == "Kerala"
assert res_in.get("district") == "Wayanad"

print("VERIFICATION_SUCCESS")
