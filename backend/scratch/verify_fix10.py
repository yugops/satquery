import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib
tools_mod = importlib.import_module('agents.03_audit_ledger.tools')

res = tools_mod.tool_calculate_calibrated_confidence([])
print(f"Empty evidence score confidence: {res}")
assert res == 0.0, f"Expected 0.0, got {res}"
print("VERIFICATION_SUCCESS")
