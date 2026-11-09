import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib
import numpy as np

tools_mod = importlib.import_module('agents.06_bitemporal_change.tools')

# 1000x1000 vs 300x400
arr1 = np.ones((3, 1000, 1000), dtype=np.float32)
arr2 = np.ones((3, 300, 400), dtype=np.float32)

res = tools_mod.tool_compute_spectral_change(arr1, arr2, gsd_meters=10.0)
print(f"extent_mismatch_warning: {res.get('extent_mismatch_warning')}")

assert res.get("extent_mismatch_warning") is not None, "Warning must be populated"
assert "different extents" in res.get("extent_mismatch_warning")

print("VERIFICATION_SUCCESS")
