import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib
tools_mod = importlib.import_module('agents.06_bitemporal_change.tools')

# Pass a non-existent/corrupt file to tool_crop_spatial_aoi
raised = False
try:
    tools_mod.tool_crop_spatial_aoi("corrupt_or_non_existent.tif")
except RuntimeError as e:
    raised = True
    print(f"Raised expected RuntimeError: {e}")

assert raised, "Expected RuntimeError when rasterio cannot read georeference"
print("VERIFICATION_SUCCESS")
