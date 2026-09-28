import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib
import numpy as np

tools_mod = importlib.import_module('agents.02_geo_validator.tools')

# Scene 1: Medium bright scene (no dark shadow, some clouds)
img1 = np.full((100, 100, 3), 120, dtype=np.float32)
img1[:20, :20] = 240 # cloud
res1 = tools_mod.tool_detect_cloud_contamination(img1)

# Scene 2: Scene with clouds + large dark shadow region (50x50 dark pixels)
img2 = np.full((100, 100, 3), 120, dtype=np.float32)
img2[:20, :20] = 240 # same cloud
img2[50:, 50:] = 10  # dark shadow area (25% of image)
res2 = tools_mod.tool_detect_cloud_contamination(img2)

print(f"Scene 1: cloud={res1.cloud_fraction}, shadow={res1.shadow_fraction}")
print(f"Scene 2: cloud={res2.cloud_fraction}, shadow={res2.shadow_fraction}")

assert res1.cloud_fraction == res2.cloud_fraction, "Clouds should be identical"
assert res1.shadow_fraction != res2.shadow_fraction, "Shadow fractions must differ"
assert res2.shadow_fraction == 0.25, f"Expected 0.25 shadow, got {res2.shadow_fraction}"
assert res1.shadow_fraction != round(res1.cloud_fraction * 0.25, 4) or res2.shadow_fraction != round(res2.cloud_fraction * 0.25, 4)

print("VERIFICATION_SUCCESS")
