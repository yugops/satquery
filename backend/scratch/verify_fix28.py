import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib
import numpy as np

sar_tools = importlib.import_module('agents.07_sar_cloud_penetration.tools')

assert hasattr(sar_tools, 'tool_apply_lee_filter'), "tool_apply_lee_filter not found"

# Create a noisy synthetic SAR patch (speckle noise on homogeneous area)
np.random.seed(42)
clean = np.full((100, 100), 10.0, dtype=np.float32)
# Multiplicative Rayleigh/gamma speckle
noise = np.random.gamma(shape=1.0, scale=1.0, size=(100, 100)).astype(np.float32)
noisy = clean * noise

filtered = sar_tools.tool_apply_lee_filter(noisy, window_size=7)

var_noisy = float(np.var(noisy))
var_filtered = float(np.var(filtered))

print(f"Variance of noisy image: {var_noisy:.4f}")
print(f"Variance of Lee-filtered image: {var_filtered:.4f}")

# Lee filter must significantly reduce speckle variance
assert var_filtered < var_noisy * 0.5, f"Expected >50% variance reduction, got {var_filtered}/{var_noisy}"

print("VERIFICATION_SUCCESS")
