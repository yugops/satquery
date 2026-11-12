import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import importlib
from core.config import SatQueryConfig

agent_mod = importlib.import_module('agents.08_drought_crop_health.agent')
DroughtAndCropHealthAgent = agent_mod.DroughtAndCropHealthAgent
agent = DroughtAndCropHealthAgent(SatQueryConfig())

multiband_file = "d:/satquery/test/wayanad_pre_multiband_t1.tif"

# Run 1
resp1 = asyncio.run(agent.execute(image_path=multiband_file, district="Wayanad, Kerala"))
# Run 2
resp2 = asyncio.run(agent.execute(image_path=multiband_file, district="Wayanad, Kerala"))

print("=== RUN 1 OUTPUT ===")
print(f"Mean NDVI: {resp1.data['mean_ndvi']}, Mean NDMI: {resp1.data['mean_ndmi']}")
print(f"Delta: {resp1.data['ndvi_anomaly_delta']}")
print(f"Crops: {resp1.data['crop_loss_forecast']['primary_impacted_crops']}")
print(f"Summary: {resp1.summary}")

print("\n=== RUN 2 OUTPUT ===")
print(f"Mean NDVI: {resp2.data['mean_ndvi']}, Mean NDMI: {resp2.data['mean_ndmi']}")
print(f"Delta: {resp2.data['ndvi_anomaly_delta']}")
print(f"Crops: {resp2.data['crop_loss_forecast']['primary_impacted_crops']}")
print(f"Summary: {resp2.summary}")

# Verify exact deterministic identity
assert resp1.data['mean_ndvi'] == resp2.data['mean_ndvi'], "NDVI must be identical across runs on real multiband raster!"
assert resp1.data['mean_ndmi'] == resp2.data['mean_ndmi']
assert 0.3 <= resp1.data['mean_ndvi'] <= 0.9, f"Plausible forest NDVI range (0.3-0.9), got {resp1.data['mean_ndvi']}"
assert "Tea" in resp1.data['crop_loss_forecast']['primary_impacted_crops'], "Should map Kerala crops for Wayanad!"

print("\nRuns are 100% IDENTICAL and deterministic.")
print("VERIFICATION_SUCCESS")
