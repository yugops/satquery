import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import importlib
import tempfile
from PIL import Image
from core.config import SatQueryConfig
from core.models import ComputationMode

agent_mod = importlib.import_module('agents.07_sar_cloud_penetration.agent')
SARCloudPenetrationAgent = agent_mod.SARCloudPenetrationAgent

# Create a single-band grayscale image
with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
    img = Image.new("L", (100, 100), color=128)
    img.save(f.name, "PNG")
    single_band_file = f.name

try:
    agent = SARCloudPenetrationAgent(SatQueryConfig())
    resp = asyncio.run(agent.execute(vv_tif=single_band_file, vh_tif=""))
    print(f"resp summary: {resp.summary}")
    print(f"computation_mode: {resp.computation_mode.value}")
    print(f"confidence: {resp.confidence}")

    assert "Dual-pol VV+VH input is required" in resp.summary or "could not be read" in resp.summary
    assert resp.confidence == 0.0
    assert resp.computation_mode == ComputationMode.RULE_BASED
    print("VERIFICATION_SUCCESS")
finally:
    try:
        Path(single_band_file).unlink()
    except Exception:
        pass
