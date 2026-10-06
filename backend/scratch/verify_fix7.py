import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import importlib
from PIL import Image
import tempfile
from core.config import SatQueryConfig

agent_mod = importlib.import_module('agents.02_geo_validator.agent')
tools_mod = importlib.import_module('agents.02_geo_validator.tools')

# Create a plain non-georeferenced JPEG
with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
    img = Image.new("RGB", (100, 100), color=(120, 150, 180))
    img.save(f.name, "JPEG")
    jpeg_file = f.name

try:
    meta = tools_mod.tool_extract_geotiff_metadata(jpeg_file)
    print(f"meta.bounds: {meta.bounds}")
    print(f"meta.crs: {meta.crs}")

    agent = agent_mod.GeoValidatorAgent(SatQueryConfig())
    resp = asyncio.run(agent.execute(image_path=jpeg_file))
    print(f"admin location: {resp.data.get('administrative_location')}")
    print(f"summary: {resp.summary}")

    assert meta.bounds is None, f"Expected bounds to be None, got {meta.bounds}"
    assert "kerala" not in resp.summary.lower(), f"Kerala found in summary: {resp.summary}"
    assert "unknown" in resp.summary.lower()
    print("VERIFICATION_SUCCESS")
finally:
    try:
        Path(jpeg_file).unlink()
    except Exception:
        pass
