import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import importlib
from core.config import SatQueryConfig

voice_mod = importlib.import_module('agents.19_bhashini_voice.agent')
BhashiniVoiceAgent = voice_mod.BhashiniVoiceAgent
agent = BhashiniVoiceAgent(SatQueryConfig())

query_wildfire = "Wildfire advancing toward north ridge"
resp = asyncio.run(agent.execute(sitrep_text=query_wildfire, language="hi"))

translated = resp.data.get("translated_text", "")
print("=== TRANSLATED WILDFIRE TEXT ===")
print(translated)
print(f"Confidence: {resp.confidence}, Mode: {resp.computation_mode.value}")

# Must NOT contain static canned flood / NH-31 terms
assert "बाढ़" not in translated, "Must not contain flood term in a wildfire query!"
assert "राजमार्ग ३१" not in translated, "Must not contain NH-31 in a wildfire query!"
assert "आग" in translated or "Wildfire" in translated or "जंगल" in translated, "Must translate wildfire query!"

# Test empty refusal
resp_empty = asyncio.run(agent.execute(sitrep_text="", query=""))
print(f"\nEmpty text refusal: summary='{resp_empty.summary}', conf={resp_empty.confidence}")
assert resp_empty.confidence == 0.0

print("VERIFICATION_SUCCESS")
