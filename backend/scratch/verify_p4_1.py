import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib
report_tools = importlib.import_module('agents.15_report_generator.tools')

# 1. Test synthetic/unverified session
sitrep_unverified = report_tools.tool_generate_markdown_sitrep({
    "session_id": "TEST-UNVERIFIED",
    "computation_mode": "synthetic_fallback",
    "court_admissible": False,
    "merkle_root": "0x123456789abcdef",
})
print("=== UNVERIFIED SITREP OUTPUT ===")
print(sitrep_unverified)

assert "UNVERIFIED / INCOMPLETE" in sitrep_unverified
assert "CERTIFIED UNTAMPERED — real inference" not in sitrep_unverified
assert "does NOT certify scientific validity" in sitrep_unverified

# 2. Test verified court-admissible session
sitrep_verified = report_tools.tool_generate_markdown_sitrep({
    "session_id": "TEST-VERIFIED",
    "computation_mode": "real_inference",
    "court_admissible": True,
    "merkle_root": "0x987654321fedcba",
})
print("\n=== VERIFIED SITREP OUTPUT ===")
print(sitrep_verified)

assert "CERTIFIED UNTAMPERED" in sitrep_verified
assert "UNVERIFIED / INCOMPLETE" not in sitrep_verified

print("VERIFICATION_SUCCESS")
