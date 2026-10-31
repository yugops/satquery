"""
SatQuery AI — Agent 15: SituationReportGeneratorAgent Tools
Problem Statement ID: 26167 (ISRO / SAC)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def tool_generate_markdown_sitrep(data: dict[str, Any]) -> str:
    """Generates official ISRO / NDRF Situation Assessment Report in Markdown."""
    session_id = data.get("session_id", "SQ-000")
    timestamp = data.get("timestamp", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
    merkle_root = data.get("merkle_root", "NOT_AVAILABLE")
    query = data.get("query", "Satellite Multimodal Assessment")
    primary_answer = data.get("primary_answer", "Analysis completed.")
    comp_mode = data.get("computation_mode", "unknown")
    if hasattr(comp_mode, "value"):
        comp_mode = comp_mode.value
    court_admissible = data.get("court_admissible", False)
    merkle_available = merkle_root not in (None, "NOT_AVAILABLE", "")

    if court_admissible and comp_mode == "real_inference" and merkle_available:
        provenance_section = f"""## \U0001f512 Cryptographic Provenance & Court-Admissible Proof
* **Merkle Root:** `{merkle_root}`
* **Verification Status:** CERTIFIED UNTAMPERED — real inference, complete provenance verified."""
    else:
        merkle_line = (
            f"`{merkle_root}` (hash chain intact, but does NOT certify scientific validity)"
            if merkle_available else
            "**NOT AVAILABLE** — no ledger entry was produced for this session"
        )
        provenance_section = f"""## \u26a0\ufe0f Audit Status: UNVERIFIED / INCOMPLETE
This session used synthetic fallback, rule-based heuristics, or incomplete inputs (computation_mode: `{comp_mode}`).
* **Merkle Root:** {merkle_line}
* Cryptographic hashing certifies process integrity only — it does NOT certify inputs were real or the conclusion is correct. Not court-admissible or field-verified."""

    return f"""# \U0001f1ee\U0001f1f3 ISRO / NDRF SATELLITE SITUATION ASSESSMENT REPORT
**Document ID:** {session_id}  
**Classification:** OFFICIAL SATELLITE INTELLIGENCE (PS 26167)  
**Timestamp:** {timestamp}  
**Cryptographic Merkle Root:** `{merkle_root if merkle_available else "NOT AVAILABLE"}`  
**Certified Computation Mode:** `{comp_mode}`  

---

## \U0001f3af Executive Intelligence Brief
* **Mission Query:** {query}
* **Synthesized Assessment:** {primary_answer}

---

## \U0001f4ca Analytical Trace & Agent Findings
{chr(10).join([f"- **{t.get('agent_name', 'Agent')}**: {t.get('summary', '')}" for t in data.get('agent_traces', [])])}

---

{provenance_section}
"""


def tool_render_pdf_html(markdown_content: str, merkle_root: str) -> str:
    """Renders printable ISRO-styled HTML / PDF template with scannable QR placeholder."""
    merkle_available = bool(merkle_root) and merkle_root != "NOT_AVAILABLE"
    merkle_display = merkle_root if merkle_available else "NOT AVAILABLE — no ledger entry for this session"
    verify_line = (
        f"<em>Scan or verify at /api/report/verify?root={merkle_root}</em>"
        if merkle_available else
        "<em>No verification link — cryptographic ledger entry was not produced.</em>"
    )
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SatQuery AI Situation Report</title>
<style>
  body {{ font-family: 'Inter', -apple-system, sans-serif; margin: 40px; color: #1e293b; line-height: 1.6; }}
  .header {{ border-bottom: 3px solid #0284c7; padding-bottom: 12px; margin-bottom: 24px; }}
  .badge {{ display: inline-block; background: #e0f2fe; color: #0369a1; padding: 4px 10px; border-radius: 4px; font-weight: 600; font-size: 12px; }}
  .merkle {{ background: #f8fafc; border: 1px dashed #cbd5e1; padding: 12px; border-radius: 6px; font-family: monospace; font-size: 12px; word-break: break-all; }}
</style>
</head>
<body>
  <div class="header">
    <span class="badge">ISRO / SAC PS 26167</span>
    <h2>\U0001f6f0\ufe0f SatQuery AI — Official Situation Assessment Report</h2>
  </div>
  <div>{markdown_content.replace(chr(10), '<br>')}</div>
  <div class="merkle">
    <strong>Cryptographic Merkle Proof:</strong> {merkle_display}<br>
    {verify_line}
  </div>
</body>
</html>"""