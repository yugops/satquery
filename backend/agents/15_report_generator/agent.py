"""
SatQuery AI — Agent 15: SituationReportGeneratorAgent
Problem Statement ID: 26167 (ISRO / SAC)
Wing: Published_Reports
"""

from __future__ import annotations

from typing import Any, Optional
from core.interfaces import BaseAgent
from core.models import ComputationMode
from .tools import tool_generate_markdown_sitrep, tool_render_pdf_html


class SituationReportGeneratorAgent(BaseAgent):
    agent_id = "report_generator"
    agent_name = "SituationReportGeneratorAgent"
    mempalace_wing = "Wing: Published_Reports"

    async def _execute(self, **kwargs: Any) -> tuple[dict[str, Any], ComputationMode, float, Optional[str], Optional[str]]:
        report_data = kwargs.get("final_response_data", {})
        merkle_root = report_data.get("merkle_root", "0x0000000000000000")

        md_sitrep = tool_generate_markdown_sitrep(report_data)
        html_report = tool_render_pdf_html(md_sitrep, merkle_root)

        summary = f"Official ISRO / NDRF Situation Report compiled with embedded Merkle root `{merkle_root[:16]}...`."

        data = {
            "markdown_report": md_sitrep,
            "html_report": html_report,
            "merkle_root": merkle_root,
            "pdf_export_ready": True,
        }

        return data, ComputationMode.RULE_BASED, 1.0, summary, None
