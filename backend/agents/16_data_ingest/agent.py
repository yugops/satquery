"""
SatQuery AI — Agent 16: DataIngestAgent
Problem Statement ID: 26167 (ISRO / SAC)
Wing: Live_Ingestion
"""

from __future__ import annotations

from typing import Any, Optional
from core.interfaces import BaseAgent
from core.models import ComputationMode
from .tools import tool_query_bhuvan_catalog, parse_search_query_parameters


class DataIngestAgent(BaseAgent):
    agent_id = "data_ingest"
    agent_name = "DataIngestAgent"
    mempalace_wing = "Wing: Live_Ingestion"

    async def _execute(self, **kwargs: Any) -> tuple[dict[str, Any], ComputationMode, float, Optional[str], Optional[str]]:
        query = kwargs.get("query", "")
        passed_bbox = kwargs.get("bbox")

        # Parse query for location, date range, and sensor
        parsed_bbox, parsed_date, parsed_sensor, loc_name = parse_search_query_parameters(query, default_bbox=passed_bbox)

        date_range = kwargs.get("date_range") or parsed_date
        bbox = passed_bbox or parsed_bbox
        sensor = kwargs.get("sensor") or parsed_sensor

        scenes = tool_query_bhuvan_catalog(bbox, date_range, sensor)

        if not scenes:
            summary = f"Live STAC query returned 0 scenes for {sensor} in interval {date_range} over {loc_name}."
            data = {
                "scenes": [],
                "selected_scene": None,
                "total_scenes": 0,
                "location_name": loc_name,
                "date_range": date_range,
                "sensor": sensor,
                "live_harvester_status": "ONLINE_EMPTY_RESULT",
            }
            return data, ComputationMode.API_INFERENCE, 0.98, summary, None

        selected = min(scenes, key=lambda s: s.get("cloud_cover_pct", 100.0) if s.get("cloud_cover_pct") is not None else 100.0)
        cloud_str = f", Cloud: {selected['cloud_cover_pct']}%" if selected.get("cloud_cover_pct") is not None else ""
        summary = (
            f"Live STAC harvester retrieved {len(scenes)} real orbital passes over {loc_name} ({sensor}, {date_range}). "
            f"Primary scene {selected['scene_id']} (Acquired: {selected['acquisition_date']}{cloud_str}) selected."
        )

        data = {
            "scenes": scenes,
            "selected_scene": selected,
            "total_scenes": len(scenes),
            "location_name": loc_name,
            "date_range": date_range,
            "sensor": sensor,
            "bbox_used": bbox,
            "live_harvester_status": "ONLINE",
        }

        return data, ComputationMode.API_INFERENCE, 0.98, summary, None
