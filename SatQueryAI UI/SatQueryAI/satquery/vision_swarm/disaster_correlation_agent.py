"""
SatQuery AI — Agent 8: DisasterCorrelationAgent

Wing: Disaster_Intelligence

Correlates satellite scene geospatial extents (bounding boxes) against
real-time active disaster feeds (wildfires, thermal anomalies, flooding)
from the Kanari API. Part 2 extension stub.
"""

from __future__ import annotations

from typing import Any

from core.config import SatQueryConfig
from core.external.disaster_feeds import get_recent_events
from core.interfaces import BaseAgent
from core.models import DisasterCorrelationResult, DisasterEvent, GeoLocation, OutputType


class DisasterCorrelationAgent(BaseAgent):
    """Agent 8 — Correlates scene boundaries against active disaster events.

    Invoked when query intent is classified as DISASTER_CORRELATION
    (e.g., questions regarding active wildfires, fire perimeters, or thermal anomalies).
    """

    agent_name = "DisasterCorrelationAgent"
    mempalace_wing = "Wing: Disaster_Intelligence"

    def __init__(self, config: SatQueryConfig | None = None):
        super().__init__(config)

    async def _execute(self, **kwargs: Any) -> DisasterCorrelationResult:
        """Correlate scene geospatial bounds with active disaster feed events.

        Expected kwargs:
            bounding_box: tuple[float, float, float, float] as (min_lon, min_lat, max_lon, max_lat).
            location: Optional GeoLocation instance.
            hours: Optional lookback window in hours (default 24).

        Returns:
            DisasterCorrelationResult with matched events and threat flag.
        """
        hours = kwargs.get("hours", 24)
        bbox = kwargs.get("bounding_box")
        location: GeoLocation | None = kwargs.get("location")

        if bbox is None and location is not None and location.bounding_box is not None:
            bbox = location.bounding_box

        events = get_recent_events(hours=hours, config=self.config)

        if not bbox:
            # If no bbox provided, check if center coordinate provided
            if location is not None:
                lat, lon = location.latitude, location.longitude
                # Match within roughly ~0.5 degree (~50km) radius
                matched = [
                    ev for ev in events
                    if abs(ev.latitude - lat) <= 0.5 and abs(ev.longitude - lon) <= 0.5
                ]
            else:
                matched = []
        else:
            min_lon, min_lat, max_lon, max_lat = bbox
            matched = [
                ev for ev in events
                if (min_lat <= ev.latitude <= max_lat) and (min_lon <= ev.longitude <= max_lon)
            ]

        is_threat = len(matched) > 0
        summary = (
            f"Found {len(matched)} active disaster/wildfire event(s) overlapping the scene extent."
            if is_threat
            else "No active disaster or wildfire events detected within the scene bounding box."
        )

        out_type = OutputType.MOCK if self.config.mode == "mock" else OutputType.RULE_BASED

        return DisasterCorrelationResult(
            matched_events=matched,
            total_events_checked=len(events),
            is_threat_detected=is_threat,
            summary=summary,
            output_type=out_type,
        )
