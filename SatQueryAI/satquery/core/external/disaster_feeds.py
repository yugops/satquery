"""
SatQuery AI — Kanari Disaster & Wildfire Feed Client

Fetches recent active wildfire and natural disaster events from Kanari API
for spatial correlation against satellite scene boundaries.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

from core.config import SatQueryConfig, config as global_config
from core.exceptions import DisasterFeedError
from core.models import DisasterEvent

logger = logging.getLogger(__name__)


def get_recent_events(
    hours: int = 24,
    api_key: str = "",
    config: SatQueryConfig | None = None,
    timeout: float = 10.0,
) -> list[DisasterEvent]:
    """Fetch active disaster and wildfire events from Kanari within recent hours.

    Args:
        hours: Time window in hours (default 24).
        api_key: Optional Kanari API key override (defaults to config.kanari_api_key).
        config: Optional SatQueryConfig.
        timeout: Network timeout in seconds.

    Returns:
        List of DisasterEvent instances (empty list if feed unavailable or no events).
    """
    cfg = config or global_config
    token = api_key or cfg.kanari_api_key

    # Handle mock mode
    if cfg.mode == "mock":
        return _mock_disaster_events()

    if not token:
        logger.warning("Kanari API key missing. Returning empty disaster events.")
        return []

    url = f"{cfg.kanari_api_endpoint}?hours={hours}"

    try:
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "X-API-Key": token,
                "User-Agent": "SatQueryAI/1.0",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status != 200:
                logger.warning("Kanari API returned status %d", response.status)
                return []
            payload = json.loads(response.read().decode("utf-8"))

        events_data = payload.get("events", payload if isinstance(payload, list) else [])
        events: list[DisasterEvent] = []

        for item in events_data:
            if not isinstance(item, dict):
                continue
            ev_id = str(item.get("id", item.get("event_id", "unknown")))

            # Extract coordinates: check centroid [lon, lat] first, then latitude/longitude
            centroid = item.get("centroid", [])
            if isinstance(centroid, list) and len(centroid) >= 2:
                lon = float(centroid[0])
                lat = float(centroid[1])
            else:
                lat = float(item.get("latitude", item.get("lat", 0.0)))
                lon = float(item.get("longitude", item.get("lon", item.get("lng", 0.0))))

            # Extract title / place
            social = item.get("social", {})
            place = social.get("place") if isinstance(social, dict) else None
            default_title = f"Wildfire near {place}" if place else f"Active Thermal Event ({ev_id})"
            title = str(item.get("title", item.get("name", default_title)))

            category = str(item.get("category", item.get("type", "wildfire")))
            timestamp = item.get("firstSeen", item.get("timestamp", item.get("date", item.get("lastSeen"))))
            severity = item.get("severity", item.get("intensity", item.get("maxFrp")))

            # Normalize confidence (handles floats, strings like 'corrobore', 'h', 'high')
            conf_raw = item.get("confidence", item.get("maxConf", 1.0))
            if isinstance(conf_raw, (int, float)):
                confidence = float(conf_raw)
            elif isinstance(conf_raw, str):
                lower_c = conf_raw.lower()
                if "corrobor" in lower_c or "high" in lower_c or lower_c == "h":
                    confidence = 0.95
                elif "med" in lower_c or lower_c == "m":
                    confidence = 0.80
                else:
                    confidence = 0.70
            else:
                confidence = 1.0

            events.append(
                DisasterEvent(
                    event_id=ev_id,
                    title=title,
                    category=category,
                    latitude=lat,
                    longitude=lon,
                    timestamp=str(timestamp) if timestamp else None,
                    severity=str(severity) if severity else None,
                    confidence=confidence,
                    metadata=item,
                )
            )

        return events
    except Exception as exc:
        logger.warning("Failed to fetch Kanari disaster feed: %s", exc)
        return []


def _mock_disaster_events() -> list[DisasterEvent]:
    """Generate synthetic disaster events for mock testing."""
    return [
        DisasterEvent(
            event_id="fire-california-01",
            title="Northern Ridge Wildfire",
            category="wildfire",
            latitude=37.7749,
            longitude=-122.4194,
            timestamp="2026-08-27T10:00:00Z",
            severity="high",
            confidence=0.95,
        ),
        DisasterEvent(
            event_id="fire-mumbai-02",
            title="Industrial Perimeter Thermal Anomaly",
            category="wildfire",
            latitude=19.0760,
            longitude=72.8777,
            timestamp="2026-08-27T11:30:00Z",
            severity="medium",
            confidence=0.88,
        ),
        DisasterEvent(
            event_id="flood-texas-03",
            title="Coastal Flash Flood Event",
            category="flood",
            latitude=29.7604,
            longitude=-95.3698,
            timestamp="2026-08-27T06:00:00Z",
            severity="severe",
            confidence=0.92,
        ),
    ]
