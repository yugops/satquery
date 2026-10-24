"""
SatQuery AI — Open-Meteo Weather Integration Helper

Provides get_weather() for querying meteorological conditions to cross-reference
and validate optical cloud cover and scene conditions.
"""

from __future__ import annotations

import logging
from typing import Any
import urllib.parse
import urllib.request
import json

from core.config import SatQueryConfig, config as global_config
from core.exceptions import WeatherError, WeatherNetworkError, WeatherAPIError
from core.models import OutputType, WeatherContext

logger = logging.getLogger(__name__)


def get_weather(
    lat: float,
    lon: float,
    hourly: list[str] | None = None,
    config: SatQueryConfig | None = None,
) -> WeatherContext | None:
    """Fetch current/hourly weather context from Open-Meteo for validation.

    Args:
        lat: Latitude in decimal degrees (-90 to 90).
        lon: Longitude in decimal degrees (-180 to 180).
        hourly: List of hourly variables (default: ['temperature_2m', 'cloudcover']).
        config: Optional SatQueryConfig.

    Returns:
        WeatherContext with temperature_2m and cloudcover, or None if unavailable.
    """
    cfg = config or global_config
    hourly_vars = hourly or ["temperature_2m", "cloudcover"]

    # Handle mock mode
    if cfg.mode == "mock":
        return WeatherContext(
            temperature_2m=21.5,
            cloudcover=15.0,
            source="open-meteo",
            output_type=OutputType.MOCK,
        )

    # Normalize variable names for Open-Meteo API
    api_hourly = []
    for var in hourly_vars:
        if var == "cloudcover":
            api_hourly.append("cloud_cover")
        else:
            api_hourly.append(var)

    params = {
        "latitude": str(lat),
        "longitude": str(lon),
        "hourly": ",".join(api_hourly),
        "forecast_days": "1",
    }
    url = f"{cfg.openmeteo_api_endpoint}?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "SatQueryAI/1.0", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10.0) as response:
            if response.status != 200:
                logger.warning("Open-Meteo returned status %d", response.status)
                return None
            data = json.loads(response.read().decode("utf-8"))

        hourly_data = data.get("hourly", {})
        temps = hourly_data.get("temperature_2m", [])
        clouds = hourly_data.get("cloud_cover", hourly_data.get("cloudcover", []))

        temp_val = float(temps[0]) if temps else None
        cloud_val = float(clouds[0]) if clouds else None

        return WeatherContext(
            temperature_2m=temp_val,
            cloudcover=cloud_val,
            source="open-meteo",
            output_type=OutputType.RULE_BASED,
        )
    except Exception as exc:
        logger.warning("Failed to fetch Open-Meteo weather context: %s", exc)
        return None
