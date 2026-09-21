"""
SatQuery AI — Live Weather & Open-Meteo Integration
Problem Statement ID: 26167 (ISRO / SAC)
"""

from __future__ import annotations

import json
import urllib.request
import urllib.parse
from typing import Any, Optional


def fetch_live_weather(lat: float, lon: float, endpoint: str = "https://api.open-meteo.com/v1/forecast") -> dict[str, Any]:
    """Fetch real-time surface wind speed, wind direction, temperature, and precipitation."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m",
        "timezone": "auto"
    }
    url = f"{endpoint}?{urllib.parse.urlencode(params)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SatQueryAI/1.0"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            current = data.get("current", {})
            return {
                "temperature_c": current.get("temperature_2m", 28.0),
                "humidity_pct": current.get("relative_humidity_2m", 65.0),
                "precipitation_mm": current.get("precipitation", 0.0),
                "wind_speed_kmh": current.get("wind_speed_10m", 12.0),
                "wind_direction_deg": current.get("wind_direction_10m", 180.0),
                "status": "live_api"
            }
    except Exception:
        # Graceful fallback to seasonal baseline
        return {
            "temperature_c": 28.0,
            "humidity_pct": 65.0,
            "precipitation_mm": 0.0,
            "wind_speed_kmh": 12.0,
            "wind_direction_deg": 180.0,
            "status": "fallback_baseline"
        }
