"""
SatQuery AI — Satellite Orbital Metadata (TLE) Client

Fetches Two-Line Element (TLE) orbital sets from the Ivan Stanojevic TLE API
for satellite tracking and orbit confirmation.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

from core.config import SatQueryConfig, config as global_config
from core.exceptions import SatelliteMetadataError
from core.models import OutputType, TLEResult

logger = logging.getLogger(__name__)


def get_tle(
    norad_id: int,
    config: SatQueryConfig | None = None,
    timeout: float = 10.0,
) -> TLEResult | None:
    """Fetch Two-Line Element (TLE) satellite orbital data by NORAD Catalog ID.

    Args:
        norad_id: NORAD satellite catalog ID (e.g. 25544 for ISS, 60133).
        config: Optional SatQueryConfig.
        timeout: HTTP request timeout in seconds.

    Returns:
        TLEResult with orbital lines and epoch, or None if unavailable.
    """
    if not isinstance(norad_id, int) or norad_id <= 0:
        logger.warning("Invalid NORAD ID provided: %s", norad_id)
        return None

    cfg = config or global_config

    # Handle mock mode
    if cfg.mode == "mock":
        return TLEResult(
            norad_id=norad_id,
            name=f"SATELLITE-{norad_id}",
            line1=f"1 {norad_id:05d}U 20001A   26239.50000000  .00001000  00000+0  10000-4 0  9991",
            line2=f"2 {norad_id:05d}  98.1234 120.4567 0001234  80.1234 280.5678 14.80000000 12345",
            epoch="2026-08-27T00:00:00Z",
            output_type=OutputType.MOCK,
        )

    endpoint = cfg.tle_api_endpoint.rstrip("/")
    url = f"{endpoint}/{norad_id}"

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "SatQueryAI/1.0",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status != 200:
                logger.warning("TLE API returned status %d for NORAD ID %d", response.status, norad_id)
                return None
            payload = json.loads(response.read().decode("utf-8"))

        name = payload.get("name", f"NORAD-{norad_id}")
        line1 = payload.get("line1", "")
        line2 = payload.get("line2", "")
        epoch = payload.get("date", payload.get("epoch"))

        if not line1 or not line2:
            logger.warning("Incomplete TLE lines returned for NORAD ID %d", norad_id)
            return None

        return TLEResult(
            norad_id=int(payload.get("satelliteId", norad_id)),
            name=name,
            line1=line1,
            line2=line2,
            epoch=str(epoch) if epoch else None,
            output_type=OutputType.RULE_BASED,
        )
    except Exception as exc:
        logger.warning("Failed to fetch TLE for NORAD ID %d: %s", norad_id, exc)
        return None
