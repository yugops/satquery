"""
SatQuery AI — Agent 16: DataIngestAgent Live STAC Tools
Problem Statement ID: 26167 (ISRO / SAC)
Live Open STAC Catalog integration (Copernicus / AWS Earth Search / Microsoft Planetary Computer)
"""

from __future__ import annotations

import logging
from typing import Any
import httpx

logger = logging.getLogger("satquery.data_ingest")

STAC_ENDPOINT = "https://earth-search.aws.element84.com/v1/search"


KNOWN_LOCATIONS: dict[str, list[float]] = {
    "wayanad": [75.80, 11.45, 76.40, 11.95],
    "kerala": [74.80, 8.20, 77.40, 12.80],
    "western ghats": [73.50, 8.50, 77.50, 16.50],
    "delhi": [76.85, 28.40, 77.40, 28.90],
    "mumbai": [72.75, 18.85, 73.05, 19.30],
    "uttarakhand": [78.00, 29.50, 81.00, 31.50],
    "chamoli": [79.20, 30.20, 79.80, 30.80],
    "himachal": [75.50, 30.50, 79.00, 33.30],
    "assam": [89.50, 24.00, 96.00, 28.00],
    "bengaluru": [77.40, 12.80, 77.80, 13.15],
    "chennai": [80.10, 12.90, 80.35, 13.25],
    "sikkim": [88.00, 27.05, 88.95, 28.15],
    "ladakh": [75.50, 32.00, 79.50, 36.00],
}

MONTH_MAP: dict[str, str] = {
    "january": "01", "jan": "01",
    "february": "02", "feb": "02",
    "march": "03", "mar": "03",
    "april": "04", "apr": "04",
    "may": "05",
    "june": "06", "jun": "06",
    "july": "07", "jul": "07",
    "august": "08", "aug": "08",
    "september": "09", "sep": "09", "sept": "09",
    "october": "10", "oct": "10",
    "november": "11", "nov": "11",
    "december": "12", "dec": "12",
}


def parse_search_query_parameters(
    query: str,
    default_bbox: list[float] | None = None
) -> tuple[list[float], str, str, str]:
    """Extracts bounding box, ISO-8601 date range, sensor, and friendly location name from natural query."""
    import re
    q_clean = query.replace("[Search:", "").replace("[Search", "").strip("] ").strip().lower()
    
    # 1. Sensor
    sensor = "Sentinel-2"
    if any(k in q_clean for k in ["sar", "radar", "s1", "sentinel-1", "grd", "vv", "vh"]):
        sensor = "Sentinel-1"

    # 2. Location
    loc_name = "Target AOI"
    resolved_bbox = list(default_bbox) if default_bbox else KNOWN_LOCATIONS["wayanad"]
    for name, bbox_coords in KNOWN_LOCATIONS.items():
        if name in q_clean:
            loc_name = name.title()
            resolved_bbox = bbox_coords
            break

    # 3. Year
    year = "2024"
    y_match = re.search(r"202[0-9]", q_clean)
    if y_match:
        year = y_match.group(0)

    # 4. Month / Date Range
    date_range = f"{year}-07-15/{year}-08-05"
    for mname, mnum in MONTH_MAP.items():
        if mname in q_clean:
            date_range = f"{year}-{mnum}-01/{year}-{mnum}-28"
            break

    return resolved_bbox, date_range, sensor, loc_name


def tool_query_bhuvan_catalog(
    aoi_bbox: tuple[float, float, float, float] | list[float] | None,
    date_range: str,
    sensor: str = "Sentinel-2"
) -> list[dict[str, Any]]:
    """Queries live Open STAC catalog for real Sentinel-2 optical or Sentinel-1 SAR imagery."""
    bbox = list(aoi_bbox) if aoi_bbox else [75.8, 11.45, 76.4, 11.95]

    # Parse ISO-8601 interval
    dt = date_range
    if "/" not in dt:
        dt = f"{dt}T00:00:00Z/{dt}T23:59:59Z"
    else:
        parts = dt.split("/")
        start = parts[0] if "T" in parts[0] else f"{parts[0]}T00:00:00Z"
        end = parts[1] if "T" in parts[1] else f"{parts[1]}T23:59:59Z"
        dt = f"{start}/{end}"

    # Determine collection
    s_clean = sensor.lower().replace("-", "").replace("_", "")
    if any(k in s_clean for k in ["sar", "radar", "s1", "sentinel1", "grd", "vv", "vh"]) or "1" in sensor:
        collection = "sentinel-1-grd"
    else:
        collection = "sentinel-2-l2a"

    payload = {
        "collections": [collection],
        "bbox": bbox,
        "datetime": dt,
        "limit": 6
    }

    try:
        with httpx.Client(timeout=12.0) as client:
            resp = client.post(STAC_ENDPOINT, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                features = data.get("features", [])
                results = []
                for feat in features:
                    props = feat.get("properties", {})
                    assets = feat.get("assets", {})
                    cloud_cover = props.get("eo:cloud_cover")
                    dt_str = props.get("datetime", "")

                    # Extract primary preview or raster asset
                    download_url = None
                    for key in ["visual", "rendered_preview", "thumbnail", "overview", "data"]:
                        if key in assets and "href" in assets[key]:
                            download_url = assets[key]["href"]
                            break

                    results.append({
                        "scene_id": feat.get("id"),
                        "satellite": collection,
                        "acquisition_date": dt_str,
                        "cloud_cover_pct": round(float(cloud_cover), 2) if cloud_cover is not None else None,
                        "bbox_wgs84": feat.get("bbox", bbox),
                        "platform": props.get("platform", "Sentinel"),
                        "constellation": props.get("constellation", "Copernicus"),
                        "assets_available": list(assets.keys()),
                        "download_url": download_url or f"https://earth-search.aws.element84.com/v1/collections/{collection}/items/{feat.get('id')}",
                        "status": "live_stac_verified"
                    })
                return results
    except Exception as e:
        logger.warning(f"Live STAC query failed ({e}). Returning empty result set.")

    return []
