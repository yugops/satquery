"""
SatQuery AI — Agent 02: GeoValidatorAgent Tools
Problem Statement ID: 26167 (ISRO / SAC)

FIX LOG (this pass):
- BUG FOUND: tool_reverse_geocode_gps / tool_reverse_geocode_footprint tried
  the hardcoded `district_resolver` (60-district India-only lookup) FIRST,
  and only fell through to the real `reverse_geocoder` offline library on an
  *exception*. But district_resolver.resolve_coordinates() never raises — for
  any point outside its ~60 hardcoded districts (any non-India location, or
  an India location just outside the list), it silently returned a generic
  "Coord (x,y)" / "Global" fallback instead of ever reaching the real
  geocoder. This meant the real reverse_geocoder integration from an earlier
  fix batch was effectively dead code for anything not in the hardcoded list
  — a real problem given RSVQA/VRSBench imagery is not India-only.

- FIX: reordered to a real fallback chain, each stage genuinely attempted
  before falling to the next, with the hardcoded regional list demoted to
  true last-resort (only reached if both real geocoding paths fail):

    1. Nominatim (OpenStreetMap) reverse geocoding — real, free, global,
       no API key required. Attempted first when network is available.
       Short timeout so a live demo doesn't hang if network drops.
    2. `reverse_geocoder` (offline, pip-installable, real worldwide
       gazetteer — not hardcoded) — used when Nominatim is unreachable
       or times out.
    3. `district_resolver` hardcoded regional bounding boxes — last
       resort only, clearly labeled as such in the response, used only
       if both above genuinely fail (e.g. fully offline + library missing).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
logger = logging.getLogger(__name__)

from core.models import CloudMaskResult, GeoMetadata

# Short timeout so a flaky/absent network connection during a live demo
# fails fast into the offline fallback chain rather than hanging the request.
_NOMINATIM_TIMEOUT_SECONDS = 2.5
_NOMINATIM_USER_AGENT = "SatQueryAI-ISRO-SIH26167/1.0 (contact: team@satquery.ai)"


def tool_extract_geotiff_metadata(file_path: str) -> GeoMetadata:
    """Extracts exact CRS, Tiepoints, PixelScale, GSD, and Reprojected WGS84 Bounds from GeoTIFF."""
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Raster file does not exist: {file_path}")

    # 1. Try rasterio if available (with WGS84 degree reprojection)
    try:
        import rasterio
        import rasterio.warp

        with rasterio.open(file_path) as src:
            crs_str = str(src.crs) if src.crs else None
            gsd = float(abs(src.res[0])) if (src.crs and src.res) else None

            if src.crs:
                if src.crs.to_string() != "EPSG:4326":
                    try:
                        wgs84_bounds = list(rasterio.warp.transform_bounds(src.crs, "EPSG:4326", *src.bounds))
                    except Exception:
                        wgs84_bounds = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]
                else:
                    wgs84_bounds = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]

                wgs84_bounds = [round(float(b), 6) for b in wgs84_bounds]
            else:
                wgs84_bounds = None

            return GeoMetadata(
                crs=crs_str,
                bounds=wgs84_bounds,
                width=src.width,
                height=src.height,
                bands=src.count,
                gsd_meters=gsd,
                nodata=src.nodata,
                driver=src.driver,
                transform=list(src.transform)[:6] if (src.crs and src.transform) else None
            )
    except Exception:
        pass

    # 2. Pure Pillow GeoTIFF Tag Extractor fallback
    try:
        with Image.open(file_path) as img:
            w, h = img.size
            bands = len(img.getbands())
            crs_str = None
            bounds_list = None
            gsd = None
            logger.warning(f"No georeference available for {file_path} — reporting as ungeoreferenced, not fabricating coordinates")

            return GeoMetadata(
                crs=crs_str,
                bounds=bounds_list,
                width=w,
                height=h,
                bands=bands,
                gsd_meters=gsd,
                driver="GTiff" if getattr(img, "format", "") == "TIFF" else "Raster"
            )
    except Exception:
        pass

    return GeoMetadata(width=512, height=512, bands=3, crs=None, bounds=None, gsd_meters=None)


_KNOWN_REGIONAL_BBOXES: list[dict[str, Any]] = [
    {"name": "Wayanad", "district": "Wayanad", "state": "Kerala", "country": "IN", "bounds": [75.75, 11.40, 76.45, 11.98]},
    {"name": "Chamoli", "district": "Chamoli", "state": "Uttarakhand", "country": "IN", "bounds": [79.10, 30.10, 80.00, 31.00]},
    {"name": "Barpeta", "district": "Barpeta", "state": "Assam", "country": "IN", "bounds": [90.60, 26.10, 91.30, 26.65]},
    {"name": "Puri", "district": "Puri", "state": "Odisha", "country": "IN", "bounds": [85.50, 19.60, 86.25, 20.15]},
    {"name": "Beed", "district": "Beed", "state": "Maharashtra", "country": "IN", "bounds": [75.30, 18.50, 76.50, 19.35]},
    {"name": "Kerala", "district": "Kerala Sector", "state": "Kerala", "country": "IN", "bounds": [74.50, 8.00, 77.60, 12.90]},
    {"name": "Assam", "district": "Brahmaputra Valley", "state": "Assam", "country": "IN", "bounds": [89.40, 24.00, 96.10, 28.00]},
    {"name": "Uttarakhand", "district": "Garhwal Himalayas", "state": "Uttarakhand", "country": "IN", "bounds": [77.50, 28.80, 81.10, 31.50]},
    {"name": "Delhi", "district": "New Delhi", "state": "Delhi", "country": "IN", "bounds": [76.80, 28.35, 77.40, 28.90]},
    {"name": "Mumbai", "district": "Mumbai Suburban", "state": "Maharashtra", "country": "IN", "bounds": [72.70, 18.80, 73.10, 19.35]},
]


def _offline_hardcoded_last_resort(lat: float, lon: float, context_text: str = "") -> dict[str, Any]:
    """
    TRUE LAST-RESORT fallback. Only reached when both Nominatim (online) and
    the offline reverse_geocoder library have failed. Covers a small hardcoded
    list of India disaster-hotspot regions — NOT a substitute for real geocoding,
    which is why this is now stage 3 of 3, not stage 1.
    """
    ctx = (context_text or "").lower()
    for item in _KNOWN_REGIONAL_BBOXES:
        if item["name"].lower() in ctx or item["district"].lower() in ctx:
            return {
                "country": item["country"],
                "state": item["state"],
                "district": item["district"],
                "source": "hardcoded_last_resort (context match — real geocoding unavailable)",
            }

    for item in _KNOWN_REGIONAL_BBOXES:
        min_lon, min_lat, max_lon, max_lat = item["bounds"]
        if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
            return {
                "country": item["country"],
                "state": item["state"],
                "district": item["district"],
                "source": "hardcoded_last_resort (coordinate match — real geocoding unavailable)",
            }

    return {
        "country": "IN" if (68.0 <= lon <= 97.5 and 6.5 <= lat <= 37.5) else "Global",
        "state": f"Lat {lat:.2f}°",
        "district": f"Coord ({lat:.2f}°, {lon:.2f}°)",
        "source": "coordinate_centroid (all geocoding sources unavailable)",
    }


def _try_nominatim(lat: float, lon: float) -> Optional[dict[str, Any]]:
    """
    Real online reverse geocoding via OpenStreetMap's Nominatim API.
    Free, no API key required. Returns None (not raises) on any failure so
    the caller can fall through cleanly — network absence during a live demo
    should never be a crash.
    """
    try:
        import requests
    except ImportError:
        return None

    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 10, "addressdetails": 1},
            headers={"User-Agent": _NOMINATIM_USER_AGENT},
            timeout=_NOMINATIM_TIMEOUT_SECONDS,
        )
        if resp.status_code != 200:
            return None
        payload = resp.json()
        addr = payload.get("address", {})
        if not addr:
            return None

        district = (
            addr.get("state_district") or addr.get("county")
            or addr.get("city") or addr.get("town")
            or addr.get("suburb") or addr.get("municipality")
        )
        state = addr.get("state") or addr.get("region")
        country_code = (addr.get("country_code") or "").upper()

        if not district and not state:
            return None

        return {
            "country": country_code or None,
            "state": state,
            "district": district or payload.get("display_name", "").split(",")[0],
            "source": "nominatim (real online reverse geocoding)",
        }
    except Exception as e:
        logger.info(f"Nominatim reverse geocode unavailable ({e}); falling back.")
        return None


def _try_offline_reverse_geocoder(lat: float, lon: float) -> Optional[dict[str, Any]]:
    """Real offline worldwide gazetteer lookup (pip: reverse_geocoder). Not hardcoded — actual global dataset."""
    try:
        import reverse_geocoder as rg
        result = rg.search((lat, lon), mode=1)[0]
        district = result.get("admin2") or result.get("name")
        state = result.get("admin1")
        if not district and not state:
            return None
        return {
            "country": result.get("cc"),
            "state": state,
            "district": district,
            "source": "reverse_geocoder (offline, real global gazetteer)",
        }
    except Exception as e:
        logger.info(f"Offline reverse_geocoder unavailable ({e}); falling back.")
        return None


def tool_reverse_geocode_gps(lat: float, lon: float, crs: Optional[str] = None) -> dict[str, Any]:
    """
    Resolves coordinates into a real geographic location.

    Fallback chain (each stage genuinely attempted, not bypassed):
      1. Nominatim (OSM) — real, online, global.
      2. reverse_geocoder — real, offline, global.
      3. Hardcoded regional list — true last resort only.
    """
    if lat is None or lon is None:
        return {"country": None, "state": None, "district": None, "note": "no coordinates available"}

    result = _try_nominatim(lat, lon)
    if result is not None:
        result["aoi_matched"] = False
        return result

    result = _try_offline_reverse_geocoder(lat, lon)
    if result is not None:
        result["aoi_matched"] = False
        return result

    fallback = _offline_hardcoded_last_resort(lat, lon)
    fallback["aoi_matched"] = False
    return fallback


def tool_reverse_geocode_footprint(
    bounds: list[float],
    context_text: str = "",
    crs: Optional[str] = None,
) -> dict[str, Any]:
    """
    Performs multi-point GIS spatial footprint reverse geocoding across the
    raster bounding box, sampling a 3x3 grid and resolving each point through
    the same real geocoding chain as tool_reverse_geocode_gps.
    """
    if not bounds or len(bounds) < 4:
        return {"country": None, "state": None, "district": None, "note": "no bounds available"}

    min_lon, min_lat, max_lon, max_lat = bounds
    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0

    lats = [min_lat, center_lat, max_lat]
    lons = [min_lon, center_lon, max_lon]
    grid_points = [(la, lo) for la in lats for lo in lons]

    sampled_districts: list[str] = []
    sampled_states: list[str] = []
    source_used = None

    for pla, plo in grid_points:
        res = tool_reverse_geocode_gps(pla, plo, crs)
        source_used = source_used or res.get("source")
        d_name = res.get("district")
        s_name = res.get("state")
        if d_name and d_name not in sampled_districts and not str(d_name).startswith("Coord"):
            sampled_districts.append(d_name)
        if s_name and s_name not in sampled_states:
            sampled_states.append(s_name)

    center_res = tool_reverse_geocode_gps(center_lat, center_lon, crs)
    primary_district = center_res.get("district")
    primary_state = center_res.get("state")

    ctx = (context_text or "").lower()
    aoi_matched = False
    for sd in sampled_districts:
        if sd and sd.lower() in ctx:
            primary_district = sd
            aoi_matched = True
            break

    other_districts = [d for d in sampled_districts if d != primary_district]
    display_district = f"{primary_district} / {other_districts[0]}" if other_districts else primary_district

    return {
        "country": center_res.get("country", "IN"),
        "state": primary_state if len(sampled_states) <= 1 else f"{primary_state} / {sampled_states[0]}",
        "district": display_district or "Unresolved Region",
        "spanned_districts": sampled_districts,
        "spanned_states": sampled_states,
        "aoi_matched": aoi_matched,
        "source": center_res.get("source", source_used),
    }


def tool_detect_cloud_contamination(
    rgb_array: np.ndarray,
    threshold_pct: float = 0.30,
    is_dem: bool = False
) -> CloudMaskResult:
    """
    Calculates optical cloud pixel percentage.
    If raster is a Digital Elevation Model (DEM) or non-optical surface, cloud fraction is exactly 0.0.
    """
    if is_dem:
        return CloudMaskResult(
            cloud_fraction=0.0,
            shadow_fraction=0.0,
            usable_optical=True,
            trigger_sar_fallback=False
        )

    arr = np.asarray(rgb_array, dtype=np.float32)
    if arr.ndim == 2:
        return CloudMaskResult(
            cloud_fraction=0.0,
            shadow_fraction=0.0,
            usable_optical=True,
            trigger_sar_fallback=False
        )

    rgb_part = arr[..., :3]
    brightness = rgb_part.mean(axis=-1)
    max_c = rgb_part.max(axis=-1)
    min_c = rgb_part.min(axis=-1)
    sat = np.where(max_c > 0, (max_c - min_c) / (max_c + 1e-6), 0)

    cloud_mask = (brightness > 200.0) & (sat < 0.20)
    cloud_fraction = float(cloud_mask.mean())

    if arr.shape[-1] >= 4:
        nir = arr[..., 3]
        shadow_mask = (nir < 40.0) & (brightness < 40.0)
    else:
        shadow_mask = (brightness < 30.0) & (~cloud_mask)
    shadow_fraction = float(shadow_mask.mean())

    trigger_sar = cloud_fraction >= threshold_pct
    return CloudMaskResult(
        cloud_fraction=round(cloud_fraction, 4),
        shadow_fraction=round(shadow_fraction, 4),
        usable_optical=not trigger_sar,
        trigger_sar_fallback=trigger_sar
    )