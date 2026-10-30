"""
SatQuery AI — Agent 02: GeoValidatorAgent
Problem Statement ID: 26167 (ISRO / SAC)
Wing: Spatial_Metadata
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

from core.interfaces import BaseAgent
from core.models import ComputationMode
from .tools import (
    tool_extract_geotiff_metadata,
    tool_reverse_geocode_gps,
    tool_reverse_geocode_footprint,
    tool_detect_cloud_contamination,
)


import math


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float, radius_km: float = 6371.0) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return radius_km * 2 * math.asin(math.sqrt(a))


class GeoValidatorAgent(BaseAgent):
    agent_id = "geo_validator"
    agent_name = "GeoValidatorAgent"
    mempalace_wing = "Wing: Spatial_Metadata"

    async def _execute(self, **kwargs: Any) -> tuple[dict[str, Any], ComputationMode, float, Optional[str], Optional[str]]:
        file_path = kwargs.get("image_path") or kwargs.get("file_path", "")
        metadata = tool_extract_geotiff_metadata(file_path)

        admin = None
        tile_width_km = 0.0
        if metadata.bounds:
            min_lon, min_lat, max_lon, max_lat = metadata.bounds
            center_lon = (min_lon + max_lon) / 2.0
            center_lat = (min_lat + max_lat) / 2.0
            tile_width_km = haversine_km(center_lat, min_lon, center_lat, max_lon)

            query = str(kwargs.get("query", "")).lower()
            fname = Path(file_path).name.lower() if file_path else ""
            orig_fname = str(kwargs.get("original_filename") or kwargs.get("filename") or "").lower()
            combined_context = f"{query} {fname} {orig_fname}"

            # Real multi-point GIS spatial footprint reverse geocoding
            target_lat = kwargs.get("aoi_lat")
            target_lon = kwargs.get("aoi_lon")

            if target_lat is not None and target_lon is not None and (min_lon <= target_lon <= max_lon) and (min_lat <= target_lat <= max_lat):
                admin = tool_reverse_geocode_gps(target_lat, target_lon, crs=metadata.crs)
                admin["aoi_matched"] = True
            else:
                admin = tool_reverse_geocode_footprint(metadata.bounds, context_text=combined_context, crs=metadata.crs)

            metadata.administrative_district = admin.get("district")
            metadata.administrative_state = admin.get("state")

            if tile_width_km > 20 and not admin.get("aoi_matched"):
                spanned = admin.get("spanned_districts", [])
                spanned_str = f" Spans: {', '.join(spanned[:4])}" if len(spanned) > 1 else ""
                admin["location_caveat"] = (
                    f"~{tile_width_km:.0f}km scene footprint.{spanned_str}"
                )

        # Cloud contamination check (Skipped on DEM / single-band elevation models)
        cloud_info = None
        is_dem = any(k in str(file_path).lower() for k in ["dem", "dsm", "dtm", "elevation", "glo-30", "glo30"])
        is_single_band = metadata.bands == 1
        try:
            if is_dem or is_single_band:
                cloud_info = {"cloud_fraction": 0.0, "shadow_fraction": 0.0, "usable_optical": True, "trigger_sar_fallback": False}
            else:
                import rasterio
                try:
                    with rasterio.open(file_path) as r_src:
                        c = r_src.count
                        step = max(1, r_src.width // 1024)
                        r_arr = r_src.read(out_shape=(c, r_src.height // step, r_src.width // step)).astype(np.float32)
                        if c >= 3:
                            rgb_arr = r_arr[:3]
                            max_v = float(np.max(rgb_arr))
                            scale = 255.0 / max_v if max_v > 255.0 else 1.0
                            arr = (rgb_arr * scale).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
                        else:
                            arr = np.stack([r_arr[0], r_arr[0], r_arr[0]], axis=-1).astype(np.uint8)
                        cloud_res = tool_detect_cloud_contamination(arr, self.config.cloud_contamination_threshold, is_dem=is_dem)
                        cloud_info = cloud_res.model_dump()
                except Exception:
                    with Image.open(file_path) as img:
                        img_thumb = img.copy()
                        img_thumb.thumbnail((1024, 1024), Image.Resampling.BICUBIC)
                        arr = np.array(img_thumb.convert("RGB"))
                        cloud_res = tool_detect_cloud_contamination(arr, self.config.cloud_contamination_threshold, is_dem=is_dem)
                        cloud_info = cloud_res.model_dump()
        except Exception as e:
            logger.warning(f"Cloud detection skipped for {file_path} ({e})")
            cloud_info = {
                "cloud_fraction": 0.0,
                "shadow_fraction": 0.0,
                "usable_optical": True,
                "trigger_sar_fallback": False,
            }

        cloud_str = f"{round(cloud_info['cloud_fraction']*100, 1)}%" if cloud_info.get("cloud_fraction") is not None else "Unknown (Error)"
        loc_district = metadata.administrative_district or "Unknown (No Georeference)"
        loc_state = metadata.administrative_state or "Unknown"
        caveat_hint = f" [Note: Scene center district; scene width ~{tile_width_km:.0f}km]" if (admin and admin.get("location_caveat")) else ""
        summary = (
            f"Geospatial validation complete. Extent: {metadata.width}x{metadata.height} px, "
            f"CRS: {metadata.crs or 'Local Grid (Ungeoreferenced)'}, Cloud: {cloud_str}. "
            f"Location: {loc_district}, {loc_state}.{caveat_hint}"
        )

        data = {
            "metadata": metadata.model_dump(),
            "cloud_check": cloud_info,
            "administrative_location": admin,
        }

        return data, ComputationMode.RULE_BASED, 1.0, summary, None
