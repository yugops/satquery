"""
SatQuery AI — Agent 13: GISSpatialMeasurementAgent Tools
Problem Statement ID: 26167 (ISRO / SAC)
"""

from __future__ import annotations

from typing import Any
import numpy as np


def tool_calculate_pixel_area_sqkm(mask: np.ndarray, gsd_meters: float = 10.0) -> float:
    """Calculates exact real-world surface area in sq. km from Ground Sampling Distance."""
    positive_pixels = int(np.sum(mask > 0))
    area_m2 = positive_pixels * (gsd_meters ** 2)
    return round(float(area_m2 / 1e6), 3)


def tool_calculate_haversine_distance(
    coord_a: tuple[float, float],
    coord_b: tuple[float, float]
) -> float:
    """
    Computes great-circle distance between two (lat, lon) coordinates in kilometers using Haversine formula.
    """
    lat1, lon1 = np.radians(coord_a[0]), np.radians(coord_a[1])
    lat2, lon2 = np.radians(coord_b[0]), np.radians(coord_b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(a))
    r_earth_km = 6371.0
    return round(float(r_earth_km * c), 3)


def tool_calculate_euclidean_distance(point_a: tuple[float, float], point_b: tuple[float, float], gsd_meters: float = 10.0) -> float:
    """Measures metric Euclidean distance between two pixel coordinates in meters."""
    dx = (point_a[0] - point_b[0]) * gsd_meters
    dy = (point_a[1] - point_b[1]) * gsd_meters
    return round(float(np.sqrt(dx**2 + dy**2)), 2)


def tool_export_kml_document(features: list[dict[str, Any]], name: str = "SatQuery_GIS_Layer") -> dict[str, Any]:
    """
    Generates standard OGC KML 2.2 XML string for GIS software (QGIS/ArcGIS/Google Earth).

    Features missing real 'coordinates' are SKIPPED, not substituted with a
    fabricated placeholder polygon (the previous version silently drew a
    generic rectangle near 75.0,12.0 for any feature without coordinates,
    indistinguishable in the exported KML from a real, measured placemark).
    """
    placemarks = []
    skipped = []
    for idx, f in enumerate(features):
        coords = f.get("coordinates")
        if not coords:
            skipped.append(f.get("label", f"Feature #{idx+1}"))
            continue
        coords_str = " ".join([f"{c[0]},{c[1]},0" for c in coords])
        placemarks.append(f"""
    <Placemark>
      <name>{f.get('label', f'Feature #{idx+1}')}</name>
      <Polygon>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>{coords_str}</coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>""")

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{name}</name>
    {''.join(placemarks)}
  </Document>
</kml>"""

    return {
        "kml": kml,
        "placemark_count": len(placemarks),
        "skipped_features_missing_coordinates": skipped,
    }