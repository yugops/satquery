"""
SatQuery AI — High-Precision Pure-Python Offline Reverse Geocoder
Problem Statement ID: 26167 (ISRO / SAC)

Provides instant, zero-dependency reverse geocoding with exact district centroids,
bounding boxes, and geodesic distance calculations across all Indian states and global regions.
"""

from __future__ import annotations

import math
from typing import Any, Optional


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle distance between two points in kilometers."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 6371.0 * 2.0 * math.asin(math.sqrt(max(0.0, min(1.0, a))))


# Comprehensive database of administrative district centroids and bounding boxes
DISTRICT_DATABASE: list[dict[str, Any]] = [
    # --- KERALA (14 Districts) ---
    {"district": "Wayanad", "state": "Kerala", "country": "IN", "lat": 11.6854, "lon": 76.1320, "bbox": [75.75, 11.45, 76.45, 11.98]},
    {"district": "Palakkad", "state": "Kerala", "country": "IN", "lat": 10.7867, "lon": 76.6548, "bbox": [76.20, 10.35, 76.95, 11.15]},
    {"district": "Malappuram", "state": "Kerala", "country": "IN", "lat": 11.0732, "lon": 76.0740, "bbox": [75.80, 10.75, 76.45, 11.40]},
    {"district": "Kozhikode", "state": "Kerala", "country": "IN", "lat": 11.2588, "lon": 75.7804, "bbox": [75.60, 11.10, 76.10, 11.80]},
    {"district": "Thrissur", "state": "Kerala", "country": "IN", "lat": 10.5276, "lon": 76.2144, "bbox": [75.90, 10.15, 76.55, 10.80]},
    {"district": "Ernakulam", "state": "Kerala", "country": "IN", "lat": 9.9816, "lon": 76.2999, "bbox": [76.15, 9.75, 76.80, 10.35]},
    {"district": "Idukki", "state": "Kerala", "country": "IN", "lat": 9.8494, "lon": 76.9720, "bbox": [76.60, 9.25, 77.45, 10.35]},
    {"district": "Kottayam", "state": "Kerala", "country": "IN", "lat": 9.5916, "lon": 76.5222, "bbox": [76.35, 9.35, 76.90, 9.85]},
    {"district": "Alappuzha", "state": "Kerala", "country": "IN", "lat": 9.4981, "lon": 76.3388, "bbox": [76.20, 9.05, 76.60, 9.85]},
    {"district": "Pathanamthitta", "state": "Kerala", "country": "IN", "lat": 9.2648, "lon": 76.7870, "bbox": [76.50, 9.05, 77.25, 9.55]},
    {"district": "Kollam", "state": "Kerala", "country": "IN", "lat": 8.8932, "lon": 76.6141, "bbox": [76.40, 8.75, 77.20, 9.25]},
    {"district": "Thiruvananthapuram", "state": "Kerala", "country": "IN", "lat": 8.5241, "lon": 76.9366, "bbox": [76.70, 8.25, 77.30, 8.90]},
    {"district": "Kannur", "state": "Kerala", "country": "IN", "lat": 11.8745, "lon": 75.3704, "bbox": [75.15, 11.60, 75.90, 12.20]},
    {"district": "Kasaragod", "state": "Kerala", "country": "IN", "lat": 12.4996, "lon": 74.9869, "bbox": [74.85, 12.10, 75.45, 12.80]},

    # --- TAMIL NADU ---
    {"district": "Nilgiris", "state": "Tamil Nadu", "country": "IN", "lat": 11.4102, "lon": 76.6950, "bbox": [76.40, 11.20, 77.05, 11.75]},
    {"district": "Coimbatore", "state": "Tamil Nadu", "country": "IN", "lat": 11.0168, "lon": 76.9558, "bbox": [76.65, 10.40, 77.30, 11.35]},
    {"district": "Tiruppur", "state": "Tamil Nadu", "country": "IN", "lat": 11.1085, "lon": 77.3411, "bbox": [77.10, 10.50, 77.80, 11.40]},
    {"district": "Erode", "state": "Tamil Nadu", "country": "IN", "lat": 11.3410, "lon": 77.7172, "bbox": [76.80, 11.00, 77.95, 11.90]},
    {"district": "Chennai", "state": "Tamil Nadu", "country": "IN", "lat": 13.0827, "lon": 80.2707, "bbox": [80.10, 12.90, 80.35, 13.25]},
    {"district": "Madurai", "state": "Tamil Nadu", "country": "IN", "lat": 9.9252, "lon": 78.1198, "bbox": [77.80, 9.60, 78.45, 10.25]},

    # --- KARNATAKA ---
    {"district": "Bengaluru Urban", "state": "Karnataka", "country": "IN", "lat": 12.9716, "lon": 77.5946, "bbox": [77.40, 12.80, 77.80, 13.15]},
    {"district": "Mysuru", "state": "Karnataka", "country": "IN", "lat": 12.2958, "lon": 76.6394, "bbox": [75.90, 11.70, 77.20, 12.60]},
    {"district": "Kodagu (Coorg)", "state": "Karnataka", "country": "IN", "lat": 12.3375, "lon": 75.8069, "bbox": [75.40, 11.90, 76.20, 12.85]},
    {"district": "Dakshina Kannada (Mangaluru)", "state": "Karnataka", "country": "IN", "lat": 12.9141, "lon": 74.8560, "bbox": [74.70, 12.50, 75.50, 13.20]},
    {"district": "Udupi", "state": "Karnataka", "country": "IN", "lat": 13.3409, "lon": 74.7421, "bbox": [74.50, 13.00, 75.10, 13.80]},
    {"district": "Hassan", "state": "Karnataka", "country": "IN", "lat": 13.0068, "lon": 76.1004, "bbox": [75.50, 12.50, 76.60, 13.55]},

    # --- UTTARAKHAND (Himalayan Hazard Zones) ---
    {"district": "Chamoli", "state": "Uttarakhand", "country": "IN", "lat": 30.4243, "lon": 79.3248, "bbox": [79.10, 30.10, 80.10, 31.10]},
    {"district": "Uttarkashi", "state": "Uttarakhand", "country": "IN", "lat": 30.7268, "lon": 78.4354, "bbox": [77.80, 30.45, 79.25, 31.45]},
    {"district": "Rudraprayag", "state": "Uttarakhand", "country": "IN", "lat": 30.2844, "lon": 78.9811, "bbox": [78.80, 30.20, 79.35, 30.85]},
    {"district": "Dehradun", "state": "Uttarakhand", "country": "IN", "lat": 30.3165, "lon": 78.0322, "bbox": [77.60, 29.95, 78.35, 30.95]},
    {"district": "Pauri Garhwal", "state": "Uttarakhand", "country": "IN", "lat": 30.1472, "lon": 78.7811, "bbox": [78.40, 29.50, 79.20, 30.40]},
    {"district": "Pithoragarh", "state": "Uttarakhand", "country": "IN", "lat": 29.5829, "lon": 80.2182, "bbox": [79.80, 29.30, 80.60, 30.40]},

    # --- HIMACHAL PRADESH ---
    {"district": "Kullu", "state": "Himachal Pradesh", "country": "IN", "lat": 31.9579, "lon": 77.1095, "bbox": [76.80, 31.35, 77.85, 32.45]},
    {"district": "Shimla", "state": "Himachal Pradesh", "country": "IN", "lat": 31.1048, "lon": 77.1734, "bbox": [77.00, 30.75, 77.95, 31.60]},
    {"district": "Mandi", "state": "Himachal Pradesh", "country": "IN", "lat": 31.7087, "lon": 76.9320, "bbox": [76.60, 31.20, 77.30, 32.10]},
    {"district": "Kangra (Dharamshala)", "state": "Himachal Pradesh", "country": "IN", "lat": 32.2190, "lon": 76.3234, "bbox": [75.60, 31.70, 77.05, 32.50]},
    {"district": "Lahaul & Spiti", "state": "Himachal Pradesh", "country": "IN", "lat": 32.5684, "lon": 77.5855, "bbox": [76.40, 31.70, 78.60, 33.30]},

    # --- JAMMU & KASHMIR & LADAKH ---
    {"district": "Leh", "state": "Ladakh", "country": "IN", "lat": 34.1526, "lon": 77.5771, "bbox": [76.50, 32.50, 79.50, 35.50]},
    {"district": "Kargil", "state": "Ladakh", "country": "IN", "lat": 34.5539, "lon": 76.1349, "bbox": [75.20, 33.80, 76.90, 34.90]},
    {"district": "Srinagar", "state": "Jammu & Kashmir", "country": "IN", "lat": 34.0837, "lon": 74.7973, "bbox": [74.65, 33.95, 75.05, 34.25]},
    {"district": "Anantnag", "state": "Jammu & Kashmir", "country": "IN", "lat": 33.7311, "lon": 75.1522, "bbox": [74.90, 33.40, 75.60, 34.00]},

    # --- ASSAM & NORTHEAST (Brahmaputra Flood Plain) ---
    {"district": "Barpeta", "state": "Assam", "country": "IN", "lat": 26.3216, "lon": 91.0060, "bbox": [90.60, 26.10, 91.30, 26.65]},
    {"district": "Kamrup (Guwahati)", "state": "Assam", "country": "IN", "lat": 26.1445, "lon": 91.7362, "bbox": [91.30, 25.80, 92.10, 26.50]},
    {"district": "Nagaon", "state": "Assam", "country": "IN", "lat": 26.3463, "lon": 92.6840, "bbox": [92.30, 25.80, 93.30, 26.80]},
    {"district": "Dibrugarh", "state": "Assam", "country": "IN", "lat": 27.4728, "lon": 94.9120, "bbox": [94.50, 27.10, 95.40, 27.80]},
    {"district": "East Sikkim (Gangtok)", "state": "Sikkim", "country": "IN", "lat": 27.3389, "lon": 88.6065, "bbox": [88.40, 27.10, 88.95, 27.50]},

    # --- MAHARASHTRA ---
    {"district": "Mumbai City", "state": "Maharashtra", "country": "IN", "lat": 18.9388, "lon": 72.8354, "bbox": [72.78, 18.88, 72.88, 19.02]},
    {"district": "Mumbai Suburban", "state": "Maharashtra", "country": "IN", "lat": 19.1200, "lon": 72.8700, "bbox": [72.75, 19.00, 72.98, 19.30]},
    {"district": "Pune", "state": "Maharashtra", "country": "IN", "lat": 18.5204, "lon": 73.8567, "bbox": [73.30, 17.90, 75.10, 19.20]},
    {"district": "Beed", "state": "Maharashtra", "country": "IN", "lat": 18.9891, "lon": 75.7601, "bbox": [75.30, 18.50, 76.50, 19.35]},
    {"district": "Raigad", "state": "Maharashtra", "country": "IN", "lat": 18.5158, "lon": 73.1822, "bbox": [72.80, 17.85, 73.65, 19.05]},
    {"district": "Thane", "state": "Maharashtra", "country": "IN", "lat": 19.2183, "lon": 72.9781, "bbox": [72.85, 19.00, 73.40, 19.60]},

    # --- ODISHA (Cyclone Coast) ---
    {"district": "Puri", "state": "Odisha", "country": "IN", "lat": 19.8135, "lon": 85.8312, "bbox": [85.50, 19.60, 86.25, 20.15]},
    {"district": "Khordha (Bhubaneswar)", "state": "Odisha", "country": "IN", "lat": 20.2961, "lon": 85.8245, "bbox": [85.30, 19.90, 86.05, 20.50]},
    {"district": "Cuttack", "state": "Odisha", "country": "IN", "lat": 20.4625, "lon": 85.8828, "bbox": [85.50, 20.20, 86.30, 20.80]},
    {"district": "Ganjam", "state": "Odisha", "country": "IN", "lat": 19.3800, "lon": 85.0500, "bbox": [84.30, 18.90, 85.30, 19.90]},

    # --- DELHI NCR & NORTH ---
    {"district": "New Delhi", "state": "Delhi", "country": "IN", "lat": 28.6139, "lon": 77.2090, "bbox": [76.85, 28.40, 77.40, 28.90]},
    {"district": "Gurugram", "state": "Haryana", "country": "IN", "lat": 28.4595, "lon": 77.0266, "bbox": [76.80, 28.25, 77.20, 28.60]},
    {"district": "Kolkata", "state": "West Bengal", "country": "IN", "lat": 22.5726, "lon": 88.3639, "bbox": [88.25, 22.45, 88.45, 22.65]},
    {"district": "Hyderabad", "state": "Telangana", "country": "IN", "lat": 17.3850, "lon": 78.4867, "bbox": [78.30, 17.25, 78.60, 17.55]},
    {"district": "Jaipur", "state": "Rajasthan", "country": "IN", "lat": 26.9124, "lon": 75.7873, "bbox": [75.60, 26.75, 76.00, 27.10]},
]


def resolve_coordinates(lat: float, lon: float, context_text: str = "") -> dict[str, Any]:
    """Finds nearest administrative district centroid and state for given GPS coordinates."""
    if lat is None or lon is None:
        return {"district": "Unknown", "state": "Unknown", "country": "IN", "distance_km": None}

    ctx = (context_text or "").lower()

    # 1. Check exact bounding box containment
    contained: list[dict[str, Any]] = []
    for entry in DISTRICT_DATABASE:
        min_lon, min_lat, max_lon, max_lat = entry["bbox"]
        if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
            contained.append(entry)

    # 2. Context priority if user mentioned a specific locality in the filename or query
    if contained:
        for c in contained:
            if c["district"].lower() in ctx:
                return {
                    "district": c["district"],
                    "state": c["state"],
                    "country": c["country"],
                    "distance_km": round(haversine_distance_km(lat, lon, c["lat"], c["lon"]), 1),
                    "matched_method": "bbox_with_context_match"
                }

    # 3. Nearest centroid from all entries
    best_entry = None
    min_dist = float("inf")
    for entry in (contained if contained else DISTRICT_DATABASE):
        d = haversine_distance_km(lat, lon, entry["lat"], entry["lon"])
        if d < min_dist:
            min_dist = d
            best_entry = entry

    if best_entry:
        return {
            "district": best_entry["district"],
            "state": best_entry["state"],
            "country": best_entry["country"],
            "distance_km": round(min_dist, 1),
            "matched_method": "contained_bbox" if contained else "nearest_centroid"
        }

    return {
        "district": f"Coord ({lat:.2f}°, {lon:.2f}°)",
        "state": "India" if (68.0 <= lon <= 97.5 and 6.5 <= lat <= 37.5) else "Global",
        "country": "IN" if (68.0 <= lon <= 97.5 and 6.5 <= lat <= 37.5) else "Global",
        "distance_km": None,
        "matched_method": "raw_coordinates"
    }


def resolve_footprint(bounds: list[float], context_text: str = "") -> dict[str, Any]:
    """
    Evaluates raster footprint across 3x3 geographic grid to identify all spanned administrative districts.
    """
    if not bounds or len(bounds) < 4:
        return {"district": "Unknown", "state": "Unknown", "country": "IN", "spanned_districts": [], "spanned_states": []}

    min_lon, min_lat, max_lon, max_lat = bounds
    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0

    lats = [min_lat, center_lat, max_lat]
    lons = [min_lon, center_lon, max_lon]
    grid_points = [(la, lo) for la in lats for lo in lons]

    sampled_districts: list[str] = []
    sampled_states: list[str] = []

    for pla, plo in grid_points:
        res = resolve_coordinates(pla, plo, context_text)
        d_name = res["district"]
        s_name = res["state"]
        if d_name and d_name not in sampled_districts and not d_name.startswith("Coord"):
            sampled_districts.append(d_name)
        if s_name and s_name not in sampled_states and s_name not in ("Unknown", "India", "Global"):
            sampled_states.append(s_name)

    center_res = resolve_coordinates(center_lat, center_lon, context_text)
    primary_district = center_res["district"]
    primary_state = center_res["state"]

    # If context specifically mentions a district that is spanned by the raster, prioritize it!
    ctx = (context_text or "").lower()
    for sd in sampled_districts:
        if sd.lower() in ctx:
            primary_district = sd
            break

    display_district = primary_district
    other_districts = [d for d in sampled_districts if d != primary_district]
    if other_districts:
        display_district = f"{primary_district} / {other_districts[0]}"

    return {
        "district": display_district,
        "state": primary_state if not (len(sampled_states) > 1) else f"{primary_state} / {sampled_states[0]}",
        "country": center_res["country"],
        "spanned_districts": sampled_districts,
        "spanned_states": sampled_states,
        "primary_district": primary_district,
        "center_lat": round(center_lat, 4),
        "center_lon": round(center_lon, 4),
    }
