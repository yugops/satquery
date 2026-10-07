"""
SatQuery AI — Agent 2: GeoValidatorAgent

Wing: Geospatial_Integrity

Validates uploaded satellite imagery, extracts geospatial metadata
(CRS, affine transform, GSD, bounds), detects cloud contamination,
and identifies SAR-specific inputs. Handles both GeoTIFF and plain
image formats gracefully.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from core.config import SatQueryConfig
from core.exceptions import (
    GeocodingAuthenticationError,
    GeocodingError,
    GeocodingNetworkError,
    GeocodingNoResultsError,
    GeocodingRateLimitError,
    ImageValidationError,
    MetadataExtractionError,
)
from core.external.satellite_metadata import get_tle
from core.external.weather import get_weather
from core.interfaces import BaseAgent
from core.models import (
    AdministrativeBoundary,
    CloudMaskResult,
    GeoLocation,
    GeoLocationResult,
    GeoMetadata,
    GeoValidationResult,
    OutputType,
    TLEResult,
    WeatherContext,
)


# ---------------------------------------------------------------------------
# Tool Functions
# ---------------------------------------------------------------------------

def tool_extract_geotiff_metadata(file_path: str) -> GeoMetadata:
    """Extract geospatial metadata from a raster file.

    Uses rasterio for GeoTIFF files, falls back to PIL/imageio for
    plain images (JPEG, PNG, BMP). Non-GeoTIFF files get is_geotiff=False
    and geospatial fields set to None.

    Args:
        file_path: Absolute or relative path to the image file.

    Returns:
        GeoMetadata with all extractable fields populated.

    Raises:
        ImageValidationError: If the file does not exist or is unreadable.
    """
    if not os.path.isfile(file_path):
        raise ImageValidationError(
            message=f"File not found: {file_path}",
            detail=f"path={file_path}",
        )

    ext = os.path.splitext(file_path)[1].lower()
    geotiff_exts = {".tif", ".tiff", ".geotiff"}

    # ---- Try GeoTIFF via rasterio ----
    if ext in geotiff_exts:
        try:
            import rasterio

            with rasterio.open(file_path) as src:
                transform = src.transform
                gsd = None
                if transform and transform.a:
                    # GSD approximation from pixel size
                    gsd = abs(transform.a)

                return GeoMetadata(
                    crs=str(src.crs) if src.crs else None,
                    affine_transform=list(transform)[:6] if transform else None,
                    bounds=list(src.bounds) if src.bounds else None,
                    width=src.width,
                    height=src.height,
                    band_count=src.count,
                    gsd=gsd,
                    nodata_value=src.nodata,
                    file_format="GeoTIFF",
                    is_geotiff=True,
                )
        except ImportError:
            # rasterio not installed — fall through to PIL
            pass
        except Exception as exc:
            raise MetadataExtractionError(
                message=f"Failed to read GeoTIFF metadata: {exc}",
                detail=f"path={file_path}",
            )

    # ---- Fallback: plain image via PIL ----
    try:
        from PIL import Image

        with Image.open(file_path) as img:
            width, height = img.size
            # Approximate band count from mode
            mode_bands = {"L": 1, "LA": 2, "RGB": 3, "RGBA": 4}
            band_count = mode_bands.get(img.mode, len(img.getbands()))

            return GeoMetadata(
                width=width,
                height=height,
                band_count=band_count,
                file_format=img.format or ext.lstrip(".").upper(),
                is_geotiff=False,
            )
    except ImportError:
        # PIL also not available — return minimal metadata
        return GeoMetadata(file_format=ext.lstrip(".").upper())
    except Exception as exc:
        raise ImageValidationError(
            message=f"Failed to read image: {exc}",
            detail=f"path={file_path}",
        )


def tool_extract_geolocation(file_path: str) -> GeoLocation | None:
    """Extract GeoLocation from a GeoTIFF raster file.

    Reads the GeoTIFF's CRS, affine transform, and bounds via rasterio.
    Computes the center point in EPSG:4326 (reprojecting if source CRS differs).
    Sets the bounding box (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
    Populates place_name using reverse_geocoder if installed; otherwise None.

    Args:
        file_path: Absolute or relative path to the image file.

    Returns:
        GeoLocation if valid GeoTIFF with spatial reference; otherwise None.
    """
    if not file_path or not os.path.isfile(file_path):
        return None

    ext = os.path.splitext(file_path)[1].lower()
    geotiff_exts = {".tif", ".tiff", ".geotiff"}
    if ext not in geotiff_exts:
        return None

    try:
        import rasterio
        from rasterio.warp import transform, transform_bounds
    except ImportError:
        return None

    try:
        with rasterio.open(file_path) as src:
            if not src.crs or not src.bounds:
                return None

            bounds = src.bounds
            left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top

            # Center point in source CRS coordinates
            center_x = (left + right) / 2.0
            center_y = (bottom + top) / 2.0

            src_crs = src.crs
            src_crs_str = str(src_crs).upper()

            if src_crs_str in ("EPSG:4326", "OGC:CRS84", "WGS 84", "+PROJ=LONGLAT +DATUM=WGS84 +NO_DEFS"):
                center_lon = float(center_x)
                center_lat = float(center_y)
                min_lon = float(min(left, right))
                min_lat = float(min(bottom, top))
                max_lon = float(max(left, right))
                max_lat = float(max(bottom, top))
            else:
                # Reproject center point to EPSG:4326
                dst_xs, dst_ys = transform(src_crs, "EPSG:4326", [center_x], [center_y])
                center_lon = float(dst_xs[0])
                center_lat = float(dst_ys[0])

                # Reproject bounds to EPSG:4326 -> (min_lon, min_lat, max_lon, max_lat)
                b_left, b_bottom, b_right, b_top = transform_bounds(
                    src_crs, "EPSG:4326", left, bottom, right, top
                )
                min_lon = float(b_left)
                min_lat = float(b_bottom)
                max_lon = float(b_right)
                max_lat = float(b_top)

            # Clamp lat/lon coordinates to valid ranges
            center_lat = max(-90.0, min(90.0, center_lat))
            center_lon = max(-180.0, min(180.0, center_lon))
            min_lat = max(-90.0, min(90.0, min_lat))
            max_lat = max(-90.0, min(90.0, max_lat))
            min_lon = max(-180.0, min(180.0, min_lon))
            max_lon = max(-180.0, min(180.0, max_lon))

            bounding_box = (min_lon, min_lat, max_lon, max_lat)

            # Reverse geocode place_name if reverse_geocoder is available
            place_name = None
            try:
                import reverse_geocoder as rg  # type: ignore

                results = rg.search((center_lat, center_lon))
                if results:
                    r = results[0]
                    name = r.get("name")
                    if name and name != "Unknown":
                        place_name = name
            except Exception:
                place_name = None

            return GeoLocation(
                latitude=center_lat,
                longitude=center_lon,
                crs="EPSG:4326",
                place_name=place_name,
                bounding_box=bounding_box,
            )
    except Exception:
        return None


def tool_reverse_geocode_gps(lat: float, lon: float) -> AdministrativeBoundary:
    """Reverse-geocode a GPS coordinate to an administrative boundary.

    Uses the offline `reverse_geocoder` library (no API key needed).
    Returns a placeholder in mock mode or if the library is unavailable.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.

    Returns:
        AdministrativeBoundary with country, state, district, locality.
    """
    try:
        import reverse_geocoder as rg  # type: ignore

        results = rg.search((lat, lon))
        if results:
            r = results[0]
            return AdministrativeBoundary(
                country=r.get("cc", "Unknown"),
                state=r.get("admin1", "Unknown"),
                district=r.get("admin2", "Unknown"),
                locality=r.get("name", "Unknown"),
                lat=lat,
                lon=lon,
            )
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback placeholder
    return AdministrativeBoundary(
        country="Unknown",
        state="Unknown",
        district="Unknown",
        locality="Unknown",
        lat=lat,
        lon=lon,
    )


def tool_detect_cloud_contamination(
    rgb_image: np.ndarray,
    threshold: float = 0.4,
) -> CloudMaskResult:
    """Detect cloud contamination in an optical satellite image.

    Uses a brightness/whiteness heuristic on the RGB channels:
    pixels where all channels exceed 200 (on 0–255 scale) are
    classified as cloud. If the cloud fraction exceeds the threshold,
    the image is flagged as contaminated and SAR fallback is recommended.

    Args:
        rgb_image: (H, W, 3) uint8 array.
        threshold: Cloud fraction threshold for contamination flag.

    Returns:
        CloudMaskResult with fraction, flag, and recommendation.
    """
    if rgb_image is None or rgb_image.size == 0:
        return CloudMaskResult(
            cloud_fraction=0.0,
            is_contaminated=False,
            recommendation="No image data provided.",
        )

    # Normalize to 0–255 if needed
    img = rgb_image
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    # Ensure 3-channel
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]

    # Cloud mask: all channels bright and similar (whitish)
    bright = np.all(img > 200, axis=-1)
    total_pixels = img.shape[0] * img.shape[1]
    cloud_pixels = int(np.sum(bright))
    cloud_fraction = cloud_pixels / total_pixels if total_pixels > 0 else 0.0

    is_contaminated = cloud_fraction > threshold
    recommendation = ""
    if is_contaminated:
        recommendation = (
            f"Cloud cover {cloud_fraction:.1%} exceeds threshold "
            f"({threshold:.0%}). Consider SAR imagery fallback."
        )

    return CloudMaskResult(
        cloud_fraction=round(cloud_fraction, 4),
        is_contaminated=is_contaminated,
        recommendation=recommendation,
    )


# ---------------------------------------------------------------------------
# OpenCage Geocoding Tools
# ---------------------------------------------------------------------------

OPENCAGE_API_ENDPOINT = "https://api.opencagedata.com/geocode/v1/json"


class OpenCageRateLimiter:
    """Thread-safe request rate limiter for OpenCage API (1 request/second)."""

    def __init__(self, min_interval: float = 1.0):
        self.min_interval = min_interval
        self._last_call = 0.0
        self._lock = threading.Lock()

    def wait(self, interval: float | None = None) -> None:
        """Enforce minimum time interval between consecutive requests."""
        delay_interval = self.min_interval if interval is None else interval
        if delay_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < delay_interval:
                time.sleep(delay_interval - elapsed)
            self._last_call = time.monotonic()


_opencage_limiter = OpenCageRateLimiter(min_interval=1.0)


def tool_opencage_geocode(
    query: str,
    api_key: str,
    min_interval: float = 1.0,
    timeout: float = 10.0,
) -> GeoLocationResult:
    """Geocode a place name or coordinate query using the OpenCage API.

    Handles request throttling (1 req/sec limit on free tier), network errors,
    rate limits, authentication, and structured result extraction.

    Args:
        query: Place name (e.g. 'Mumbai') or 'lat,lng' coordinate string.
        api_key: OpenCage API key.
        min_interval: Minimum interval in seconds between requests.
        timeout: HTTP request timeout in seconds.

    Returns:
        GeoLocationResult with lat, lng, formatted address, confidence,
        and output_type=OutputType.RULE_BASED.

    Raises:
        GeocodingNoResultsError: If query is empty or no matching location is found.
        GeocodingAuthenticationError: If API key is missing or invalid (HTTP 401/403).
        GeocodingRateLimitError: If rate limit is hit (HTTP 429) or quota reached (HTTP 402).
        GeocodingNetworkError: If a network error or connection timeout occurs.
        GeocodingError: For other unexpected geocoding errors.
    """
    if not query or not query.strip():
        raise GeocodingNoResultsError(
            message="Geocoding query cannot be empty.",
            detail="query=''",
        )

    if not api_key or not api_key.strip():
        raise GeocodingAuthenticationError(
            message="OpenCage API key is missing. Set SATQUERY_OPENCAGE_API_KEY.",
            detail="api_key=''",
        )

    # Respect rate limiting (free tier: 1 req/sec)
    _opencage_limiter.wait(min_interval)

    params = {
        "q": query.strip(),
        "key": api_key.strip(),
        "limit": 1,
        "no_annotations": 0,
    }
    url = f"{OPENCAGE_API_ENDPOINT}?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "SatQueryAI/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw_body = response.read().decode("utf-8")
            data = json.loads(raw_body)
    except urllib.error.HTTPError as err:
        error_body = ""
        try:
            error_body = err.read().decode("utf-8")
        except Exception:
            pass

        if err.code in (401, 403):
            raise GeocodingAuthenticationError(
                message=f"OpenCage authentication failed (HTTP {err.code}): {err.reason}",
                detail=error_body or f"code={err.code}",
            ) from err
        elif err.code == 402:
            raise GeocodingRateLimitError(
                message="OpenCage quota exceeded (HTTP 402). Daily request limit reached.",
                detail=error_body or "quota_exceeded",
            ) from err
        elif err.code == 429:
            raise GeocodingRateLimitError(
                message="OpenCage rate limit hit (HTTP 429). Free tier allows max 1 request/second.",
                detail=error_body or "rate_limit_exceeded",
            ) from err
        else:
            raise GeocodingError(
                message=f"OpenCage HTTP error {err.code}: {err.reason}",
                detail=error_body or f"code={err.code}",
            ) from err
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise GeocodingNetworkError(
            message=f"Network error while connecting to OpenCage: {exc}",
            detail=f"query={query}",
        ) from exc
    except json.JSONDecodeError as exc:
        raise GeocodingError(
            message=f"Failed to parse OpenCage JSON response: {exc}",
            detail=f"query={query}",
        ) from exc
    except Exception as exc:
        if isinstance(exc, GeocodingError):
            raise
        raise GeocodingError(
            message=f"Unexpected geocoding error: {exc}",
            detail=f"query={query}",
        ) from exc

    results = data.get("results", [])
    if not results:
        raise GeocodingNoResultsError(
            message=f"No geocoding results found for: '{query}'",
            detail=f"query={query}",
        )

    first = results[0]
    geom = first.get("geometry", {})
    lat = float(geom.get("lat", 0.0))
    lng = float(geom.get("lng", 0.0))
    formatted = str(first.get("formatted", ""))
    confidence_raw = first.get("confidence")
    confidence = None
    if confidence_raw is not None:
        try:
            confidence = int(confidence_raw)
        except (ValueError, TypeError):
            confidence = None

    return GeoLocationResult(
        lat=lat,
        lng=lng,
        formatted=formatted,
        confidence=confidence,
        output_type=OutputType.RULE_BASED,
        raw=first,
    )


def tool_opencage_forward_geocode(
    place_name: str,
    api_key: str,
    min_interval: float = 1.0,
    timeout: float = 10.0,
) -> GeoLocationResult:
    """Forward-geocode a place name to a GeoLocationResult."""
    return tool_opencage_geocode(
        query=place_name,
        api_key=api_key,
        min_interval=min_interval,
        timeout=timeout,
    )


def tool_opencage_reverse_geocode(
    lat: float,
    lng: float,
    api_key: str,
    min_interval: float = 1.0,
    timeout: float = 10.0,
) -> GeoLocationResult:
    """Reverse-geocode latitude/longitude coordinates to a GeoLocationResult."""
    query = f"{lat:.7f},{lng:.7f}"
    return tool_opencage_geocode(
        query=query,
        api_key=api_key,
        min_interval=min_interval,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# GeoValidatorAgent
# ---------------------------------------------------------------------------

class GeoValidatorAgent(BaseAgent):
    """Agent 2 — Validates imagery and extracts geospatial metadata.

    Invoked on every newly uploaded raster or image before specialist
    agents run. Provides normalized GeoMetadata to the orchestrator.
    Also provides forward and reverse geocoding via OpenCage API.
    """

    agent_name = "GeoValidatorAgent"
    mempalace_wing = "Wing: Geospatial_Integrity"

    def forward_geocode(self, place_name: str) -> tuple[float, float] | None:
        """Convert a place name into (lat, lng) coordinates.

        Useful for query inputs like 'show me flooding near Mumbai'.
        In mock mode, returns synthetic coordinates without making API calls.

        Args:
            place_name: Human-readable location query (e.g. 'Mumbai, India').

        Returns:
            Tuple of (latitude, longitude) or None if place name is empty.

        Raises:
            GeocodingNoResultsError: If no matching location is found.
            GeocodingRateLimitError: If OpenCage rate limit is hit.
            GeocodingAuthenticationError: If API key is missing or invalid.
            GeocodingNetworkError: If network connection fails.
        """
        if not place_name or not place_name.strip():
            return None

        result = self.forward_geocode_detailed(place_name)
        return (result.lat, result.lng)

    def reverse_geocode(self, lat: float, lng: float) -> str | None:
        """Convert scene coordinates into a human-readable place name.

        In mock mode, returns a synthetic place name without making API calls.

        Args:
            lat: Latitude in decimal degrees.
            lng: Longitude in decimal degrees.

        Returns:
            Formatted address string or None.

        Raises:
            GeocodingNoResultsError: If no matching location is found.
            GeocodingRateLimitError: If OpenCage rate limit is hit.
            GeocodingAuthenticationError: If API key is missing or invalid.
            GeocodingNetworkError: If network connection fails.
        """
        result = self.reverse_geocode_detailed(lat, lng)
        return result.formatted

    def forward_geocode_detailed(self, place_name: str) -> GeoLocationResult:
        """Forward-geocode a place name returning a structured GeoLocationResult.

        In mock mode, returns a synthetic GeoLocationResult with output_type=OutputType.MOCK.
        In real mode, queries OpenCage API with output_type=OutputType.RULE_BASED.
        """
        if not place_name or not place_name.strip():
            raise GeocodingNoResultsError(
                message="Geocoding place name cannot be empty.",
                detail="place_name=''",
            )

        if self.config.mode == "mock":
            # Deterministic synthetic mock result
            clean_name = place_name.strip()
            return GeoLocationResult(
                lat=19.0760,
                lng=72.8777,
                formatted=f"{clean_name}, Synthetic Region",
                confidence=10,
                output_type=OutputType.MOCK,
                raw={"query": clean_name, "mock": True},
            )

        return tool_opencage_forward_geocode(
            place_name=place_name,
            api_key=self.config.opencage_api_key,
            min_interval=self.config.opencage_rate_limit_delay,
        )

    def reverse_geocode_detailed(self, lat: float, lng: float) -> GeoLocationResult:
        """Reverse-geocode coordinates returning a structured GeoLocationResult.

        In mock mode, returns a synthetic GeoLocationResult with output_type=OutputType.MOCK.
        In real mode, queries OpenCage API with output_type=OutputType.RULE_BASED.
        """
        if self.config.mode == "mock":
            # Synthetic mock address
            return GeoLocationResult(
                lat=lat,
                lng=lng,
                formatted=f"Mock Location ({lat:.4f}, {lng:.4f}), Earth",
                confidence=10,
                output_type=OutputType.MOCK,
                raw={"lat": lat, "lng": lng, "mock": True},
            )

        return tool_opencage_reverse_geocode(
            lat=lat,
            lng=lng,
            api_key=self.config.opencage_api_key,
            min_interval=self.config.opencage_rate_limit_delay,
        )

    async def _execute(self, **kwargs: Any) -> GeoValidationResult:
        """Validate an image file and return structured metadata.

        Expected kwargs:
            image_path (str): Path to the uploaded image.
            image_array (np.ndarray | None): Pre-loaded RGB array for
                cloud detection (optional — loaded from path if absent).
            norad_id (int | None): Optional satellite catalog ID for TLE lookup.

        Returns:
            GeoValidationResult with geo_metadata, cloud_result, is_sar flag,
            location, weather_context, and satellite_tle.
        """
        image_path: str | None = kwargs.get("image_path")
        image_array: np.ndarray | None = kwargs.get("image_array")
        norad_id: int | None = kwargs.get("norad_id")

        geo_metadata: GeoMetadata | None = None
        location: GeoLocation | None = None
        is_sar: bool = False
        cloud_result: CloudMaskResult | None = None
        weather_context: WeatherContext | None = None
        satellite_tle: TLEResult | None = None

        # Step 1: Extract metadata and geolocation
        if image_path:
            geo_metadata = tool_extract_geotiff_metadata(image_path)
            location = tool_extract_geolocation(image_path)

            if norad_id is None and geo_metadata and geo_metadata.norad_id:
                norad_id = geo_metadata.norad_id
            elif norad_id is None:
                # Check filename pattern for norad / tle id
                import re
                match = re.search(r"(?:norad|tle)[-_]?(\d+)", image_path, re.IGNORECASE)
                if match:
                    try:
                        norad_id = int(match.group(1))
                    except ValueError:
                        pass

            # Heuristic SAR detection: single-band or filename hint
            lower_path = image_path.lower()
            if any(tag in lower_path for tag in ["_vv", "_vh", "sar", "sentinel1", "s1"]):
                is_sar = True
            elif geo_metadata.band_count is not None and geo_metadata.band_count <= 2:
                is_sar = True

        # Step 2: Cloud detection (optical only)
        if image_array is not None and not is_sar:
            cloud_result = tool_detect_cloud_contamination(
                image_array,
                threshold=self.config.cloud_contamination_threshold,
            )
        elif image_path and not is_sar:
            # Try to load the image for cloud detection
            try:
                from PIL import Image

                img = Image.open(image_path).convert("RGB")
                arr = np.array(img)
                cloud_result = tool_detect_cloud_contamination(
                    arr,
                    threshold=self.config.cloud_contamination_threshold,
                )
            except Exception:
                pass  # Cloud detection is best-effort

        # Step 3: Weather context from Open-Meteo for cloud sanity-check
        if location is not None:
            weather_context = get_weather(
                lat=location.latitude,
                lon=location.longitude,
                config=self.config,
            )
            if weather_context and weather_context.cloudcover is not None and cloud_result is not None:
                # Sanity check optical cloud fraction against Open-Meteo meteorological report
                if cloud_result.is_contaminated:
                    cloud_result.recommendation += f" (Open-Meteo confirms cloud cover at {weather_context.cloudcover:.1f}%)"
                elif weather_context.cloudcover >= 60.0:
                    cloud_result.recommendation += f" (Note: Open-Meteo reports regional cloud cover at {weather_context.cloudcover:.1f}%)"

        # Step 4: Satellite TLE orbital data
        if norad_id is not None:
            satellite_tle = get_tle(norad_id=norad_id, config=self.config)

        return GeoValidationResult(
            geo_metadata=geo_metadata,
            cloud_result=cloud_result,
            is_sar=is_sar,
            validation_passed=True,
            location=location,
            weather_context=weather_context,
            satellite_tle=satellite_tle,
        )
