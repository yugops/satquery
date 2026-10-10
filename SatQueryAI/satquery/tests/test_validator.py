import json
import os
import tempfile
import urllib.error
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.config import SatQueryConfig
from core.exceptions import (
    GeocodingAuthenticationError,
    GeocodingError,
    GeocodingNetworkError,
    GeocodingNoResultsError,
    GeocodingRateLimitError,
    ImageValidationError,
)
from core.models import (
    AgentStatus,
    CloudMaskResult,
    GeoLocation,
    GeoLocationResult,
    GeoMetadata,
    GeoValidationResult,
    OutputType,
)
from core.validator import (
    GeoValidatorAgent,
    OpenCageRateLimiter,
    tool_detect_cloud_contamination,
    tool_extract_geolocation,
    tool_extract_geotiff_metadata,
    tool_opencage_forward_geocode,
    tool_opencage_geocode,
    tool_opencage_reverse_geocode,
    tool_reverse_geocode_gps,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_config():
    return SatQueryConfig(mode="mock", db_path=":memory:")


@pytest.fixture
def validator(mock_config):
    return GeoValidatorAgent(config=mock_config)


@pytest.fixture
def temp_png():
    """Create a temporary PNG file for testing."""
    from PIL import Image

    img = Image.fromarray(
        np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    )
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_jpeg():
    """Create a temporary JPEG file for testing."""
    from PIL import Image

    img = Image.fromarray(
        np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
    )
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img.save(f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def sample_geotiff_epsg4326():
    """Create a temporary GeoTIFF with EPSG:4326 coordinate reference system."""
    import rasterio
    from rasterio.transform import from_bounds

    # Coordinates around Mumbai: min_lon=72.8, min_lat=18.9, max_lon=73.0, max_lat=19.1
    transform = from_bounds(72.8, 18.9, 73.0, 19.1, 64, 64)
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
        path = f.name

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(np.zeros((3, 64, 64), dtype=np.uint8))

    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def sample_geotiff_utm():
    """Create a temporary GeoTIFF with UTM Zone 43N (EPSG:32643) coordinates."""
    import rasterio
    from rasterio.transform import from_bounds

    # UTM coordinates for Mumbai area (~ 72.9 E, 19.1 N)
    transform = from_bounds(276000, 2108000, 286000, 2118000, 64, 64)
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
        path = f.name

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=3,
        dtype="uint8",
        crs="EPSG:32643",
        transform=transform,
    ) as dst:
        dst.write(np.zeros((3, 64, 64), dtype=np.uint8))

    yield path
    if os.path.exists(path):
        os.unlink(path)


# ---------------------------------------------------------------------------
# Metadata Extraction Tests
# ---------------------------------------------------------------------------

class TestExtractMetadata:
    def test_png_not_geotiff(self, temp_png):
        meta = tool_extract_geotiff_metadata(temp_png)
        assert isinstance(meta, GeoMetadata)
        assert meta.is_geotiff is False
        assert meta.width == 32
        assert meta.height == 32
        assert meta.band_count == 3
        assert meta.crs is None

    def test_jpeg_not_geotiff(self, temp_jpeg):
        meta = tool_extract_geotiff_metadata(temp_jpeg)
        assert meta.is_geotiff is False
        assert meta.width == 48

    def test_nonexistent_file_raises(self):
        with pytest.raises(ImageValidationError):
            tool_extract_geotiff_metadata("/nonexistent/path/image.tif")

    def test_file_format_detected(self, temp_png):
        meta = tool_extract_geotiff_metadata(temp_png)
        assert meta.file_format.upper() == "PNG"


# ---------------------------------------------------------------------------
# Cloud Detection Tests
# ---------------------------------------------------------------------------

class TestCloudDetection:
    def test_clear_image(self):
        # Dark image — no clouds
        img = np.full((64, 64, 3), 50, dtype=np.uint8)
        result = tool_detect_cloud_contamination(img)
        assert isinstance(result, CloudMaskResult)
        assert result.is_contaminated is False
        assert result.cloud_fraction < 0.1

    def test_cloudy_image(self):
        # All-white image — maximum cloud cover
        img = np.full((64, 64, 3), 230, dtype=np.uint8)
        result = tool_detect_cloud_contamination(img)
        assert result.is_contaminated is True
        assert result.cloud_fraction > 0.5
        assert "SAR" in result.recommendation

    def test_partially_cloudy(self):
        # Half dark, half white
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[32:, :, :] = 220
        result = tool_detect_cloud_contamination(img)
        assert 0.0 < result.cloud_fraction < 1.0

    def test_empty_image(self):
        result = tool_detect_cloud_contamination(np.array([]))
        assert result.cloud_fraction == 0.0


# ---------------------------------------------------------------------------
# Reverse Geocode Tests
# ---------------------------------------------------------------------------

class TestReverseGeocode:
    def test_returns_boundary(self):
        result = tool_reverse_geocode_gps(23.0225, 72.5714)
        assert result.lat == 23.0225
        assert result.lon == 72.5714
        # May be "Unknown" if reverse_geocoder library not installed
        assert isinstance(result.country, str)


# ---------------------------------------------------------------------------
# GeoLocation Extraction & Reprojection Tests
# ---------------------------------------------------------------------------

class TestGeoLocationExtraction:
    def test_geolocation_model_constraints(self):
        # Valid location
        loc = GeoLocation(
            latitude=19.0760,
            longitude=72.8777,
            crs="EPSG:4326",
            place_name="Mumbai",
            bounding_box=(72.8, 18.9, 73.0, 19.1),
        )
        assert loc.latitude == 19.0760
        assert loc.longitude == 72.8777
        assert loc.crs == "EPSG:4326"
        assert loc.place_name == "Mumbai"
        assert loc.bounding_box == (72.8, 18.9, 73.0, 19.1)

        # Invalid latitude > 90
        with pytest.raises(Exception):
            GeoLocation(latitude=91.0, longitude=0.0)

        # Invalid latitude < -90
        with pytest.raises(Exception):
            GeoLocation(latitude=-91.0, longitude=0.0)

        # Invalid longitude > 180
        with pytest.raises(Exception):
            GeoLocation(latitude=0.0, longitude=181.0)

        # Invalid longitude < -180
        with pytest.raises(Exception):
            GeoLocation(latitude=0.0, longitude=-181.0)

    def test_geolocation_epsg4326_fixture(self, sample_geotiff_epsg4326):
        loc = tool_extract_geolocation(sample_geotiff_epsg4326)
        assert loc is not None
        assert isinstance(loc, GeoLocation)
        assert loc.crs == "EPSG:4326"
        assert pytest.approx(loc.latitude, abs=1e-4) == 19.0
        assert pytest.approx(loc.longitude, abs=1e-4) == 72.9
        assert loc.bounding_box is not None
        min_lon, min_lat, max_lon, max_lat = loc.bounding_box
        assert pytest.approx(min_lon, abs=1e-4) == 72.8
        assert pytest.approx(min_lat, abs=1e-4) == 18.9
        assert pytest.approx(max_lon, abs=1e-4) == 73.0
        assert pytest.approx(max_lat, abs=1e-4) == 19.1
        # place_name should be populated when reverse_geocoder is available
        assert isinstance(loc.place_name, str)
        assert len(loc.place_name) > 0

    def test_geolocation_reproject_utm_fixture(self, sample_geotiff_utm):
        loc = tool_extract_geolocation(sample_geotiff_utm)
        assert loc is not None
        assert isinstance(loc, GeoLocation)
        assert loc.crs == "EPSG:4326"
        # Reprojected center should be around Mumbai (lat ~19.098, lon ~72.918)
        assert 19.0 <= loc.latitude <= 19.2
        assert 72.8 <= loc.longitude <= 73.0
        assert loc.bounding_box is not None
        min_lon, min_lat, max_lon, max_lat = loc.bounding_box
        assert 72.8 <= min_lon < max_lon <= 73.1
        assert 19.0 <= min_lat < max_lat <= 19.2
        assert isinstance(loc.place_name, str)
        assert len(loc.place_name) > 0

    def test_geolocation_non_geotiff_returns_none(self, temp_png, temp_jpeg):
        assert tool_extract_geolocation(temp_png) is None
        assert tool_extract_geolocation(temp_jpeg) is None
        assert tool_extract_geolocation("/nonexistent/file.tif") is None

    def test_geolocation_without_reverse_geocoder(self, sample_geotiff_epsg4326):
        with patch.dict("sys.modules", {"reverse_geocoder": None}):
            # When reverse_geocoder fails to import
            loc = tool_extract_geolocation(sample_geotiff_epsg4326)
            # Should still succeed in computing coordinates and bounds with place_name=None
            assert loc is not None
            assert loc.place_name is None
            assert pytest.approx(loc.latitude, abs=1e-4) == 19.0
            assert pytest.approx(loc.longitude, abs=1e-4) == 72.9

    @pytest.mark.asyncio
    async def test_validator_agent_populates_location_geotiff(
        self, validator, sample_geotiff_epsg4326, sample_geotiff_utm
    ):
        # EPSG:4326
        resp = await validator.run(image_path=sample_geotiff_epsg4326)
        assert resp.status == AgentStatus.SUCCESS
        assert resp.result.location is not None
        assert isinstance(resp.result.location, GeoLocation)
        assert resp.result["location"] is not None
        assert pytest.approx(resp.result.location.latitude, abs=1e-4) == 19.0

        # UTM (Reprojected)
        resp_utm = await validator.run(image_path=sample_geotiff_utm)
        assert resp_utm.status == AgentStatus.SUCCESS
        assert resp_utm.result.location is not None
        assert 19.0 <= resp_utm.result.location.latitude <= 19.2

    @pytest.mark.asyncio
    async def test_validator_agent_location_none_for_png(self, validator, temp_png):
        resp = await validator.run(image_path=temp_png)
        assert resp.status == AgentStatus.SUCCESS
        assert resp.result.location is None
        assert resp.result["location"] is None


# ---------------------------------------------------------------------------
# Agent Run Tests
# ---------------------------------------------------------------------------

class TestValidatorAgent:
    @pytest.mark.asyncio
    async def test_run_with_png(self, validator, temp_png):
        response = await validator.run(image_path=temp_png)
        assert response.status == AgentStatus.SUCCESS
        result = response.result
        assert result["validation_passed"] is True
        assert result["geo_metadata"] is not None
        assert result["is_sar"] is False

    @pytest.mark.asyncio
    async def test_run_with_array(self, validator):
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        response = await validator.run(image_array=arr)
        assert response.status == AgentStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_sar_detection_from_filename(self, validator):
        """SAR detection from filename heuristic (file doesn't need to exist
        for the heuristic check — but metadata extraction will fail)."""
        from PIL import Image

        with tempfile.NamedTemporaryFile(
            suffix="_VV.tif", delete=False, prefix="sentinel1_"
        ) as f:
            # Create a minimal file so extraction doesn't crash
            img = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))
            img.save(f.name)
            path = f.name

        try:
            response = await validator.run(image_path=path)
            result = response.result
            assert result["is_sar"] is True
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# OpenCage Geocoding Tests — Mock Mode
# ---------------------------------------------------------------------------

class TestOpenCageMockMode:
    """Verify mock mode behaviour: synthetic results without network calls."""

    def test_forward_geocode_mock(self, validator):
        coords = validator.forward_geocode("Mumbai")
        assert coords == (19.0760, 72.8777)

    def test_forward_geocode_mock_empty(self, validator):
        coords = validator.forward_geocode("")
        assert coords is None

    def test_forward_geocode_detailed_mock(self, validator):
        result = validator.forward_geocode_detailed("Mumbai")
        assert isinstance(result, GeoLocationResult)
        assert result.lat == 19.0760
        assert result.lng == 72.8777
        assert "Mumbai" in result.formatted
        assert result.output_type == OutputType.MOCK
        assert result.confidence == 10

    def test_forward_geocode_detailed_empty_raises(self, validator):
        with pytest.raises(GeocodingNoResultsError):
            validator.forward_geocode_detailed("")

    def test_reverse_geocode_mock(self, validator):
        formatted = validator.reverse_geocode(19.0760, 72.8777)
        assert isinstance(formatted, str)
        assert "19.0760" in formatted

    def test_reverse_geocode_detailed_mock(self, validator):
        result = validator.reverse_geocode_detailed(19.0760, 72.8777)
        assert isinstance(result, GeoLocationResult)
        assert result.lat == 19.0760
        assert result.lng == 72.8777
        assert result.output_type == OutputType.MOCK
        assert result.confidence == 10


# ---------------------------------------------------------------------------
# OpenCage Geocoding Tests — Tool Functions & Error Handling
# ---------------------------------------------------------------------------

class TestOpenCageToolsAndErrors:
    """Verify OpenCage HTTP tools, parsing, and exception mapping."""

    def test_tool_forward_geocode_success(self):
        fake_response_data = {
            "results": [
                {
                    "geometry": {"lat": 19.0759837, "lng": 72.8776559},
                    "formatted": "Mumbai, Maharashtra, India",
                    "confidence": 9,
                }
            ],
            "status": {"code": 200, "message": "OK"},
            "total_results": 1,
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(fake_response_data).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = tool_opencage_forward_geocode(
                place_name="Mumbai",
                api_key="dummy_key",
                min_interval=0.0,
            )

        assert isinstance(result, GeoLocationResult)
        assert result.lat == 19.0759837
        assert result.lng == 72.8776559
        assert result.formatted == "Mumbai, Maharashtra, India"
        assert result.confidence == 9
        assert result.output_type == OutputType.RULE_BASED

    def test_tool_reverse_geocode_success(self):
        fake_response_data = {
            "results": [
                {
                    "geometry": {"lat": 51.5074456, "lng": -0.1277653},
                    "formatted": "Trafalgar Square, London, United Kingdom",
                    "confidence": 8,
                }
            ],
            "status": {"code": 200, "message": "OK"},
            "total_results": 1,
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(fake_response_data).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = tool_opencage_reverse_geocode(
                lat=51.5074,
                lng=-0.1278,
                api_key="dummy_key",
                min_interval=0.0,
            )

        assert isinstance(result, GeoLocationResult)
        assert result.formatted == "Trafalgar Square, London, United Kingdom"
        assert result.output_type == OutputType.RULE_BASED

    def test_no_results_raises_geocoding_no_results_error(self):
        fake_response_data = {
            "results": [],
            "status": {"code": 200, "message": "OK"},
            "total_results": 0,
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(fake_response_data).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(GeocodingNoResultsError) as exc_info:
                tool_opencage_geocode(
                    query="nonexistentplace12345xyz",
                    api_key="dummy_key",
                    min_interval=0.0,
                )
            assert "No geocoding results found" in str(exc_info.value)

    def test_empty_query_raises_no_results(self):
        with pytest.raises(GeocodingNoResultsError):
            tool_opencage_geocode(query="", api_key="dummy_key", min_interval=0.0)

    def test_missing_api_key_raises_auth_error(self):
        with pytest.raises(GeocodingAuthenticationError):
            tool_opencage_geocode(query="Mumbai", api_key="", min_interval=0.0)

    def test_auth_failure_401_raises(self):
        http_err = urllib.error.HTTPError(
            url="https://api.opencagedata.com/geocode/v1/json",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=None,
        )
        with patch("urllib.request.urlopen", side_effect=http_err):
            with pytest.raises(GeocodingAuthenticationError):
                tool_opencage_geocode(
                    query="Mumbai", api_key="invalid_key", min_interval=0.0
                )

    def test_rate_limit_429_raises(self):
        http_err = urllib.error.HTTPError(
            url="https://api.opencagedata.com/geocode/v1/json",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=None,
        )
        with patch("urllib.request.urlopen", side_effect=http_err):
            with pytest.raises(GeocodingRateLimitError):
                tool_opencage_geocode(
                    query="Mumbai", api_key="dummy_key", min_interval=0.0
                )

    def test_quota_exceeded_402_raises(self):
        http_err = urllib.error.HTTPError(
            url="https://api.opencagedata.com/geocode/v1/json",
            code=402,
            msg="Payment Required / Quota Exceeded",
            hdrs={},
            fp=None,
        )
        with patch("urllib.request.urlopen", side_effect=http_err):
            with pytest.raises(GeocodingRateLimitError):
                tool_opencage_geocode(
                    query="Mumbai", api_key="dummy_key", min_interval=0.0
                )

    def test_network_failure_raises(self):
        url_err = urllib.error.URLError("Connection refused")
        with patch("urllib.request.urlopen", side_effect=url_err):
            with pytest.raises(GeocodingNetworkError):
                tool_opencage_geocode(
                    query="Mumbai", api_key="dummy_key", min_interval=0.0
                )


# ---------------------------------------------------------------------------
# OpenCage Geocoding Tests — Real Mode Agent Delegation
# ---------------------------------------------------------------------------

class TestOpenCageRealModeAgent:
    @pytest.fixture
    def real_validator(self):
        cfg = SatQueryConfig(
            mode="real",
            opencage_api_key="test_api_key",
            opencage_rate_limit_delay=0.0,
            db_path=":memory:",
        )
        return GeoValidatorAgent(config=cfg)

    def test_real_forward_geocode(self, real_validator):
        fake_response_data = {
            "results": [
                {
                    "geometry": {"lat": 19.0760, "lng": 72.8777},
                    "formatted": "Mumbai, Maharashtra, India",
                    "confidence": 9,
                }
            ],
            "status": {"code": 200, "message": "OK"},
            "total_results": 1,
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(fake_response_data).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None

        with patch("urllib.request.urlopen", return_value=mock_resp):
            coords = real_validator.forward_geocode("Mumbai")
            assert coords == (19.0760, 72.8777)

            detailed = real_validator.forward_geocode_detailed("Mumbai")
            assert detailed.output_type == OutputType.RULE_BASED
            assert detailed.formatted == "Mumbai, Maharashtra, India"

    def test_real_reverse_geocode(self, real_validator):
        fake_response_data = {
            "results": [
                {
                    "geometry": {"lat": 19.0760, "lng": 72.8777},
                    "formatted": "Mumbai, Maharashtra, India",
                    "confidence": 9,
                }
            ],
            "status": {"code": 200, "message": "OK"},
            "total_results": 1,
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(fake_response_data).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None

        with patch("urllib.request.urlopen", return_value=mock_resp):
            addr = real_validator.reverse_geocode(19.0760, 72.8777)
            assert addr == "Mumbai, Maharashtra, India"

            detailed = real_validator.reverse_geocode_detailed(19.0760, 72.8777)
            assert detailed.output_type == OutputType.RULE_BASED
            assert detailed.lat == 19.0760


# ---------------------------------------------------------------------------
# OpenCage Throttling & Live Integration Tests
# ---------------------------------------------------------------------------

class TestOpenCageThrottling:
    def test_limiter_delay(self):
        import time
        limiter = OpenCageRateLimiter(min_interval=0.05)
        start = time.monotonic()
        limiter.wait()
        limiter.wait()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.04


class TestOpenCageLiveIntegration:
    """Integration test using the provided API key."""

    LIVE_KEY = "18caf12a20db4020a36767e8dde09038"

    def test_live_forward_and_reverse_geocode(self):
        # Forward geocode
        result_fwd = tool_opencage_forward_geocode(
            place_name="Mumbai",
            api_key=self.LIVE_KEY,
            min_interval=0.5,
        )
        assert isinstance(result_fwd, GeoLocationResult)
        assert 18.0 <= result_fwd.lat <= 20.0
        assert 72.0 <= result_fwd.lng <= 74.0
        assert "India" in result_fwd.formatted
        assert result_fwd.output_type == OutputType.RULE_BASED

        # Reverse geocode
        result_rev = tool_opencage_reverse_geocode(
            lat=result_fwd.lat,
            lng=result_fwd.lng,
            api_key=self.LIVE_KEY,
            min_interval=0.5,
        )
        assert isinstance(result_rev, GeoLocationResult)
        assert isinstance(result_rev.formatted, str)
        assert len(result_rev.formatted) > 0
        assert result_rev.output_type == OutputType.RULE_BASED
