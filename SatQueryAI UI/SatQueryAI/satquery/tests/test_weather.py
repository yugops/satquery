"""
SatQuery AI — Tests for Open-Meteo Weather Service & Tools
"""

import json
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from core.config import SatQueryConfig
from core.exceptions import (
    WeatherAPIError,
    WeatherLocationError,
    WeatherNetworkError,
)
from core.models import OutputType, WeatherForecastResult
from core.weather import (
    OpenMeteoWeatherService,
    tool_fetch_hourly_weather,
    tool_get_current_temperature,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_config():
    return SatQueryConfig(mode="mock", db_path=":memory:")


@pytest.fixture
def real_config():
    return SatQueryConfig(mode="real", db_path=":memory:")


@pytest.fixture
def weather_service_mock(mock_config):
    return OpenMeteoWeatherService(config=mock_config)


@pytest.fixture
def weather_service_real(real_config):
    return OpenMeteoWeatherService(config=real_config)


# ---------------------------------------------------------------------------
# Coordinate Validation Tests
# ---------------------------------------------------------------------------

class TestCoordinateValidation:
    def test_valid_coordinates_pass(self, weather_service_mock):
        weather_service_mock.validate_coordinates(52.52, 13.41)
        weather_service_mock.validate_coordinates(-90.0, -180.0)
        weather_service_mock.validate_coordinates(90.0, 180.0)
        weather_service_mock.validate_coordinates(0.0, 0.0)

    def test_latitude_out_of_bounds_raises(self, weather_service_mock):
        with pytest.raises(WeatherLocationError) as exc_info:
            weather_service_mock.validate_coordinates(91.0, 0.0)
        assert "Latitude 91.0 out of range" in str(exc_info.value)

        with pytest.raises(WeatherLocationError):
            weather_service_mock.validate_coordinates(-90.1, 0.0)

    def test_longitude_out_of_bounds_raises(self, weather_service_mock):
        with pytest.raises(WeatherLocationError) as exc_info:
            weather_service_mock.validate_coordinates(0.0, 180.1)
        assert "Longitude 180.1 out of range" in str(exc_info.value)

        with pytest.raises(WeatherLocationError):
            weather_service_mock.validate_coordinates(0.0, -180.1)

    def test_non_numeric_coordinates_raises(self, weather_service_mock):
        with pytest.raises(WeatherLocationError):
            weather_service_mock.validate_coordinates("invalid", 10.0)  # type: ignore


# ---------------------------------------------------------------------------
# Mock Mode Tests
# ---------------------------------------------------------------------------

class TestWeatherMockMode:
    def test_get_hourly_forecast_mock(self, weather_service_mock):
        result = weather_service_mock.get_hourly_forecast(lat=52.52, lon=13.41, forecast_days=3)
        assert isinstance(result, WeatherForecastResult)
        assert result.latitude == 52.52
        assert result.longitude == 13.41
        assert result.output_type == OutputType.MOCK
        assert len(result.hourly_timestamps) == 3 * 24
        assert len(result.hourly_temperatures) == 3 * 24
        assert result.current_temperature is not None
        assert "temperature" in result.units

    def test_to_dataframe_mock(self, weather_service_mock):
        result = weather_service_mock.get_hourly_forecast(lat=52.52, lon=13.41, forecast_days=2)
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 48
        assert "time" in df.columns
        assert "temperature_2m" in df.columns

    def test_tool_functions_mock(self, mock_config):
        result = tool_fetch_hourly_weather(52.52, 13.41, config=mock_config)
        assert isinstance(result, WeatherForecastResult)
        assert result.output_type == OutputType.MOCK

        temp = tool_get_current_temperature(52.52, 13.41, config=mock_config)
        assert temp is not None
        assert isinstance(temp, float)


# ---------------------------------------------------------------------------
# Live Open-Meteo Integration Tests (Berlin 52.52, 13.41)
# ---------------------------------------------------------------------------

class TestOpenMeteoLiveIntegration:
    """Test dynamic live query against Open-Meteo API."""

    def test_live_forecast_berlin(self, weather_service_real):
        # Berlin sample coordinates as required: 52.52, 13.41
        result = weather_service_real.get_hourly_forecast(
            lat=52.52,
            lon=13.41,
            hourly_variables=["temperature_2m"],
            forecast_days=3,
        )

        assert isinstance(result, WeatherForecastResult)
        assert pytest.approx(result.latitude, abs=0.1) == 52.52
        assert pytest.approx(result.longitude, abs=0.1) == 13.41
        assert result.output_type == OutputType.RULE_BASED
        assert len(result.hourly_timestamps) == 3 * 24
        assert len(result.hourly_temperatures) == 3 * 24
        assert result.current_temperature is not None
        assert isinstance(result.current_temperature, float)
        assert -50.0 <= result.current_temperature <= 60.0  # reasonable Earth temp
        assert result.units.get("temperature") == "°C"

        # Verify DataFrame conversion
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3 * 24
        assert "time" in df.columns
        assert "temperature_2m" in df.columns

    def test_live_forecast_multi_variable(self, weather_service_real):
        result = weather_service_real.get_hourly_forecast(
            lat=52.52,
            lon=13.41,
            hourly_variables=["temperature_2m", "relative_humidity_2m"],
            forecast_days=1,
        )

        assert len(result.hourly_temperatures) == 24
        assert "relative_humidity_2m" in result.hourly_variables
        assert len(result.hourly_variables["relative_humidity_2m"]) == 24

        df = result.to_dataframe()
        assert "relative_humidity_2m" in df.columns
        assert len(df) == 24

    def test_live_current_temperature_tool(self, real_config):
        temp = tool_get_current_temperature(52.52, 13.41, config=real_config)
        assert temp is not None
        assert isinstance(temp, float)
        assert -50.0 <= temp <= 60.0


# ---------------------------------------------------------------------------
# Error Handling & Resilience Tests
# ---------------------------------------------------------------------------

class TestWeatherErrorHandling:
    def test_network_failure_raises_weather_network_error(self, weather_service_real):
        with patch.object(
            weather_service_real._client,
            "weather_api",
            side_effect=Exception("Connection timed out to api.open-meteo.com"),
        ):
            with pytest.raises(WeatherNetworkError) as exc_info:
                weather_service_real.get_hourly_forecast(52.52, 13.41)
            assert "Network error" in str(exc_info.value)

    def test_api_empty_response_raises_weather_api_error(self, weather_service_real):
        with patch.object(weather_service_real._client, "weather_api", return_value=[]):
            with pytest.raises(WeatherAPIError) as exc_info:
                weather_service_real.get_hourly_forecast(52.52, 13.41)
            assert "empty response" in str(exc_info.value)

    def test_get_hourly_forecast_safe_returns_none_on_error(self, weather_service_real):
        with patch.object(
            weather_service_real._client,
            "weather_api",
            side_effect=Exception("Catastrophic connection failure"),
        ):
            res = weather_service_real.get_hourly_forecast_safe(52.52, 13.41)
            assert res is None

    def test_get_current_temperature_safe_returns_none_on_error(self, weather_service_real):
        with patch.object(
            weather_service_real,
            "get_hourly_forecast_safe",
            return_value=None,
        ):
            temp = weather_service_real.get_current_temperature(52.52, 13.41)
            assert temp is None


# ---------------------------------------------------------------------------
# core/external/weather.py Tests
# ---------------------------------------------------------------------------

class TestExternalGetWeather:
    def test_get_weather_mock(self, mock_config):
        from core.external.weather import get_weather

        ctx = get_weather(52.52, 13.41, config=mock_config)
        assert ctx is not None
        assert ctx.source == "open-meteo"
        assert ctx.output_type == OutputType.MOCK
        assert ctx.temperature_2m is not None
        assert ctx.cloudcover is not None

    def test_get_weather_mocked_http_success(self, real_config):
        import io
        from unittest.mock import MagicMock, patch
        from core.external.weather import get_weather

        mock_payload = {
            "hourly": {
                "time": ["2026-08-27T00:00"],
                "temperature_2m": [18.4],
                "cloud_cover": [42.0],
            }
        }
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(mock_payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None

        with patch("urllib.request.urlopen", return_value=mock_resp):
            ctx = get_weather(52.52, 13.41, config=real_config)
            assert ctx is not None
            assert ctx.temperature_2m == 18.4
            assert ctx.cloudcover == 42.0
            assert ctx.source == "open-meteo"
            assert ctx.output_type == OutputType.RULE_BASED

    def test_get_weather_network_error_returns_none(self, real_config):
        from unittest.mock import patch
        from core.external.weather import get_weather

        with patch("urllib.request.urlopen", side_effect=Exception("Timeout")):
            ctx = get_weather(52.52, 13.41, config=real_config)
            assert ctx is None

