"""
SatQuery AI — Open-Meteo Weather Service

Provides structured access to the Open-Meteo Weather Forecast API
using openmeteo-requests, requests-cache, retry-requests, numpy, and pandas.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

from core.config import SatQueryConfig, config as global_config
from core.exceptions import (
    WeatherAPIError,
    WeatherError,
    WeatherLocationError,
    WeatherNetworkError,
)
from core.models import OutputType, WeatherForecastResult

logger = logging.getLogger(__name__)


class OpenMeteoWeatherService:
    """Service client for the Open-Meteo Forecast API.

    Features:
        - Response caching via `requests-cache` (configurable TTL).
        - Exponential backoff retries via `retry-requests`.
        - High-performance binary FlatBuffers parsing via `openmeteo-requests`.
        - Conversion of time series to numpy arrays and pandas DataFrames.
        - Graceful error handling and mock mode support.
    """

    def __init__(
        self,
        config: SatQueryConfig | None = None,
        session: Any | None = None,
    ):
        self.config = config or global_config
        self._session = session or self._create_cached_retry_session()
        self._client = openmeteo_requests.Client(session=self._session)

    def _create_cached_retry_session(self) -> Any:
        """Initialize requests-cache session wrapped with retry-requests."""
        cache_session = requests_cache.CachedSession(
            cache_name=self.config.weather_cache_dir,
            expire_after=self.config.weather_cache_expire_after,
        )
        retry_session = retry(
            cache_session,
            retries=self.config.weather_retries,
            backoff_factor=self.config.weather_backoff_factor,
        )
        return retry_session

    def validate_coordinates(self, lat: float, lon: float) -> None:
        """Validate latitude and longitude ranges.

        Raises:
            WeatherLocationError: If coordinates are out of bounds.
        """
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            raise WeatherLocationError(
                message=f"Coordinates must be numeric: lat={lat}, lon={lon}",
                detail=f"lat={lat}, lon={lon}",
            )
        if lat < -90.0 or lat > 90.0:
            raise WeatherLocationError(
                message=f"Latitude {lat} out of range [-90, 90]",
                detail=f"lat={lat}",
            )
        if lon < -180.0 or lon > 180.0:
            raise WeatherLocationError(
                message=f"Longitude {lon} out of range [-180, 180]",
                detail=f"lon={lon}",
            )

    def get_hourly_forecast(
        self,
        lat: float,
        lon: float,
        hourly_variables: list[str] | str | None = None,
        forecast_days: int = 7,
    ) -> WeatherForecastResult:
        """Fetch and parse hourly weather forecast from Open-Meteo.

        Args:
            lat: Latitude in decimal degrees (-90 to 90).
            lon: Longitude in decimal degrees (-180 to 180).
            hourly_variables: Single variable name or list of variables
                (e.g., ["temperature_2m", "relative_humidity_2m"]).
                Defaults to ["temperature_2m"].
            forecast_days: Number of forecast days (1 to 16, default 7).

        Returns:
            WeatherForecastResult with timestamps, temperatures, and units.

        Raises:
            WeatherLocationError: If coordinates are invalid.
            WeatherNetworkError: If network connection fails.
            WeatherAPIError: If Open-Meteo API returns an error response.
            WeatherError: For unexpected exceptions.
        """
        self.validate_coordinates(lat, lon)

        if hourly_variables is None:
            hourly_list = ["temperature_2m"]
        elif isinstance(hourly_variables, str):
            hourly_list = [hourly_variables]
        else:
            hourly_list = list(hourly_variables)

        # Mock mode fallback for deterministic unit testing without network
        if self.config.mode == "mock":
            return self._generate_mock_forecast(lat, lon, hourly_list, forecast_days)

        params: dict[str, Any] = {
            "latitude": lat,
            "longitude": lon,
            "hourly": hourly_list,
            "forecast_days": max(1, min(16, forecast_days)),
        }

        try:
            responses = self._client.weather_api(
                self.config.openmeteo_api_endpoint,
                params=params,
            )
            if not responses:
                raise WeatherAPIError(
                    message="Open-Meteo returned an empty response.",
                    detail=f"endpoint={self.config.openmeteo_api_endpoint}",
                )
            return self._parse_openmeteo_response(responses[0], hourly_list)
        except (WeatherLocationError, WeatherAPIError):
            raise
        except (requests_cache.requests.RequestException, Exception) as exc:
            err_msg = str(exc)
            logger.error("Open-Meteo API request failed: %s", err_msg)
            if any(
                keyword in err_msg.lower()
                for keyword in ["timeout", "connection", "connect", "dns", "network", "unreachable"]
            ):
                raise WeatherNetworkError(
                    message=f"Network error querying Open-Meteo: {err_msg}",
                    detail=f"endpoint={self.config.openmeteo_api_endpoint}, lat={lat}, lon={lon}",
                ) from exc
            raise WeatherAPIError(
                message=f"Failed to fetch weather forecast: {err_msg}",
                detail=f"lat={lat}, lon={lon}",
            ) from exc

    def get_hourly_forecast_safe(
        self,
        lat: float,
        lon: float,
        hourly_variables: list[str] | str | None = None,
        forecast_days: int = 7,
    ) -> WeatherForecastResult | None:
        """Safely fetch hourly weather forecast without raising exceptions.

        Returns None if an error occurs.
        """
        try:
            return self.get_hourly_forecast(
                lat=lat,
                lon=lon,
                hourly_variables=hourly_variables,
                forecast_days=forecast_days,
            )
        except Exception as exc:
            logger.warning("Weather forecast lookup failed safely: %s", exc)
            return None

    def get_current_temperature(self, lat: float, lon: float) -> float | None:
        """Fetch the current temperature for a coordinate pair."""
        result = self.get_hourly_forecast_safe(lat, lon, hourly_variables=["temperature_2m"], forecast_days=1)
        if result and result.current_temperature is not None:
            return result.current_temperature
        return None

    def _parse_openmeteo_response(
        self,
        response: Any,
        hourly_list: list[str],
    ) -> WeatherForecastResult:
        """Parse raw Open-Meteo response into WeatherForecastResult."""
        lat = float(response.Latitude())
        lon = float(response.Longitude())
        elevation = float(response.Elevation()) if response.Elevation() is not None else None
        timezone_str = response.Timezone() if hasattr(response, "Timezone") else None
        tz_abbr = (
            response.TimezoneAbbreviation()
            if hasattr(response, "TimezoneAbbreviation")
            else None
        )
        utc_offset = (
            int(response.UtcOffsetSeconds())
            if hasattr(response, "UtcOffsetSeconds")
            else None
        )

        hourly = response.Hourly()
        date_range = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
        timestamps = [ts.isoformat() for ts in date_range]

        hourly_vars_data: dict[str, list[float]] = {}
        hourly_temps: list[float] = []

        for idx, var_name in enumerate(hourly_list):
            raw_array = hourly.Variables(idx).ValuesAsNumpy()
            clean_values = [round(float(v), 2) for v in raw_array]
            hourly_vars_data[var_name] = clean_values
            if var_name == "temperature_2m":
                hourly_temps = clean_values

        current_temp = hourly_temps[0] if hourly_temps else None

        return WeatherForecastResult(
            latitude=lat,
            longitude=lon,
            elevation=elevation,
            timezone=timezone_str,
            timezone_abbreviation=tz_abbr,
            utc_offset_seconds=utc_offset,
            hourly_timestamps=timestamps,
            hourly_temperatures=hourly_temps,
            hourly_variables=hourly_vars_data,
            units={"temperature": "°C", "time": "iso8601"},
            current_temperature=current_temp,
            output_type=OutputType.RULE_BASED,
        )

    def _generate_mock_forecast(
        self,
        lat: float,
        lon: float,
        hourly_list: list[str],
        forecast_days: int,
    ) -> WeatherForecastResult:
        """Generate synthetic deterministic forecast for testing in mock mode."""
        total_hours = forecast_days * 24
        now_ts = pd.Timestamp.now(tz="UTC").floor("h")
        date_range = pd.date_range(start=now_ts, periods=total_hours, freq="h")
        timestamps = [ts.isoformat() for ts in date_range]

        # Diurnal temperature cycle: base + amplitude * sin(2*pi*hour/24)
        hour_indices = np.arange(total_hours)
        base_temp = 20.0 - abs(lat) * 0.2
        temps = base_temp + 5.0 * np.sin(2 * np.pi * (hour_indices - 6) / 24)
        hourly_temps = [round(float(t), 2) for t in temps]

        hourly_vars: dict[str, list[float]] = {}
        for var in hourly_list:
            if var == "temperature_2m":
                hourly_vars[var] = hourly_temps
            else:
                hourly_vars[var] = [0.0] * total_hours

        return WeatherForecastResult(
            latitude=lat,
            longitude=lon,
            elevation=50.0,
            timezone="UTC",
            timezone_abbreviation="UTC",
            utc_offset_seconds=0,
            hourly_timestamps=timestamps,
            hourly_temperatures=hourly_temps,
            hourly_variables=hourly_vars,
            units={"temperature": "°C", "time": "iso8601"},
            current_temperature=hourly_temps[0] if hourly_temps else None,
            output_type=OutputType.MOCK,
        )


# ---------------------------------------------------------------------------
# Standalone Tool Functions
# ---------------------------------------------------------------------------

def tool_fetch_hourly_weather(
    lat: float,
    lon: float,
    hourly_variables: list[str] | str | None = None,
    forecast_days: int = 7,
    config: SatQueryConfig | None = None,
) -> WeatherForecastResult:
    """Fetch hourly weather forecast data using the Open-Meteo API.

    Args:
        lat: Latitude in decimal degrees (-90 to 90).
        lon: Longitude in decimal degrees (-180 to 180).
        hourly_variables: Variable names to retrieve (default: ['temperature_2m']).
        forecast_days: Number of forecast days (1 to 16).
        config: Optional SatQueryConfig override.

    Returns:
        WeatherForecastResult with hourly series.
    """
    service = OpenMeteoWeatherService(config=config)
    return service.get_hourly_forecast(
        lat=lat,
        lon=lon,
        hourly_variables=hourly_variables,
        forecast_days=forecast_days,
    )


def tool_get_current_temperature(
    lat: float,
    lon: float,
    config: SatQueryConfig | None = None,
) -> float | None:
    """Retrieve the current temperature at coordinates or None if unavailable."""
    service = OpenMeteoWeatherService(config=config)
    return service.get_current_temperature(lat, lon)
