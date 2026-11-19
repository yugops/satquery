"""
SatQuery AI — Custom Exception Hierarchy

All exceptions inherit from SatQueryError for unified error handling.
Agents catch specific exceptions and convert them into structured
AgentResponse objects with status=FAILED.
"""


class SatQueryError(Exception):
    """Base exception for all SatQuery AI errors."""

    def __init__(self, message: str = "", detail: str = ""):
        self.message = message
        self.detail = detail
        super().__init__(self.message)


class ImageValidationError(SatQueryError):
    """Raised when an uploaded image is invalid, corrupt, or unreadable."""
    pass


class MetadataExtractionError(SatQueryError):
    """Raised when CRS, affine transform, or other geospatial metadata
    cannot be extracted from the raster file."""
    pass


class AgentExecutionError(SatQueryError):
    """Raised when an agent's _execute() method fails."""

    def __init__(self, agent_name: str, message: str = "", detail: str = ""):
        self.agent_name = agent_name
        super().__init__(message=message, detail=detail)


class ModelUnavailableError(SatQueryError):
    """Raised when a required ML model is not loaded and mode != 'mock'."""

    def __init__(self, model_id: str, message: str = ""):
        self.model_id = model_id
        super().__init__(
            message=message or f"Model '{model_id}' is not available.",
            detail=f"model_id={model_id}"
        )


class DAGExecutionError(SatQueryError):
    """Raised when the orchestrator's DAG execution encounters an
    unrecoverable error (e.g., circular dependency, missing agent)."""
    pass


class PairRequiredError(SatQueryError):
    """Raised when change detection is requested but only one image
    is provided."""

    def __init__(self, message: str = ""):
        super().__init__(
            message=message or "Change detection requires an image pair (t1 and t2).",
            detail="has_pair=False"
        )


# ---------------------------------------------------------------------------
# Geocoding Exceptions
# ---------------------------------------------------------------------------

class GeocodingError(SatQueryError):
    """Base exception for all geocoding-related errors."""
    pass


class GeocodingNoResultsError(GeocodingError):
    """Raised when no geocoding results are found for a query or coordinate pair."""
    pass


class GeocodingRateLimitError(GeocodingError):
    """Raised when the OpenCage rate limit is exceeded (HTTP 429) or quota reached (HTTP 402)."""
    pass


class GeocodingAuthenticationError(GeocodingError):
    """Raised when the OpenCage API key is missing, invalid, or unauthorized (HTTP 401/403)."""
    pass


class GeocodingNetworkError(GeocodingError):
    """Raised when a network failure or connection timeout occurs during geocoding."""
    pass


# ---------------------------------------------------------------------------
# Weather API Exceptions
# ---------------------------------------------------------------------------

class WeatherError(SatQueryError):
    """Base exception for all weather-related errors."""
    pass


class WeatherAPIError(WeatherError):
    """Raised when the Open-Meteo weather API returns an error or invalid payload."""
    pass


class WeatherNetworkError(WeatherError):
    """Raised when a network failure or connection timeout occurs during weather queries."""
    pass


class WeatherLocationError(WeatherError):
    """Raised when invalid coordinates are provided for weather forecasting."""
    pass


# ---------------------------------------------------------------------------
# Satellite Orbital (TLE) Exceptions
# ---------------------------------------------------------------------------

class SatelliteMetadataError(SatQueryError):
    """Base exception for satellite orbital / TLE metadata retrieval errors."""
    pass


# ---------------------------------------------------------------------------
# Disaster Feeds (Kanari) Exceptions
# ---------------------------------------------------------------------------

class DisasterFeedError(SatQueryError):
    """Base exception for disaster and wildfire feed retrieval errors."""
    pass


