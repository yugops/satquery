"""
SatQuery AI — Shared Data Models

All Pydantic models for the typed internal schema used across agents.
Defines enums, request/response types, and geospatial metadata structures.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class IntentType(StrEnum):
    """Classified intent categories for user queries."""
    VQA = "vqa"
    CHANGE = "change_detection"
    GROUNDING = "grounding"
    SAR = "sar_analysis"
    GIS = "gis"
    DISASTER = "disaster"
    DISASTER_CORRELATION = "disaster_correlation"
    UNKNOWN = "unknown"


class AgentStatus(StrEnum):
    """Execution status of an agent run."""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class OutputType(StrEnum):
    """Distinguishes the origin of an agent's result (Rule #12)."""
    MODEL = "model"
    RULE_BASED = "rule_based"
    MOCK = "mock"
    UNAVAILABLE = "unavailable"


# ---------------------------------------------------------------------------
# Geospatial & External Context Models
# ---------------------------------------------------------------------------

class GeoMetadata(BaseModel):
    """Geospatial metadata extracted from a raster file."""
    crs: str | None = None
    affine_transform: list[float] | None = None  # 6-element affine
    bounds: list[float] | None = None            # [left, bottom, right, top]
    width: int | None = None
    height: int | None = None
    band_count: int | None = None
    gsd: float | None = None                     # ground sample distance in meters
    nodata_value: float | None = None
    file_format: str = "unknown"
    is_geotiff: bool = False
    norad_id: int | None = None

    model_config = {"arbitrary_types_allowed": True}


class GeoLocation(BaseModel):
    """Geographic location and extent in EPSG:4326 coordinate system."""
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude in decimal degrees (-90 to 90)")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude in decimal degrees (-180 to 180)")
    crs: str = Field(default="EPSG:4326", description="Coordinate reference system")
    place_name: str | None = Field(default=None, description="Human-readable place or city name")
    bounding_box: tuple[float, float, float, float] | None = Field(
        default=None,
        description="Bounding box coordinates as (min_lon, min_lat, max_lon, max_lat)",
    )

    model_config = {"arbitrary_types_allowed": True}


class WeatherContext(BaseModel):
    """Weather context retrieved from Open-Meteo for validation cross-reference."""
    temperature_2m: float | None = None
    cloudcover: float | None = None
    source: str = "open-meteo"
    output_type: OutputType = OutputType.RULE_BASED

    model_config = {"arbitrary_types_allowed": True}


class TLEResult(BaseModel):
    """Satellite Two-Line Element set and orbital metadata."""
    norad_id: int
    name: str
    line1: str
    line2: str
    epoch: str | None = None
    output_type: OutputType = OutputType.RULE_BASED

    model_config = {"arbitrary_types_allowed": True}


class DisasterEvent(BaseModel):
    """Individual disaster/wildfire event record from kanari/disaster feed."""
    event_id: str
    title: str
    category: str = "wildfire"
    latitude: float
    longitude: float
    timestamp: str | None = None
    severity: str | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


class DisasterCorrelationResult(BaseModel):
    """Result of correlating scene bounding box with disaster feeds."""
    matched_events: list[DisasterEvent] = Field(default_factory=list)
    total_events_checked: int = 0
    is_threat_detected: bool = False
    summary: str = ""
    output_type: OutputType = OutputType.RULE_BASED

    model_config = {"arbitrary_types_allowed": True}


class GeoValidationResult(BaseModel):
    """Structured result of geospatial and cloud validation."""
    geo_metadata: GeoMetadata | None = None
    cloud_result: CloudMaskResult | None = None
    is_sar: bool = False
    validation_passed: bool = False
    location: GeoLocation | None = None
    weather_context: WeatherContext | None = None
    satellite_tle: TLEResult | None = None

    model_config = {"arbitrary_types_allowed": True}

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def keys(self) -> list[str]:
        return [
            "geo_metadata",
            "cloud_result",
            "is_sar",
            "validation_passed",
            "location",
            "weather_context",
            "satellite_tle",
        ]

    def __iter__(self):
        return iter(self.keys())


class AdministrativeBoundary(BaseModel):
    """Result of reverse geocoding a lat/lon pair."""
    country: str = "Unknown"
    state: str = "Unknown"
    district: str = "Unknown"
    locality: str = "Unknown"
    lat: float = 0.0
    lon: float = 0.0


class GeoLocationResult(BaseModel):
    """Structured output from forward or reverse geocoding."""
    lat: float
    lng: float
    formatted: str
    confidence: int | None = None
    output_type: OutputType = OutputType.RULE_BASED
    raw: dict[str, Any] | None = None

    model_config = {"arbitrary_types_allowed": True}


class CloudMaskResult(BaseModel):
    """Cloud contamination detection output."""
    cloud_fraction: float = 0.0
    is_contaminated: bool = False
    recommendation: str = ""

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("cloud_fraction")
    @classmethod
    def clamp_fraction(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class WeatherForecastResult(BaseModel):
    """Structured weather forecast result from Open-Meteo API."""
    latitude: float
    longitude: float
    elevation: float | None = None
    timezone: str | None = None
    timezone_abbreviation: str | None = None
    utc_offset_seconds: int | None = None
    hourly_timestamps: list[str] = Field(default_factory=list)
    hourly_temperatures: list[float] = Field(default_factory=list)
    hourly_variables: dict[str, list[float]] = Field(default_factory=dict)
    units: dict[str, str] = Field(default_factory=dict)
    current_temperature: float | None = None
    output_type: OutputType = OutputType.RULE_BASED

    model_config = {"arbitrary_types_allowed": True}

    def to_dataframe(self) -> Any:
        """Convert hourly series to a pandas DataFrame."""
        import pandas as pd

        data: dict[str, Any] = {}
        if self.hourly_timestamps:
            data["time"] = pd.to_datetime(self.hourly_timestamps)
        if self.hourly_temperatures:
            data["temperature_2m"] = self.hourly_temperatures
        for var_name, values in self.hourly_variables.items():
            if var_name not in data and len(values) == len(self.hourly_timestamps):
                data[var_name] = values
        return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Intent & Orchestration Models
# ---------------------------------------------------------------------------

class IntentPlan(BaseModel):
    """Output of intent classification."""
    intent_type: IntentType = IntentType.UNKNOWN
    has_pair: bool = False
    is_sar: bool = False
    requires_agents: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    raw_query: str = ""


class AgentExecutionNode(BaseModel):
    """Single node in the execution DAG."""
    agent_name: str
    depends_on: list[str] = Field(default_factory=list)
    priority: int = 0
    params: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent Response Models
# ---------------------------------------------------------------------------

class AgentResponse(BaseModel):
    """Standardized output from any agent (Rule #9)."""
    agent: str
    status: AgentStatus = AgentStatus.SUCCESS
    result: Any = None
    confidence: float | None = None
    execution_time_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    output_type: OutputType = OutputType.MOCK

    model_config = {"arbitrary_types_allowed": True}


class FinalResponse(BaseModel):
    """Orchestrator's combined output returned to the caller."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    intent: IntentPlan = Field(default_factory=IntentPlan)
    agent_responses: list[AgentResponse] = Field(default_factory=list)
    synthesized_answer: str = ""
    total_execution_ms: float = 0.0
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    location: GeoLocation | None = None
    satellite_tle: TLEResult | None = None


# ---------------------------------------------------------------------------
# VQA Models
# ---------------------------------------------------------------------------

class VQAResult(BaseModel):
    """Visual question answering output."""
    answer: str = ""
    reasoning: str = ""
    confidence: float | None = None
    source: str = "MODEL"
    model: str = ""


class CountResult(BaseModel):
    """Object counting output."""
    count: int = 0
    target_class: str = ""
    reasoning: str = ""
    confidence: float | None = None
    source: str = "MODEL"
    model: str = ""


# ---------------------------------------------------------------------------
# Grounding / Segmentation Models
# ---------------------------------------------------------------------------

class BBox(BaseModel):
    """Bounding box with confidence and label."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 0.0
    label: str = ""


class GeoJSONPolygon(BaseModel):
    """GeoJSON-compatible polygon output from segmentation."""
    type: str = "FeatureCollection"
    features: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Change Detection Models
# ---------------------------------------------------------------------------

class HeatmapResult(BaseModel):
    """Change-detection difference heatmap."""
    heatmap_shape: tuple[int, int] | None = None
    min_val: float = 0.0
    max_val: float = 1.0
    mean_change: float = 0.0

    model_config = {"arbitrary_types_allowed": True}


class ChangeClassification(BaseModel):
    """Semantic change classification output."""
    categories: list[str] = Field(default_factory=list)
    changed_area_fraction: float = 0.0
    summary: str = ""
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# SAR Models
# ---------------------------------------------------------------------------

class SARFalseColor(BaseModel):
    """SAR false-color composite output."""
    shape: tuple[int, int, int] | None = None  # (H, W, 3)
    vv_db_range: tuple[float, float] | None = None
    vh_db_range: tuple[float, float] | None = None
    calibration_applied: bool = False

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Audit / Ledger Models
# ---------------------------------------------------------------------------

class LedgerEntry(BaseModel):
    """Audit record stored in SQLite."""
    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    agent_name: str = ""
    query: str = ""
    intent: str = ""
    execution_duration_ms: float = 0.0
    model_version: str = ""
    input_summary: str = ""
    output_summary: str = ""
    confidence: float = 0.0
    status: str = "success"
    output_type: str = "mock"
    error_detail: str = ""
