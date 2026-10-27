"""
SatQuery AI — Configuration Management

Uses pydantic-settings to load configuration from environment variables
or .env files. All settings are prefixed with SATQUERY_ in env vars.

Default mode is 'mock' so the system runs without model downloads.
"""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings


class SatQueryConfig(BaseSettings):
    """Global configuration for SatQuery AI."""

    # Execution mode: 'mock' runs without real models, 'real' loads them
    mode: Literal["mock", "real"] = "real"

    # Explicit fallback to mock VQA if real model is unavailable
    enable_mock_vqa: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "SATQUERY_ENABLE_MOCK_VQA",
            "ENABLE_MOCK_VQA",
            "enable_mock_vqa",
        ),
    )

    # SQLite database path for audit ledger
    db_path: str = "satquery.db"

    # ---- Model IDs (used only in 'real' mode) ----
    vlm_model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    vlm_provider: Literal["hf_inference", "transformers", "mock"] = "hf_inference"
    grounding_dino_model_id: str = "IDEA-Research/grounding-dino-tiny"
    sam_model_id: str = "ChaoningZhang/MobileSAM"
    change_model_id: str = "custom-bit-cd"

    # ---- Inference defaults ----
    max_tokens: int = 256
    default_box_threshold: float = 0.35
    cloud_contamination_threshold: float = 0.4
    sar_water_threshold_db: float = -18.0

    # ---- External APIs ----
    hf_api_token: str = Field(
        default="",
        validation_alias=AliasChoices(
            "SATQUERY_HF_API_TOKEN",
            "HF_API_TOKEN",
            "HF_TOKEN",
            "HUGGINGFACE_HUB_TOKEN",
            "hf_api_token",
        ),
    )
    groq_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "SATQUERY_GROQ_API_KEY",
            "GROQ_API_KEY",
            "groq_api_key",
        ),
    )
    llm_backend: Literal["rule_based", "groq"] = Field(
        default="rule_based",
        validation_alias=AliasChoices(
            "SATQUERY_LLM_BACKEND",
            "LLM_BACKEND",
            "llm_backend",
        ),
    )
    groq_model_id: str = "qwen/qwen3.8-27b"

    kanari_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "SATQUERY_KANARI_API_KEY",
            "KANARI_API_KEY",
            "kanari_api_key",
        ),
    )
    kanari_api_endpoint: str = "https://kanari.io/api/events"

    opencage_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "SATQUERY_OPENCAGE_API_KEY",
            "OPENCAGE_API_KEY",
            "opencage_api_key",
        ),
    )
    opencage_rate_limit_delay: float = 1.0  # Min interval in seconds (free tier: 1 req/sec)

    # ---- Weather API (Open-Meteo) ----
    openmeteo_api_endpoint: str = "https://api.open-meteo.com/v1/forecast"
    weather_cache_dir: str = ".cache"
    weather_cache_expire_after: int = 3600  # seconds (1 hour)
    weather_retries: int = 5
    weather_backoff_factor: float = 0.2

    # ---- Satellite Metadata (TLE) ----
    tle_api_endpoint: str = "https://tle.ivanstanojevic.me/api/tle"

    # ---- Logging ----
    log_level: str = "INFO"

    model_config = {
        "env_prefix": "SATQUERY_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Module-level singleton — importable as `from core.config import config`
config = SatQueryConfig()


def reload_config(**overrides) -> SatQueryConfig:
    """Create a fresh config, optionally overriding fields.
    Useful for tests that need a different mode or db_path."""
    global config
    config = SatQueryConfig(**overrides)
    return config
