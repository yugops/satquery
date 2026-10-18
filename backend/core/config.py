"""
SatQuery AI — Configuration Management
Problem Statement ID: 26167 (ISRO / SAC)

Manages system paths, model IDs, checkpoint locations, API keys,
and execution parameters.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

try:
    from pydantic_settings import BaseSettings
    from pydantic import AliasChoices, Field
    HAS_SETTINGS = True
except ImportError:
    from pydantic import BaseModel as BaseSettings, Field
    HAS_SETTINGS = False

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_ROOT = Path(__file__).resolve().parent.parent


# Load .env file if present
env_path = BACKEND_ROOT / ".env"
if env_path.exists():
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip()
                    if k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass


class SatQueryConfig(BaseSettings):
    """Global configuration for SatQuery AI Swarm."""

    # Execution mode: 'real' uses loaded model weights; 'mock' allows offline fallback
    mode: Literal["mock", "real"] = "real"
    enable_mock_vqa: bool = False

    # Database configuration (PostgreSQL or SQLite fallback)
    database_url: str = os.getenv("DATABASE_URL", "")
    db_path: str = str(BACKEND_ROOT / "database" / "satquery.db")

    # Directory paths
    uploads_dir: str = str(BACKEND_ROOT / "uploads")
    evidence_dir: str = str(BACKEND_ROOT / "evidence")

    # Model Checkpoints
    sar_model_path: str = str(WORKSPACE_ROOT / "sar_convnext_shard0.pt")
    vlm_adapter_path: str = str(WORKSPACE_ROOT / "kaggle" / "working" / "satquery_final_adapter")
    vlm_base_model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    vlm_provider: Literal["local_adapter", "hf_inference", "transformers", "mock"] = "local_adapter"

    # Vision Swarm Models
    grounding_dino_model_id: str = "IDEA-Research/grounding-dino-tiny"
    sam_model_id: str = "ChaoningZhang/MobileSAM"
    change_model_id: str = "custom-bit-cd"

    # Thresholds & Parameters
    max_tokens: int = 256
    default_box_threshold: float = 0.35
    cloud_contamination_threshold: float = 0.40
    sar_water_threshold_db: float = -18.0

    # API Keys & External Providers
    hf_api_token: str = os.getenv("HF_TOKEN", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    llm_backend: Literal["rule_based", "groq"] = os.getenv("LLM_BACKEND", "rule_based")  # type: ignore[assignment]
    groq_model_id: str = os.getenv("GROQ_MODEL_ID", "llama-3.3-70b-versatile")

    kanari_api_key: str = ""
    kanari_api_endpoint: str = "https://kanari.io/api/events"

    opencage_api_key: str = ""
    opencage_rate_limit_delay: float = 1.0

    # Weather & Satellite Live Catalogs
    openmeteo_api_endpoint: str = "https://api.open-meteo.com/v1/forecast"
    weather_cache_dir: str = str(BACKEND_ROOT / ".cache")
    tle_api_endpoint: str = "https://tle.ivanstanojevic.me/api/tle"
    bhuvan_stac_endpoint: str = "https://bhuvan-app1.nrsc.gov.in/bhuvan2d/bhuvan/stac"

    # Logging
    log_level: str = "INFO"

    if HAS_SETTINGS:
        model_config = {
            "env_prefix": "SATQUERY_",
            "env_file": ".env",
            "env_file_encoding": "utf-8",
            "extra": "ignore",
        }


config = SatQueryConfig()


def reload_config(**overrides) -> SatQueryConfig:
    global config
    config = SatQueryConfig(**overrides)
    return config
