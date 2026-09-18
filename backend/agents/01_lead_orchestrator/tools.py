"""
SatQuery AI — Agent 01: LeadOrchestratorAgent Tools
Problem Statement ID: 26167 (ISRO / SAC)

FIX LOG (this pass):
- Pure-SAR branch (#4) was unconditionally attaching `flood_inundation` to
  required_agents for ANY SAR query, even ones with nothing to do with
  flooding (e.g. "describe backscatter texture in this SAR image").
  Now gated behind an actual flood-keyword check, matching how every other
  disaster-specific pipeline in this file is gated.
"""

from __future__ import annotations

import re
from typing import Any
from core.models import IntentPlan, IntentType


_INTENT_KEYWORD_MAP: dict[IntentType, list[str]] = {
    IntentType.VQA: ["what", "describe", "identify", "how many", "count", "is there", "are there", "scene"],
    IntentType.GROUNDING: ["locate", "find", "where is", "where are", "bounding box", "highlight", "segment"],
    IntentType.CHANGE: ["change", "difference", "compare", "temporal", "before and after", "deforestation", "growth"],
    IntentType.SAR: ["sar", "radar", "sentinel-1", "s1", "backscatter", "polarization", "vv", "vh"],
    IntentType.DROUGHT: ["drought", "crop", "agriculture", "ndvi", "ndmi", "canopy", "yield", "wilting"],
    IntentType.FLOOD: ["flood", "submerged", "inundation", "monsoon overflow", "water level", "dam breach"],
    IntentType.CYCLONE: ["cyclone", "storm surge", "salinity", "coastal risk", "tsunami", "sea water"],
    IntentType.LANDSLIDE: ["landslide", "glof", "slope failure", "mountain slide", "debris flow", "shalstab", "glacial lake"],
    IntentType.WILDFIRE: ["wildfire", "fire", "burn scar", "nbr", "smoke plume", "thermal anomaly", "rothermel"],
    IntentType.DATA_INGEST: ["fetch", "download", "query bhuvan", "stac", "catalog", "get imagery for", "pull pass", "imagery for", "images for", "satellite images", "satellite data", "available scenes", "orbital pass", "find imagery", "find images", "latest images", "recent images", "sentinel for", "sentinel over", "search imagery"],
    IntentType.EVAC_ROUTING: ["evacuate", "evacuation", "route", "relief truck", "safe corridor", "severed road", "detour"],
    IntentType.BHASHINI_VOICE: ["voice", "hindi", "tamil", "telugu", "malayalam", "bengali", "marathi", "gujarati", "kannada", "odia", "assamese", "audio dispatch"],
    IntentType.VERIFIABLE_LEDGER: ["merkle", "proof", "audit ledger", "court admissible", "provenance", "verify integrity"],
    IntentType.EDGE_DEPLOY: ["edge", "export", "onnx", "tensorrt", "gguf", "jetson", "uav", "offline"],
    IntentType.CLOUD_DIFFUSER: ["cloud removal", "diffuser", "latent diffusion", "remove clouds", "sar to optical"],
    IntentType.SUPERRES: ["pansharpen", "super resolution", "unmixing", "cartosat", "subpixel"],
    IntentType.GIS: ["area", "distance", "measurement", "kml", "hectares", "sq km"],
}

# Keywords that justify attaching the flood agent onto a pure-SAR pipeline.
# Kept separate from IntentType.FLOOD's list so this stays easy to audit.
_FLOOD_RELEVANT_KEYWORDS: list[str] = [
    "flood", "submerged", "inundation", "water level", "dam breach",
    "monsoon overflow", "waterlogging", "overflow", "levee", "embankment breach",
]


def tool_classify_intent(query: str, has_pair: bool = False, is_sar: bool = False, mode: str | None = None) -> IntentPlan:
    """Classifies user intent, detecting compound and disaster intents."""
    q_lower = query.lower()
    m_lower = (mode or "").lower()

    # 1. Live Global Satellite Catalog & STAC Ingestion — natural language + explicit prefix
    _stac_triggers = [
        "[search:", "query bhuvan", "stac catalog", "orbital pass", "search catalog",
        "find imagery", "fetch pass", "find sentinel", "get sentinel", "sentinel imagery",
        "satellite imagery for", "available scenes", "satellite passes", "search for imagery",
        "find satellite", "pull imagery", "retrieve imagery", "satellite data for",
        "sar imagery for", "optical imagery", "download imagery",
        "show me imagery", "show imagery", "latest images", "recent images",
        "images of", "images for", "imagery of", "imagery for",
        "satellite images", "satellite scenes", "satellite data",
        "sentinel-2 for", "sentinel-1 for", "sentinel 2 for", "sentinel 1 for",
        "sentinel-2 over", "sentinel-1 over", "sentinel 2 over", "sentinel 1 over",
        "get images", "get scenes", "get data for",
        "show scenes", "show passes", "show satellite",
        "search sentinel", "search satellite", "search images",
        "available imagery", "available images", "available data",
        "imagery over", "images over", "passes over", "passes for",
        "find images", "find scenes", "find data for", "find passes",
        "latest sentinel", "recent sentinel", "latest satellite",
        "fetch imagery", "fetch images", "fetch satellite", "fetch data",
        "pull images", "pull scenes", "pull satellite", "pull data",
        "retrieve images", "retrieve scenes", "retrieve satellite",
        "download images", "download scenes", "download satellite",
        "catalog query", "catalog search", "global catalog",
        "sentinel over", "sentinel for", "sentinel imagery for",
    ]
    if any(k in q_lower for k in _stac_triggers):
        return IntentPlan(
            intent_type=IntentType.DATA_INGEST,
            confidence=0.98,
            required_agents=["data_ingest", "audit_ledger", "verifiable_ledger"],
            description="Global Satellite Catalog & STAC Harvester Pipeline"
        )

    # 2. Optical + SAR Fusion Mode
    is_optical_sar = any(k in m_lower for k in ["optical-sar", "optical_sar", "optical + sar", "sar_optical"]) or (
        ("optical" in q_lower and "sar" in q_lower) or ("radar" in q_lower and "rgb" in q_lower)
    )
    if is_optical_sar:
        return IntentPlan(
            intent_type=IntentType.SAR,
            confidence=0.96,
            required_agents=["geo_validator", "single_scene_vqa", "sar_cloud_penetration", "audit_ledger", "verifiable_ledger"],
            description="Cross-Modal Optical & SAR Cloud Penetration Pipeline"
        )

    # 3. Bi-Temporal Change Detection
    is_change_mode = any(k in m_lower for k in ["bitemporal", "change", "compare", "temporal", "dual"])
    is_change_query = any(k in q_lower for k in ["bitemporal", "before and after", "change detection", "damage assessment", "compare", "scour"])
    if has_pair or is_change_mode or is_change_query:
        return IntentPlan(
            intent_type=IntentType.CHANGE,
            confidence=0.96,
            required_agents=["geo_validator", "bitemporal_change", "audit_ledger", "verifiable_ledger"],
            description="Bi-Temporal Change & Damage Assessment Pipeline"
        )

    # 4. Pure SAR Radar Mode
    # FIX: only attach flood_inundation when the query is actually flood-relevant.
    # Previously this fired on every SAR query regardless of content, e.g. a pure
    # "describe backscatter texture" query would silently get a flood report
    # tacked on top for no reason.
    if is_sar or ("sar" in q_lower or "sentinel-1" in q_lower):
        is_flood_relevant = any(k in q_lower for k in _FLOOD_RELEVANT_KEYWORDS)
        if is_flood_relevant:
            return IntentPlan(
                intent_type=IntentType.SAR,
                confidence=0.95,
                required_agents=["geo_validator", "sar_cloud_penetration", "flood_inundation", "audit_ledger", "verifiable_ledger"],
                description="Sentinel-1 SAR Radar & Flood/Cloud Penetration Pipeline"
            )
        return IntentPlan(
            intent_type=IntentType.SAR,
            confidence=0.95,
            required_agents=["geo_validator", "sar_cloud_penetration", "audit_ledger", "verifiable_ledger"],
            description="Sentinel-1 SAR Radar & Cloud Penetration Pipeline"
        )

    # Keyword scoring
    scores: dict[IntentType, int] = {}
    for itype, kws in _INTENT_KEYWORD_MAP.items():
        score = sum(1 for kw in kws if kw in q_lower)
        if score > 0:
            scores[itype] = score

    if not scores:
        return IntentPlan(
            intent_type=IntentType.VQA,
            confidence=0.80,
            required_agents=["geo_validator", "single_scene_vqa", "audit_ledger", "verifiable_ledger"],
            description="General Satellite Vision-Language Reasoning Pipeline"
        )

    best_intent = max(scores, key=scores.get)

    # Route agent dependencies
    agent_map = {
        IntentType.VQA: ["geo_validator", "single_scene_vqa", "audit_ledger", "verifiable_ledger"],
        IntentType.GROUNDING: ["geo_validator", "visual_grounding", "audit_ledger", "verifiable_ledger"],
        IntentType.CHANGE: ["geo_validator", "bitemporal_change", "audit_ledger", "verifiable_ledger"],
        IntentType.SAR: ["geo_validator", "sar_cloud_penetration", "audit_ledger", "verifiable_ledger"],
        IntentType.DROUGHT: ["geo_validator", "drought_crop_health", "historical_memory", "audit_ledger", "verifiable_ledger"],
        IntentType.FLOOD: ["geo_validator", "sar_cloud_penetration", "flood_inundation", "evac_routing", "historical_memory", "audit_ledger", "verifiable_ledger"],
        IntentType.CYCLONE: ["geo_validator", "cyclone_coastal_risk", "historical_memory", "audit_ledger", "verifiable_ledger"],
        IntentType.LANDSLIDE: ["geo_validator", "landslide_glof", "historical_memory", "audit_ledger", "verifiable_ledger"],
        IntentType.WILDFIRE: ["geo_validator", "wildfire_burn_scar", "audit_ledger", "verifiable_ledger"],
        IntentType.DATA_INGEST: ["data_ingest", "geo_validator", "audit_ledger", "verifiable_ledger"],
        IntentType.EVAC_ROUTING: ["geo_validator", "evac_routing", "audit_ledger", "verifiable_ledger"],
        IntentType.BHASHINI_VOICE: ["geo_validator", "single_scene_vqa", "bhashini_voice", "audit_ledger", "verifiable_ledger"],
        IntentType.VERIFIABLE_LEDGER: ["geo_validator", "audit_ledger", "verifiable_ledger"],
        IntentType.EDGE_DEPLOY: ["edge_deploy", "audit_ledger", "verifiable_ledger"],
        IntentType.CLOUD_DIFFUSER: ["geo_validator", "sar_cloud_penetration", "cloud_diffuser", "audit_ledger", "verifiable_ledger"],
        IntentType.SUPERRES: ["geo_validator", "superres_pansharpen", "audit_ledger", "verifiable_ledger"],
        IntentType.GIS: ["geo_validator", "gis_spatial_measurement", "audit_ledger", "verifiable_ledger"],
    }

    req_agents = agent_map.get(best_intent, ["geo_validator", "single_scene_vqa", "audit_ledger", "verifiable_ledger"])

    return IntentPlan(
        intent_type=best_intent,
        confidence=round(min(0.98, 0.75 + scores[best_intent] * 0.08), 2),
        required_agents=req_agents,
        description=f"Specialized {best_intent.value.upper()} Intelligence Pipeline"
    )