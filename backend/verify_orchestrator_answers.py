"""
SatQuery AI — Verification of End-to-End Orchestration & Intelligence Synthesis
Tests domain queries: Drought, Landslide, STAC Search, Cyclone, Evac Routing, and Merkle Provenance.
"""

import asyncio
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from core.config import config
from agents import (
    LeadOrchestratorAgent,
    GeoValidatorAgent,
    AuditLedgerAgent,
    SingleSceneVQAAgent,
    VisualGroundingAgent,
    BiTemporalChangeAgent,
    SARCloudPenetrationAgent,
    DroughtAndCropHealthAgent,
    FloodAndInundationAgent,
    CycloneAndCoastalRiskAgent,
    LandslideAndGLOFAgent,
    WildfireAndBurnScarAgent,
    GISSpatialMeasurementAgent,
    HistoricalMemoryAgent,
    SituationReportGeneratorAgent,
    DataIngestAgent,
    SuperResPansharpenAgent,
    EvacRoutingAgent,
    BhashiniVoiceAgent,
    VerifiableLedgerAgent,
    EdgeDeployExportAgent,
    CloudPenetrationDiffuser,
)


async def run_verification():
    print("=" * 80)
    print("🛰️  VERIFYING DOMAIN-SPECIFIC REMOTE SENSING INTELLIGENCE SYNTHESIS")
    print("=" * 80)

    # Initialize Swarm
    orchestrator = LeadOrchestratorAgent(config)
    agents = [
        GeoValidatorAgent(config),
        AuditLedgerAgent(config),
        SingleSceneVQAAgent(config),
        VisualGroundingAgent(config),
        BiTemporalChangeAgent(config),
        SARCloudPenetrationAgent(config),
        DroughtAndCropHealthAgent(config),
        FloodAndInundationAgent(config),
        CycloneAndCoastalRiskAgent(config),
        LandslideAndGLOFAgent(config),
        WildfireAndBurnScarAgent(config),
        GISSpatialMeasurementAgent(config),
        HistoricalMemoryAgent(config),
        SituationReportGeneratorAgent(config),
        DataIngestAgent(config),
        SuperResPansharpenAgent(config),
        EvacRoutingAgent(config),
        BhashiniVoiceAgent(config),
        VerifiableLedgerAgent(config),
        EdgeDeployExportAgent(config),
        CloudPenetrationDiffuser(config),
    ]
    for a in agents:
        orchestrator.register_agent(a.agent_id, a)

    workspace_dir = BACKEND_DIR.parent
    opt_t1 = str(workspace_dir / "frontend" / "src" / "assets" / "optical-before.jpg")
    wayanad_tif = r"C:\Users\yugv3\Downloads\wayanad_20240704_TCI.tif"
    if os.path.exists(wayanad_tif):
        opt_t1 = wayanad_tif

    sar_img = str(workspace_dir / "frontend" / "src" / "assets" / "sar-scene.jpg")

    queries = [
        ("Drought & Crop Health", "Assess agricultural drought, crop canopy moisture, and NDVI health in Wayanad", {"image_path": opt_t1, "district": "Wayanad"}),
        ("Landslide Stability", "Evaluate landslide susceptibility, slope failure risk, and SHALSTAB safety factor", {"image_path": opt_t1, "district": "Wayanad"}),
        ("STAC Satellite Search", "Find Sentinel-2 satellite imagery passes over Wayanad for July 2024", {"district": "Wayanad"}),
        ("Cyclone & Coastal", "Assess cyclone storm surge risk and soil salinity hazard with 140 km/h wind and 40 mbar pressure drop", {"wind_speed_kmh": 140.0, "pressure_drop_mbar": 40.0, "district": "Puri Coast, Odisha"}),
        ("Evacuation Corridor", "Calculate safe evacuation corridor and bypass flooded hazard zones", {"start": (0, 0), "goal": (19, 19), "water_mask": None}),
    ]

    for label, query, extra in queries:
        print(f"\n{'─' * 80}")
        print(f"📌 TEST: {label}")
        print(f"❓ Query: \"{query}\"")
        print(f"{'─' * 80}")
        resp = await orchestrator.run(query=query, **extra)
        print(f"📊 Intent: {resp.intent.intent_type.value.upper()} (Agents: {', '.join(resp.agents_executed)})")
        print(f"🎯 Confidence: {resp.confidence * 100:.1f}% ({resp.uncertainty_level}) | Mode: {resp.computation_mode.value}")
        print(f"🔒 Merkle Root: {resp.merkle_root[:22]}...")
        print(f"\n💡 Synthesized Intelligence Answer:\n{resp.primary_answer}")

    print("\n" + "=" * 80)
    print("✅ ALL DOMAIN SYNTHESIS TESTS COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_verification())
