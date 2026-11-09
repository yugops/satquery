"""
Direct test of LeadOrchestratorAgent with real assets.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    SARCloudPenetrationAgent,
    VerifiableLedgerAgent,
    EvacRoutingAgent,
)

async def main():
    orch = LeadOrchestratorAgent(config)
    orch.register_agent("geo_validator", GeoValidatorAgent(config))
    orch.register_agent("single_scene_vqa", SingleSceneVQAAgent(config))
    orch.register_agent("sar_cloud_penetration", SARCloudPenetrationAgent(config))
    orch.register_agent("audit_ledger", AuditLedgerAgent(config))
    orch.register_agent("verifiable_ledger", VerifiableLedgerAgent(config))
    orch.register_agent("evac_routing", EvacRoutingAgent(config))

    # Test 1: Real SAR image with ConvNeXt-Tiny
    sar_img = "d:/satquery/frontend/src/assets/sar-scene.jpg"
    print("--- Running Test 1: SAR Cloud Penetration ---")
    res1 = await orch.run(
        query="Analyze Sentinel-1 SAR backscatter for water bodies.",
        image_path=sar_img
    )
    print("Answer 1:", res1.primary_answer)
    print("Confidence 1:", res1.confidence)
    print("Computation Mode 1:", res1.computation_mode.value)
    print("Merkle Root 1:", res1.merkle_root)
    print("Traces 1:")
    for t in res1.agent_traces:
        print(f"  - {t.agent_name}: {t.computation_mode.value} ({t.execution_time_ms}ms) | {t.summary}")

    # Test 2: Real Optical image with GeoValidator
    opt_img = "d:/satquery/frontend/src/assets/optical-before.jpg"
    print("\n--- Running Test 2: Optical Scene Assessment ---")
    res2 = await orch.run(
        query="Describe this optical satellite scene and count visible buildings.",
        image_path=opt_img
    )
    print("Answer 2:", res2.primary_answer)
    print("Confidence 2:", res2.confidence)
    print("Computation Mode 2:", res2.computation_mode.value)
    print("Merkle Root 2:", res2.merkle_root)
    for t in res2.agent_traces:
        print(f"  - {t.agent_name}: {t.computation_mode.value} ({t.execution_time_ms}ms) | {t.summary}")

if __name__ == "__main__":
    asyncio.run(main())
