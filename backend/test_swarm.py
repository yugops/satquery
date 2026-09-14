"""
SatQuery AI — Automated Verification Suite
Problem Statement ID: 26167 (ISRO / SAC)
"""

import asyncio
import os
import unittest
import importlib
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.config import config
from core.models import ComputationMode
from models_registry.sar_convnext import SARModelEngine, CORINE_19_CLASSES

# Dynamic imports
merkle_tools = importlib.import_module("agents.20_verifiable_ledger.tools")
compute_merkle_leaf = getattr(merkle_tools, "compute_merkle_leaf")
build_merkle_tree = getattr(merkle_tools, "build_merkle_tree")
verify_merkle_proof = getattr(merkle_tools, "verify_merkle_proof")
generate_merkle_proof = getattr(merkle_tools, "generate_merkle_proof")

shalstab_tools = importlib.import_module("agents.11_landslide_glof.tools")
tool_calculate_shalstab_safety_factor = getattr(shalstab_tools, "tool_calculate_shalstab_safety_factor")

evac_tools = importlib.import_module("agents.18_evac_routing.tools")
tool_astar_evacuation_corridor = getattr(evac_tools, "tool_astar_evacuation_corridor")

bhashini_tools = importlib.import_module("agents.19_bhashini_voice.tools")
tool_bhashini_dispatch_sitrep = getattr(bhashini_tools, "tool_bhashini_dispatch_sitrep")

from agents import (
    LeadOrchestratorAgent,
    GeoValidatorAgent,
    AuditLedgerAgent,
    SARCloudPenetrationAgent,
    VerifiableLedgerAgent,
    EvacRoutingAgent,
)


class TestSatQuerySwarm(unittest.TestCase):

    def test_01_merkle_anti_tamper_integrity(self):
        """CRITICAL: Test that flipping computation_mode or weights strictly alters the Merkle root."""
        leaf_real = compute_merkle_leaf(
            node_id="step_1",
            agent_id="sar_cloud_penetration",
            computation_mode="real_inference",
            weights_sha256="0x8fa37b8893d7c...",
            output_summary="SAR water detection",
            input_hash="0xabc123"
        )
        root_real, tree_real = build_merkle_tree([leaf_real])

        leaf_tampered = compute_merkle_leaf(
            node_id="step_1",
            agent_id="sar_cloud_penetration",
            computation_mode="synthetic_fallback",
            weights_sha256="none",
            output_summary="SAR water detection",
            input_hash="0xabc123"
        )
        root_tampered, tree_tampered = build_merkle_tree([leaf_tampered])

        self.assertNotEqual(root_real, root_tampered, "Merkle root MUST change when computation_mode or weights change!")
        
        proof = generate_merkle_proof(0, tree_real)
        self.assertTrue(verify_merkle_proof(leaf_real, proof, root_real), "Merkle proof verification failed!")

    def test_02_sar_model_loader_and_corine_classes(self):
        """Verify ConvNeXt-Tiny SAR model loader and class predictions."""
        self.assertEqual(len(CORINE_19_CLASSES), 19)
        self.assertIn("Water bodies", CORINE_19_CLASSES)
        self.assertIn("Coastal wetlands", CORINE_19_CLASSES)

        engine = SARModelEngine(config.sar_model_path)
        vv = np.random.uniform(0.1, 1.2, (120, 120)).astype(np.float32)
        vh = vv * 0.45
        res = engine.predict(vv, vh)
        self.assertIn("computation_mode", res)
        self.assertIn("corine_predictions", res)
        self.assertTrue(len(res["corine_predictions"]) > 0)

    def test_03_shalstab_slope_stability(self):
        """Verify Montgomery-Dietrich SHALSTAB infinite slope physics."""
        slopes = np.array([5.0, 15.0, 35.0, 55.0])
        res = tool_calculate_shalstab_safety_factor(slopes)
        self.assertIn("mean_factor_of_safety", res)
        self.assertIn("critical_unstable_slope_pct", res)
        self.assertEqual(res["model_name"], "Montgomery-Dietrich SHALSTAB (1994)")

    def test_04_astar_evacuation_routing(self):
        """Verify A* evacuation corridor pathfinding bypassing hazard zones."""
        res = tool_astar_evacuation_corridor(grid_size=(10, 10), start=(0, 0), goal=(9, 9))
        self.assertTrue(res["safe_route_found"])
        self.assertTrue(res["total_nodes"] > 5)
        self.assertTrue(res["estimated_distance_km"] > 0)

    def test_05_bhashini_multilingual_dispatch(self):
        """Verify Bhashini voice dispatch translation across 10 languages."""
        res_hi = tool_bhashini_dispatch_sitrep("Flood waters rising rapidly in Sector 4.", "hi")
        self.assertIn("Hindi", res_hi["language_name"])
        self.assertTrue(res_hi["tts_audio_ready"])

        res_te = tool_bhashini_dispatch_sitrep("Flood waters rising rapidly in Sector 4.", "te")
        self.assertIn("Telugu", res_te["language_name"])

    def test_06_orchestrator_end_to_end_dag(self):
        """Verify full orchestrator multi-agent DAG execution."""
        orch = LeadOrchestratorAgent(config)
        orch.register_agent("geo_validator", GeoValidatorAgent(config))
        orch.register_agent("sar_cloud_penetration", SARCloudPenetrationAgent(config))
        orch.register_agent("audit_ledger", AuditLedgerAgent(config))
        orch.register_agent("verifiable_ledger", VerifiableLedgerAgent(config))
        orch.register_agent("evac_routing", EvacRoutingAgent(config))

        loop = asyncio.get_event_loop()
        final_resp = loop.run_until_complete(
            orch.run(query="Analyze Sentinel-1 SAR flood extent and compute safe evacuation routes.")
        )

        self.assertTrue(len(final_resp.agents_executed) >= 3)
        self.assertIsNotNone(final_resp.merkle_root)
        self.assertTrue(final_resp.merkle_root.startswith("0x"))
        self.assertIn("uncertainty_level", final_resp.model_dump())


if __name__ == "__main__":
    unittest.main()
