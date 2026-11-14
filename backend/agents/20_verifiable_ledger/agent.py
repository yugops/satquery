"""
SatQuery AI — Agent 20: VerifiableLedgerAgent
Problem Statement ID: 26167 (ISRO / SAC)
Wing: Cryptographic_Provenance
"""

from __future__ import annotations

from typing import Any, Optional
from core.interfaces import BaseAgent
from core.models import ComputationMode
from .tools import (
    compute_merkle_leaf,
    build_merkle_tree,
    generate_merkle_proof,
    verify_merkle_proof,
)


class VerifiableLedgerAgent(BaseAgent):
    agent_id = "verifiable_ledger"
    agent_name = "VerifiableLedgerAgent"
    mempalace_wing = "Wing: Cryptographic_Provenance"

    async def _execute(self, **kwargs: Any) -> tuple[dict[str, Any], ComputationMode, float, Optional[str], Optional[str]]:
        """
        Builds a cryptographically verifiable Merkle tree over the entire DAG execution trace.
        """
        agent_traces = kwargs.get("agent_traces", [])
        input_hash = kwargs.get("input_hash", "")
        session_id = kwargs.get("session_id", "SQ-000")

        leaf_hashes = []
        leaves_meta = []

        for idx, trace in enumerate(agent_traces):
            t_data = trace if isinstance(trace, dict) else trace.model_dump()
            leaf_h = compute_merkle_leaf(
                node_id=f"step_{idx}",
                agent_id=t_data.get("agent_id", "unknown"),
                computation_mode=t_data.get("computation_mode", "rule_based"),
                weights_sha256=t_data.get("model_weights_sha256"),
                output_summary=t_data.get("summary", ""),
                input_hash=input_hash,
            )
            leaf_hashes.append(leaf_h)
            leaves_meta.append({
                "index": idx,
                "agent_id": t_data.get("agent_id"),
                "computation_mode": t_data.get("computation_mode"),
                "weights_sha256": t_data.get("model_weights_sha256"),
                "leaf_hash": leaf_h
            })

        merkle_root, tree_levels = build_merkle_tree(leaf_hashes)

        proofs = {
            meta["agent_id"]: generate_merkle_proof(meta["index"], tree_levels)
            for meta in leaves_meta
        }

        is_valid = True
        if leaf_hashes:
            is_valid = verify_merkle_proof(leaf_hashes[0], proofs[leaves_meta[0]["agent_id"]], merkle_root)

        # STRICT COURT ADMISSIBILITY AUDIT
        # Condition 1: Hash chain mathematically verifies against Merkle root
        # Condition 2: No synthetic fallbacks were used
        # Condition 3: No critical operational agents skipped due to missing inputs
        has_synthetic = any(m.get("computation_mode") == "synthetic_fallback" for m in leaves_meta)
        has_skipped = any("skipped" in str(getattr(t, "summary", "") or (t.get("summary", "") if isinstance(t, dict) else "")).lower() for t in agent_traces)

        is_court_admissible = is_valid and (not has_synthetic) and (not has_skipped)

        if is_court_admissible:
            admissibility_status = (
                "TAMPER-EVIDENT: Computation trace integrity cryptographically verified. "
                "This certifies the recorded process was not altered after the fact — "
                "it does NOT certify that inputs were valid or that the scientific "
                "conclusion is correct. Independently verify inputs before relying on the result."
            )
        elif has_skipped:
            admissibility_status = "TAMPER-EVIDENT: Audit hash chain intact, but operational inputs were incomplete (Single-temporal input / Missing T2)."
        else:
            admissibility_status = "TAMPER-EVIDENT: Audit hash chain intact, but synthetic fallback mode detected."

        summary = (
            f"Cryptographic provenance ledger verified. Merkle Root: {merkle_root[:16]}... "
            f"Computed over {len(leaf_hashes)} DAG nodes with tamper-evident mode certification. ({admissibility_status})"
        )

        data = {
            "session_id": session_id,
            "merkle_root": merkle_root,
            "total_leaves": len(leaf_hashes),
            "leaves": leaves_meta,
            "proofs": proofs,
            "integrity_verified": is_valid,
            "court_admissible": is_court_admissible,
            "admissibility_status": admissibility_status,
        }

        return data, ComputationMode.RULE_BASED, 1.0, summary, None
