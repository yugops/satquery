"""
SatQuery AI — Pipeline Smoke Test & Demo

Runs all major query scenarios through LeadOrchestratorAgent to verify
agent dispatch, DAG execution, SQLite ledger auditing, and answer synthesis.
"""

from core.orchestrator import LeadOrchestratorAgent


def main():
    orchestrator = LeadOrchestratorAgent()

    test_cases = [
        ("VQA (Single Scene)", "How many buildings are visible in this satellite scene?", "scene.png", None),
        ("Visual Grounding", "Locate and draw bounding boxes on airplanes", "scene.png", None),
        ("Change Detection", "Compare before and after images to detect change", "t1_pre.png", "t2_post.png"),
        ("SAR Cloud Penetration", "Analyze Sentinel-1 SAR VV and VH radar backscatter for flood detection", "scene.png", None),
        ("Disaster Assessment", "Assess flood damage in this coastal area", "pre_flood.png", "post_flood.png"),
        ("GIS Land Cover", "Identify land cover and urban forest areas", "scene.png", None),
    ]

    print("=" * 70)
    print(" SATQUERY AI — MULTI-AGENT ORCHESTRATION PIPELINE TEST")
    print("=" * 70)

    for idx, (category, query, img1, img2) in enumerate(test_cases, 1):
        print(f"\n[Test {idx}/{len(test_cases)}] Scenario: {category}")
        print(f"  Query: \"{query}\"")
        
        response = orchestrator.run_sync(
            query=query,
            image_path=img1,
            image_path_t2=img2,
        )

        print(f"  Session ID: {response.session_id}")
        print(f"  Intent:     {response.intent.intent_type.value} (confidence: {response.intent.confidence:.2f})")
        print(f"  Agents:     {[r.agent for r in response.agent_responses]}")
        print(f"  Latency:    {response.total_execution_ms:.2f} ms")
        print(f"  Synthesized Answer:\n    {response.synthesized_answer.replace(chr(10), chr(10) + '    ')}")
        print("-" * 70)

    print("\nAll pipeline tests completed successfully!")


if __name__ == "__main__":
    main()
