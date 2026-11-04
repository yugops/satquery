"""
Unit & Integration Tests for SatQuery AI FastAPI Server
"""

import io
import pytest
from fastapi.testclient import TestClient
from main import app
from PIL import Image


@pytest.fixture
def client():
    return TestClient(app)


def test_system_status(client):
    response = client.get("/api/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "v1.0.0-part1"
    assert data["agentController"] == "READY"
    assert len(data["agents"]) == 7
    assert len(data["services"]) == 6
    assert "compute" in data


def test_models_endpoint(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    models = response.json()
    assert len(models) == 6
    tasks = [m["task"] for m in models]
    assert "vqa" in tasks
    assert "grounding" in tasks
    assert "change-detection" in tasks


def test_datasets_endpoint(client):
    response = client.get("/api/datasets")
    assert response.status_code == 200
    datasets = response.json()
    assert len(datasets) == 4
    ids = [d["id"] for d in datasets]
    assert "rsvqa" in ids
    assert "levir-cd" in ids


def test_upload_and_validate_and_analyze(client):
    # 1. Create a dummy test image
    img = Image.new("RGB", (256, 256), color=(50, 100, 150))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    # 2. Upload image
    upload_resp = client.post(
        "/api/upload",
        files={"file": ("test_scene.png", buf, "image/png")},
        data={"role": "primary"},
    )
    assert upload_resp.status_code == 200
    img_data = upload_resp.json()
    assert img_data["status"] == "ready"
    assert img_data["metadata"]["width"] == 256
    assert img_data["metadata"]["height"] == 256
    img_id = img_data["id"]

    # 3. Validate
    val_resp = client.post(
        "/api/validate",
        json={"mode": "single", "imageIds": [img_id]},
    )
    assert val_resp.status_code == 200
    val_data = val_resp.json()
    assert val_data["ok"] is True
    assert len(val_data["checks"]) >= 5

    # 4. Analyze
    analyze_resp = client.post(
        "/api/analyze",
        json={
            "mode": "single",
            "query": "How many buildings are visible in this satellite scene?",
            "imageIds": [img_id],
            "demo": False,
        },
    )
    assert analyze_resp.status_code == 200
    res = analyze_resp.json()
    assert res["status"] == "completed"
    assert res["query"] == "How many buildings are visible in this satellite scene?"
    assert res["answer"] is not None
    assert len(res["steps"]) == 4
    analysis_id = res["id"]

    # 5. Get Analysis Detail
    detail_resp = client.get(f"/api/analysis/{analysis_id}")
    assert detail_resp.status_code == 200
    assert detail_resp.json()["id"] == analysis_id

    # 6. Get Analysis Trace
    trace_resp = client.get(f"/api/analysis/{analysis_id}/trace")
    assert trace_resp.status_code == 200
    assert len(trace_resp.json()["entries"]) == 4

    # 7. Get Analysis Report
    report_json_resp = client.get(f"/api/analysis/{analysis_id}/report?format=json")
    assert report_json_resp.status_code == 200
    report_html_resp = client.get(f"/api/analysis/{analysis_id}/report?format=pdf")
    assert report_html_resp.status_code == 200

    # 8. History
    history_resp = client.get("/api/analysis")
    assert history_resp.status_code == 200
    assert len(history_resp.json()) >= 1
