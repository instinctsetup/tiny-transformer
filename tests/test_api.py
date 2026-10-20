"""API tests against the trained checkpoint (skipped if not trained yet)."""
import pytest
from pathlib import Path

pytestmark = pytest.mark.skipif(
    not Path("tinytransformer/tinygpt.pt").exists(),
    reason="run python train.py first")


def test_health_and_generate():
    from fastapi.testclient import TestClient
    from server import app
    c = TestClient(app)
    h = c.get("/api/health").json()
    assert h["status"] == "ok" and h["params"] > 0
    r = c.post("/api/generate", json={"seed": "The", "temperature": 0.8, "max_chars": 20})
    data = r.json()
    assert r.status_code == 200
    assert len(data["attention"]) == 2
    T = len(data["attention"][0])
    assert T <= 48 and len(data["sequence"]) == T


def test_generate_rejects_unknown_chars():
    from fastapi.testclient import TestClient
    from server import app
    c = TestClient(app)
    r = c.post("/api/generate", json={"seed": "中文", "temperature": 0.8})
    assert r.status_code == 400
