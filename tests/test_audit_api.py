from fastapi.testclient import TestClient

from api.main import app


def test_decisions_endpoint():
    r = TestClient(app).get("/audit/decisions?limit=5")
    assert r.status_code == 200
    assert "events" in r.json()


def test_changes_endpoint():
    r = TestClient(app).get("/audit/changes")
    assert r.status_code == 200


def test_monitoring_endpoint():
    r = TestClient(app).get("/audit/monitoring")
    assert r.status_code == 200
