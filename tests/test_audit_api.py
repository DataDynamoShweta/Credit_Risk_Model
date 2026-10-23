"""Audit endpoints are disabled by default and token-protected when enabled (G9)."""
import pytest
from fastapi.testclient import TestClient

from api.main import app
from credit_risk_model.config import reset_settings_cache
from tests.helpers import FEATURES_OK

ENDPOINTS = ["/audit/decisions", "/audit/changes", "/audit/monitoring",
             "/audit/adjustments", "/audit/verify"]
AUTH = {"Authorization": "Bearer t0ken"}


@pytest.fixture
def enabled(trained_project, monkeypatch):
    monkeypatch.setenv("AUDIT_API_TOKEN", "t0ken")
    reset_settings_cache()
    return trained_project


@pytest.mark.parametrize("path", ENDPOINTS)
def test_disabled_without_token_configured(project, path):
    assert TestClient(app).get(path).status_code == 403


@pytest.mark.parametrize("path", ENDPOINTS)
def test_requires_valid_token(enabled, path):
    client = TestClient(app)
    assert client.get(path).status_code == 401
    assert client.get(path, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get(path, headers=AUTH).status_code == 200


def test_decisions_and_changes_return_data(enabled):
    client = TestClient(app)
    client.post("/predict", json={"features": FEATURES_OK}, headers={"x-request-id": "aud-1"})
    decisions = client.get("/audit/decisions?request_id=aud-1", headers=AUTH).json()
    assert decisions["total"] == 1
    changes = client.get("/audit/changes", headers=AUTH).json()
    types = {e["change_type"] for e in changes["events"]}
    assert "model_trained" in types  # written by the training pipeline
    assert client.get("/audit/verify", headers=AUTH).json()["ok"] is True


def test_page_size_capped(enabled, monkeypatch):
    monkeypatch.setenv("AUDIT_MAX_PAGE_SIZE", "5")
    reset_settings_cache()
    body = TestClient(app).get("/audit/decisions?limit=1000", headers=AUTH).json()
    assert body["limit"] == 5


def test_bad_date_rejected(enabled):
    r = TestClient(app).get("/audit/decisions?date=not-a-date", headers=AUTH)
    assert r.status_code == 422
