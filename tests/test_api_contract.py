"""API hardening and feature-contract regression tests (G4.5, G4.6, G9, G10)."""
import json

import pytest
from fastapi.testclient import TestClient

from api.main import app
from credit_risk_model.config import reset_settings_cache
from tests.helpers import FEATURES_OK


@pytest.fixture
def client(trained_project):
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _decisions(root):
    files = sorted((root / "governance" / "audit_trail" / "decisions").glob("*.jsonl"))
    return [json.loads(line) for f in files for line in f.read_text().splitlines() if line]


def test_valid_payload_is_scored(client):
    r = client.post("/predict", json={"features": FEATURES_OK})
    assert r.status_code == 200
    body = r.json()
    assert 0 < body["probability_of_default"] < 1
    assert body["model_name"] == "credit_risk_champion"
    assert body["model_version"] == "1.3.0"


def test_health_reports_loaded_model(client):
    body = client.get("/health").json()
    assert body["model_loaded"] is True, body["reason"]


def test_empty_payload_rejected(client):
    r = client.post("/predict", json={"features": {}})
    assert r.status_code == 422


def test_missing_feature_rejected(client):
    features = {k: v for k, v in FEATURES_OK.items() if k != "dti"}
    r = client.post("/predict", json={"features": features})
    assert r.status_code == 422
    assert "dti" in json.dumps(r.json())


def test_unexpected_feature_rejected(client):
    r = client.post("/predict", json={"features": {**FEATURES_OK, "favourite_colour": "blue"}})
    assert r.status_code == 422
    assert "unexpected" in json.dumps(r.json())


@pytest.mark.parametrize("field,value", [("int_rate", "high"), ("home_ownership", 3),
                                         ("emp_length", True)])
def test_wrong_feature_type_rejected(client, field, value):
    r = client.post("/predict", json={"features": {**FEATURES_OK, field: value}})
    assert r.status_code == 422


def test_malformed_payload_not_echoed(client):
    secret = "APPLICANT-SECRET-12345"
    for body in ({"features": secret}, {"features": FEATURES_OK, "extra": secret}):
        r = client.post("/predict", json=body)
        assert r.status_code == 422
        assert secret not in r.text
    r = client.post("/predict", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_oversized_request_rejected(client, monkeypatch):
    monkeypatch.setenv("MAX_REQUEST_BYTES", "512")
    reset_settings_cache()
    r = client.post("/predict", json={"features": {**FEATURES_OK, "pad": "x" * 2000}})
    assert r.status_code == 413


def test_invalid_payload_request_id_rejected(client):
    r = client.post("/predict", json={"features": FEATURES_OK, "request_id": "bad id\n"})
    assert r.status_code == 422


def test_malformed_header_request_id_replaced(client):
    r = client.post("/predict", json={"features": FEATURES_OK},
                    headers={"x-request-id": "evil\r\ninjected"})
    assert r.status_code == 200
    rid = r.headers["x-request-id"]
    assert "injected" not in rid and rid == r.json()["request_id"]


def test_one_request_id_everywhere(client, trained_project):
    r = client.post("/predict", json={"features": FEATURES_OK},
                    headers={"x-request-id": "corr-123"})
    assert r.headers["x-request-id"] == r.json()["request_id"] == "corr-123"
    assert _decisions(trained_project)[-1]["request_id"] == "corr-123"
    r = client.post("/predict", json={"features": FEATURES_OK, "request_id": "payload-9"})
    assert r.headers["x-request-id"] == r.json()["request_id"] == "payload-9"
    assert _decisions(trained_project)[-1]["request_id"] == "payload-9"


def test_audit_event_has_no_feature_values_or_decision(client, trained_project):
    client.post("/predict", json={"features": FEATURES_OK},
                headers={"x-client-id": "demo-client"})
    event = _decisions(trained_project)[-1]
    assert event["api_user_or_client_id"] == "demo-client"
    assert "decision" not in event and "15000" not in json.dumps(event)
    assert event["record_hash"] and event["prev_hash"]


def test_audit_failure_withholds_prediction(client, monkeypatch):
    from credit_risk_model.governance import audit

    def broken(path):
        raise OSError("disk full")
    monkeypatch.setattr(audit.AuditStore, "_last_hash", staticmethod(broken))
    r = client.post("/predict", json={"features": FEATURES_OK})
    assert r.status_code == 503
    assert "probability_of_default" not in r.text


def test_unhandled_error_returns_safe_500_with_request_id(client, monkeypatch):
    from api import main

    def boom():
        raise RuntimeError("internal secret detail")
    monkeypatch.setattr(main, "load_state", boom)
    r = client.get("/health", headers={"x-request-id": "err-1"})
    assert r.status_code == 500
    assert r.json() == {"detail": "Internal server error", "request_id": "err-1"}
    assert "secret" not in r.text


def test_predict_token_enforced_when_configured(client, monkeypatch):
    monkeypatch.setenv("PREDICT_API_TOKEN", "p-token")
    reset_settings_cache()
    assert client.post("/predict", json={"features": FEATURES_OK}).status_code == 401
    r = client.post("/predict", json={"features": FEATURES_OK},
                    headers={"Authorization": "Bearer p-token"})
    assert r.status_code == 200


def test_model_card_is_dynamic(client):
    card = client.get("/model-card").json()
    assert card["status"] == "not-approved-for-production"
    assert card["training"]["class_weight"] is None
    assert {f["name"] for f in card["features"]} == set(FEATURES_OK)
    assert card["inventory"]["use_restrictions"]
    assert card["inventory"]["risk_tier"] == "high"
