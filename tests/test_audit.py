from pathlib import Path

from credit_risk_model.governance.audit import AuditStore, PredictionEvent


def _event(pd: float = 0.1) -> PredictionEvent:
    return PredictionEvent(
        request_id="req-1", timestamp_utc="2026-01-01T00:00:00Z",
        model_name="m", model_version="1", model_status="active",
        probability_of_default=pd, score=600.0,
        decision_threshold=0.2, decision="approve",
        input_feature_names=["a", "b"], latency_ms=5.0, environment="test")


def test_write_and_read(tmp_path: Path):
    store = AuditStore(tmp_path)
    store.write_prediction(_event())
    rows, total = store.read_decisions(limit=10, offset=0)
    assert total == 1
    assert rows[0]["request_id"] == "req-1"


def test_filter_by_version(tmp_path: Path):
    store = AuditStore(tmp_path)
    store.write_prediction(_event())
    _, total = store.read_decisions(model_version="2")
    assert total == 0


def test_no_pii_in_event():
    payload = _event().to_json()
    assert "loan_amnt" not in payload
