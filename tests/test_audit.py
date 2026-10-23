import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from credit_risk_model.governance import audit
from credit_risk_model.governance.audit import (
    AuditStore,
    AuditWriteError,
    PredictionEvent,
    record_change,
    verify_file,
)


def _event(pd: float = 0.1, request_id: str = "req-1") -> PredictionEvent:
    return PredictionEvent(
        request_id=request_id, timestamp_utc="2026-01-01T00:00:00Z",
        model_name="m", model_version="1", model_status="active",
        probability_of_default=pd, score=600.0,
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


def test_records_are_hash_chained_and_verify(tmp_path):
    store = AuditStore(tmp_path)
    for i in range(3):
        store.write_prediction(_event(request_id=f"r{i}"))
    path = next((tmp_path / "decisions").glob("*.jsonl"))
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    assert lines[0]["prev_hash"] == audit.GENESIS_HASH
    assert lines[1]["prev_hash"] == lines[0]["record_hash"]
    assert verify_file(path)[0]


@pytest.mark.parametrize("tamper", ["edit", "delete", "reorder"])
def test_tampering_is_detected(tmp_path, tamper):
    store = AuditStore(tmp_path)
    for i in range(3):
        store.write_prediction(_event(pd=0.1 * (i + 1), request_id=f"r{i}"))
    path = next((tmp_path / "decisions").glob("*.jsonl"))
    lines = path.read_text().splitlines()
    if tamper == "edit":
        lines[1] = lines[1].replace('"probability_of_default":0.2', '"probability_of_default":0.01')
    elif tamper == "delete":
        del lines[1]
    else:
        lines[1], lines[2] = lines[2], lines[1]
    path.write_text("\n".join(lines) + "\n")
    ok, message = verify_file(path)
    assert not ok and path.name in message
    assert audit.main(["verify", "--root", str(tmp_path)]) == 1


def test_fail_closed_raises(tmp_path, monkeypatch):
    def broken(path):
        raise OSError("disk full")
    monkeypatch.setattr(AuditStore, "_last_hash", staticmethod(broken))
    with pytest.raises(AuditWriteError):
        AuditStore(tmp_path, fail_closed=True).write_prediction(_event())


def test_fail_open_warns_loudly(tmp_path, monkeypatch, capsys):
    def broken(path):
        raise OSError("disk full")
    monkeypatch.setattr(AuditStore, "_last_hash", staticmethod(broken))
    AuditStore(tmp_path, fail_closed=False).write_prediction(_event())
    assert "NOT persisted" in capsys.readouterr().err


def test_prune_removes_only_old_files(tmp_path):
    store = AuditStore(tmp_path)
    store.write_prediction(_event())
    old = tmp_path / "decisions" / "decisions-2000-01-01.jsonl"
    old.write_text("")
    removed = store.prune(30, today=datetime.now(timezone.utc) + timedelta(days=1))
    assert removed == [old]
    assert list((tmp_path / "decisions").glob("*.jsonl"))


def test_record_change_is_readable(tmp_path):
    store = AuditStore(tmp_path)
    record_change("model_trained", model_name="m", model_version="1", store=store, auc=0.7)
    events, total = store.read_changes()
    assert total == 1 and events[0]["change_type"] == "model_trained"
    assert events[0]["details"] == {"auc": 0.7}


def test_store_does_not_create_dirs_until_write(tmp_path):
    root = tmp_path / "audit"
    store = AuditStore(root)
    assert store.read_decisions() == ([], 0)
    assert not root.exists()
