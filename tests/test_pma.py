import pytest

from credit_risk_model.governance.pma import PMAEvent, PMAEventType, PMALog, PMAStatus


def test_append_and_read(tmp_path):
    log = PMALog(tmp_path / "pma.csv")
    evt = PMAEvent(
        model_name="m", model_version="1",
        event_type=PMAEventType.MONITORING_ALERT,
        trigger="PSI>0.25", description="drift", owner="team")
    log.append(evt)
    rows = log._read_rows()
    assert len(rows) == 1
    assert rows[0]["status"] == PMAStatus.OPEN.value


def test_duplicate_rejected(tmp_path):
    log = PMALog(tmp_path / "pma.csv")
    evt = PMAEvent(
        model_name="m", model_version="1",
        event_type=PMAEventType.MANUAL_REVIEW,
        trigger="t", description="d", owner="o", pma_id="PMA-FIXED")
    log.append(evt)
    with pytest.raises(ValueError, match="Duplicate"):
        log.append(evt)
