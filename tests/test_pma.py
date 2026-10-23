"""Post-model adjustment register (G6)."""
from datetime import date

import pytest
from pydantic import ValidationError

from credit_risk_model.governance.audit import AuditStore
from credit_risk_model.governance.pma import COLUMNS, Adjustment, PMARegister, PMARegisterError


def _adj(**overrides) -> Adjustment:
    values = {"model_name": "m", "model_version": "1", "adjustment_type": "overlay",
                  "direction": "increase", "magnitude": 0.005, "unit": "pd_absolute",
                  "rationale": "Economic downturn overlay", "effective_date": date(2026, 1, 1),
                  "review_date": date(2026, 6, 30), "expiry_date": date(2026, 12, 31),
                  "owner": "dev-team", "approver": "validator", "approval_state": "approved",
                  "evidence_reference": "memo-001"}
    values.update(overrides)
    return Adjustment(**values)


def test_schema_requires_core_fields():
    with pytest.raises(ValidationError):
        Adjustment(model_name="m")


@pytest.mark.parametrize("dates", [
    {"effective_date": date(2026, 7, 1), "review_date": date(2026, 6, 1)},
    {"review_date": date(2027, 6, 1), "expiry_date": date(2027, 1, 1)},
])
def test_date_order_enforced(dates):
    with pytest.raises(ValidationError, match="effective_date <= review_date <= expiry_date"):
        _adj(**dates)


def test_approved_needs_independent_approver():
    with pytest.raises(ValidationError, match="needs an approver"):
        _adj(approver=None)
    with pytest.raises(ValidationError, match="independent"):
        _adj(approver="dev-team")
    assert _adj(approval_state="proposed", approver=None).approver is None


def test_negative_magnitude_rejected():
    with pytest.raises(ValidationError):
        _adj(magnitude=-1)


def test_append_load_and_duplicates(tmp_path):
    register = PMARegister(tmp_path / "pma.csv")
    adj = _adj(pma_id="PMA-FIXED")
    register.append(adj, record_change_event=False)
    assert (tmp_path / "pma.csv").read_text().splitlines()[0] == ",".join(COLUMNS)
    loaded = register.load()
    assert len(loaded) == 1 and loaded[0].magnitude == 0.005
    with pytest.raises(PMARegisterError, match="Duplicate"):
        register.append(adj, record_change_event=False)


def test_active_overdue_and_expired(tmp_path):
    register = PMARegister(tmp_path / "pma.csv")
    register.append(_adj(pma_id="A"), record_change_event=False)
    register.append(_adj(pma_id="P", approval_state="proposed", approver=None),
                    record_change_event=False)
    assert [a.pma_id for a in register.active(date(2026, 3, 1))] == ["A"]
    assert register.overdue_reviews(date(2026, 3, 1)) == []
    assert [a.pma_id for a in register.overdue_reviews(date(2026, 8, 1))] == ["A"]
    assert [a.pma_id for a in register.expired(date(2027, 1, 1))] == ["A"]
    assert register.active(date(2027, 1, 1)) == []
    summary = register.summary(date(2026, 8, 1))
    assert summary == {"as_of": "2026-08-01", "total": 2, "active": ["A"],
                       "overdue_review": ["A"], "expired": []}


def test_lfs_pointer_is_rejected_not_corrupted(tmp_path):
    path = tmp_path / "pma.csv"
    pointer = "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 164\n"
    path.write_text(pointer)
    with pytest.raises(PMARegisterError, match="LFS pointer"):
        PMARegister(path).append(_adj(), record_change_event=False)
    assert path.read_text() == pointer


def test_append_writes_change_event(project):
    PMARegister().append(_adj(pma_id="PMA-EVT"))
    events, _ = AuditStore().read_changes()
    assert events[0]["change_type"] == "post_model_adjustment_added"
    assert events[0]["details"]["pma_id"] == "PMA-EVT"


def test_shipped_register_is_plain_csv_with_expected_header():
    from pathlib import Path
    REPO = Path(__file__).resolve().parents[1]
    text = (REPO / "governance" / "pma_log.csv").read_text(encoding="utf-8")
    assert not text.startswith("version https://git-lfs")
    assert text.splitlines()[0] == ",".join(COLUMNS)
    PMARegister(REPO / "governance" / "pma_log.csv").load()
