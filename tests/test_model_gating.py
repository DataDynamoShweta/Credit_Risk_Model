from datetime import date

import pytest

from credit_risk_model.governance.inventory import ModelEntry, ModelStatus, can_serve


def _entry(status, next_review=None):
    return ModelEntry(
        model_name="m", model_version="1", model_type="lr",
        owner="o", purpose="p", status=status,
        approved_by="v" if status is ModelStatus.ACTIVE else None,
        approval_date=date(2024, 1, 1) if status is ModelStatus.ACTIVE else None,
        next_review_date=next_review, artifact_path="a")


@pytest.mark.parametrize("status", [
    ModelStatus.DRAFT, ModelStatus.SUSPENDED,
    ModelStatus.RETIRED, ModelStatus.UNDER_REVIEW])
def test_blocked_statuses(status):
    ok, _ = can_serve(_entry(status), allow_unreviewed=False)
    assert not ok


def test_active_passes():
    ok, _ = can_serve(_entry(ModelStatus.ACTIVE, date(2030, 1, 1)))
    assert ok


def test_missing_entry_blocks():
    ok, _ = can_serve(None)
    assert not ok
