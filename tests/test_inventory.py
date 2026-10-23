import textwrap
from datetime import date
from pathlib import Path

import pytest

from credit_risk_model.governance.inventory import (
    InventoryValidationError,
    ModelEntry,
    ModelInventory,
    ModelStatus,
    can_serve,
)


@pytest.fixture
def good_inventory(tmp_path: Path) -> Path:
    p = tmp_path / "inv.yaml"
    p.write_text(textwrap.dedent("""
        models:
          - model_name: champion
            model_version: 1.0.0
            model_type: lr
            owner: team
            purpose: pd
            status: active
            approved_by: validator
            approval_date: 2025-01-01
            next_review_date: 2030-01-01
            artifact_path: artifacts/champion.joblib
    """))
    return p


def test_load_and_query(good_inventory):
    inv = ModelInventory.load(good_inventory)
    e = inv.get("champion", "1.0.0")
    assert e is not None
    assert e.status is ModelStatus.ACTIVE


def test_duplicate_rejected(tmp_path):
    p = tmp_path / "inv.yaml"
    p.write_text("""
models:
  - {model_name: a, model_version: '1', model_type: x, owner: t, purpose: p,
     status: draft, artifact_path: x}
  - {model_name: a, model_version: '1', model_type: x, owner: t, purpose: p,
     status: draft, artifact_path: x}
""")
    with pytest.raises(InventoryValidationError, match="Duplicate"):
        ModelInventory.load(p)


def test_active_requires_approval(tmp_path):
    p = tmp_path / "inv.yaml"
    p.write_text("""
models:
  - {model_name: a, model_version: '1', model_type: x, owner: t, purpose: p,
     status: active, artifact_path: x}
""")
    with pytest.raises(InventoryValidationError):
        ModelInventory.load(p)


def test_can_serve_blocks_draft():
    entry = ModelEntry(
        model_name="m", model_version="1", model_type="lr",
        owner="o", purpose="p", status=ModelStatus.DRAFT,
        artifact_path="a")
    ok, _ = can_serve(entry, allow_unreviewed=False)
    assert not ok
    ok, _ = can_serve(entry, allow_unreviewed=True)
    assert ok


def test_can_serve_overdue_review():
    entry = ModelEntry(
        model_name="m", model_version="1", model_type="lr",
        owner="o", purpose="p", status=ModelStatus.ACTIVE,
        approved_by="v", approval_date=date(2024, 1, 1),
        next_review_date=date(2024, 6, 1), artifact_path="a")
    ok, _ = can_serve(entry, as_of=date(2025, 1, 1))
    assert not ok
