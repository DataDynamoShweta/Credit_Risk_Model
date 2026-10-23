import textwrap
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from credit_risk_model.governance.inventory import (
    InventoryValidationError,
    ModelEntry,
    ModelInventory,
    ModelStatus,
    can_serve,
    record_inventory_snapshot,
)

REPO = Path(__file__).resolve().parents[1]


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


def test_unknown_fields_rejected(tmp_path):
    p = tmp_path / "inv.yaml"
    p.write_text("""
models:
  - {model_name: a, model_version: '1', model_type: x, owner: t, purpose: p,
     status: draft, artifact_path: x, monitoring_thresholds: {psi: 0.1}}
""")
    with pytest.raises(InventoryValidationError, match="monitoring_thresholds"):
        ModelInventory.load(p)


def test_risk_tier_needs_rationale_and_closed_finding_needs_evidence():
    base = {"model_name": "m", "model_version": "1", "model_type": "lr", "owner": "o", "purpose": "p",
                "status": ModelStatus.DRAFT, "artifact_path": "a"}
    with pytest.raises(ValueError, match="rationale"):
        ModelEntry(**base, risk_tier="high")
    finding = {"finding_id": "F1", "title": "t", "severity": "high", "action_owner": "o",
                   "due_date": date(2027, 1, 1), "status": "closed"}
    with pytest.raises(ValueError, match="closure_evidence"):
        ModelEntry(**base, findings=[finding])
    entry = ModelEntry(**base, findings=[{**finding, "closure_evidence": "PR #12"},
                                         {**finding, "finding_id": "F2", "status": "open"}])
    assert [f.finding_id for f in entry.open_findings()] == ["F2"]


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


def test_shipped_inventory_serves_champion_today():
    """Regression for G1: the shipped inventory must allow serving with today's date.

    This test is expected to fail once next_review_date passes; that is the point.
    """
    inventory = ModelInventory.load(REPO / "governance" / "model_inventory.yaml")
    entry = inventory.get("credit_risk_champion")
    today = datetime.now(timezone.utc).date()
    ok, reason = can_serve(entry, allow_unreviewed=False, as_of=today)
    assert ok, f"shipped champion cannot be served today: {reason}"
    challenger = inventory.get("credit_risk_challenger")
    assert not can_serve(challenger, allow_unreviewed=False, as_of=today)[0]


def test_shipped_inventory_references_only_generated_or_shipped_files():
    inventory = ModelInventory.load(REPO / "governance" / "model_inventory.yaml")
    produced = {"artifacts/champion.joblib", "artifacts/challenger.joblib",
                "artifacts/feature_list.json", "artifacts/training_metadata.json",
                "reports/validation_report.md"}
    for entry in inventory.entries:
        for ref in entry.referenced_paths().values():
            assert ref in produced, f"{entry.model_name} references {ref}, which nothing produces"


def test_inventory_snapshot_only_on_change(project):
    from credit_risk_model.config import get_settings
    from credit_risk_model.governance.audit import AuditStore
    path = get_settings().inventory_path
    assert record_inventory_snapshot(path) is True
    assert record_inventory_snapshot(path) is False
    path.write_text(path.read_text() + "\n# edited\n")
    assert record_inventory_snapshot(path) is True
    events, _ = AuditStore().read_changes()
    assert [e["change_type"] for e in events] == ["inventory_changed"] * 2
