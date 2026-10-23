"""Persistent YAML model inventory, strict schema validation and the serving gate.

``governance/model_inventory.yaml`` is the single source of model identity (name,
version, status) for the API, model card, audit events and reports. Unknown fields are
rejected so typos cannot silently disable a control.
"""
from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class ModelStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    UNDER_REVIEW = "under_review"
    SUSPENDED = "suspended"
    RETIRED = "retired"


class RiskTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FindingStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


# Statuses that ALLOW_UNREVIEWED_MODEL may unlock for local demos; suspended/retired never serve.
_OVERRIDABLE = {ModelStatus.DRAFT, ModelStatus.UNDER_REVIEW}


class InventoryValidationError(ValueError):
    """Raised when governance/model_inventory.yaml is missing, malformed or inconsistent."""


def _today() -> date:
    return datetime.now(timezone.utc).date()


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_id: str
    title: str
    severity: Severity
    action_owner: str
    due_date: date
    status: FindingStatus = FindingStatus.OPEN
    closure_evidence: str | None = None

    @model_validator(mode="after")
    def _closed_needs_evidence(self):
        if self.status is FindingStatus.CLOSED and not self.closure_evidence:
            raise ValueError(f"finding {self.finding_id}: closed findings need closure_evidence")
        return self


class ModelEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_name: str
    model_version: str
    model_type: str
    owner: str
    purpose: str
    status: ModelStatus
    artifact_path: str
    approved_by: str | None = None
    approval_date: date | None = None
    next_review_date: date | None = None
    last_validation_date: date | None = None
    risk_tier: RiskTier | None = None
    risk_tier_rationale: str | None = None
    intended_use: str | None = None
    limitations: list[str] = Field(default_factory=list)
    use_restrictions: list[str] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    training_data_reference: str | None = None
    feature_list_reference: str | None = None
    metadata_reference: str | None = None
    artifact_sha256: str | None = None
    mlflow_model_uri: str | None = None
    validation_report: str | None = None

    @field_validator("model_version", mode="before")
    @classmethod
    def _version_as_str(cls, value):
        return str(value)

    @model_validator(mode="after")
    def _consistency(self):
        if self.status is ModelStatus.ACTIVE and not (self.approved_by and self.approval_date):
            raise ValueError(f"{self.model_name} {self.model_version}: active models require "
                             "approved_by and approval_date")
        if self.risk_tier and not self.risk_tier_rationale:
            raise ValueError(f"{self.model_name}: risk_tier needs risk_tier_rationale")
        return self

    def open_findings(self) -> list[Finding]:
        return [f for f in self.findings if f.status is FindingStatus.OPEN]

    def referenced_paths(self) -> dict[str, str]:
        refs = {"artifact_path": self.artifact_path,
                "feature_list_reference": self.feature_list_reference,
                "metadata_reference": self.metadata_reference,
                "validation_report": self.validation_report}
        return {k: v for k, v in refs.items() if v}


class ModelInventory:
    def __init__(self, entries: list[ModelEntry], path: Path | None = None):
        self.entries = entries
        self.path = path

    @classmethod
    def load(cls, path) -> ModelInventory:
        path = Path(path)
        if not path.exists():
            raise InventoryValidationError(f"Inventory not found: {path}")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise InventoryValidationError(f"Inventory is not valid YAML: {exc}") from exc
        unknown = set(raw) - {"models"}
        if unknown:
            raise InventoryValidationError(f"Unknown top-level keys: {sorted(unknown)}")
        try:
            entries = [ModelEntry(**item) for item in raw.get("models") or []]
        except (ValidationError, TypeError) as exc:
            raise InventoryValidationError(str(exc)) from exc
        seen = set()
        for e in entries:
            key = (e.model_name, e.model_version)
            if key in seen:
                raise InventoryValidationError(f"Duplicate inventory entry: {key}")
            seen.add(key)
        return cls(entries, path)

    def get(self, model_name: str, model_version: str | None = None) -> ModelEntry | None:
        matches = [e for e in self.entries if e.model_name == model_name
                   and (model_version is None or e.model_version == model_version)]
        return matches[-1] if matches else None

    def missing_references(self, root: Path) -> list[str]:
        """Referenced files that do not exist (relative paths resolved against ``root``)."""
        missing = []
        for entry in self.entries:
            for field_name, ref in entry.referenced_paths().items():
                path = Path(ref) if Path(ref).is_absolute() else Path(root) / ref
                if not path.exists():
                    missing.append(f"{entry.model_name}.{field_name}: {ref}")
        return missing


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def can_serve(entry: ModelEntry | None, *, allow_unreviewed: bool = False,
              as_of: date | None = None) -> tuple[bool, str]:
    """Return (allowed, reason) for serving a model under its inventory record."""
    if entry is None:
        return False, "model has no inventory entry"
    as_of = as_of or _today()
    if entry.status is ModelStatus.ACTIVE:
        if entry.next_review_date and as_of > entry.next_review_date:
            if allow_unreviewed:
                return True, f"review overdue since {entry.next_review_date}; override enabled"
            return False, f"review overdue since {entry.next_review_date}"
        return True, "active"
    if allow_unreviewed and entry.status in _OVERRIDABLE:
        return True, f"status {entry.status.value}; override enabled"
    return False, f"status {entry.status.value} is not servable"


def record_inventory_snapshot(path: Path, *, actor: str = "pipeline", store=None) -> bool:
    """Write an ``inventory_changed`` change event if the inventory file changed.

    Compares the file's SHA-256 with the last snapshot in the change log. Returns True if
    a new event was written.
    """
    from credit_risk_model.governance.audit import AuditStore, record_change
    store = store or AuditStore()
    digest = file_sha256(path)
    events, _ = store.read_changes(limit=10_000)
    last = next((e for e in events if e.get("change_type") == "inventory_changed"), None)
    if last and last.get("details", {}).get("inventory_sha256") == digest:
        return False
    record_change("inventory_changed", actor=actor, store=store,
                  inventory_path=str(path), inventory_sha256=digest,
                  previous_sha256=(last or {}).get("details", {}).get("inventory_sha256"))
    return True
