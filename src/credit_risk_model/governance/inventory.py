"""Model inventory record and risk-tier assignment."""
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator


@dataclass
class ModelInventoryRecord:
    model_id: str
    name: str
    owner: str
    purpose: str
    model_type: str
    status: str = "development"
    risk_tier: str = "high"
    version: str = "0.1.0"
    limitations: str = "Not approved for production use"
    last_validation_date: str | None = None

    def to_dict(self):
        result = asdict(self)
        result["record_updated_at"] = datetime.now(timezone.utc).isoformat()
        return result

def assign_risk_tier(impact: str, complexity: str, reliance: str = "high") -> str:
    """Transparent illustrative tiering heuristic; replace with approved policy."""
    score = sum(x.lower() in {"high", "material", "critical"} for x in [impact, complexity, reliance])
    return "high" if score >= 2 else ("medium" if score == 1 else "low")


# --- Persistent YAML inventory and serving gate ---------------------------------------------

class ModelStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    UNDER_REVIEW = "under_review"
    SUSPENDED = "suspended"
    RETIRED = "retired"


# Statuses that ALLOW_UNREVIEWED_MODEL may unlock for local demos; suspended/retired never serve.
_OVERRIDABLE = {ModelStatus.DRAFT, ModelStatus.UNDER_REVIEW}


class InventoryValidationError(ValueError):
    """Raised when governance/model_inventory.yaml is missing, malformed or inconsistent."""


class ModelEntry(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())

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
    validation_report: str | None = None

    @field_validator("model_version", mode="before")
    @classmethod
    def _version_as_str(cls, value):
        return str(value)

    @model_validator(mode="after")
    def _active_requires_approval(self):
        if self.status is ModelStatus.ACTIVE and not (self.approved_by and self.approval_date):
            raise ValueError(f"{self.model_name} {self.model_version}: active models require "
                             "approved_by and approval_date")
        return self


class ModelInventory:
    def __init__(self, entries: list[ModelEntry]):
        self.entries = entries

    @classmethod
    def load(cls, path) -> "ModelInventory":
        path = Path(path)
        if not path.exists():
            raise InventoryValidationError(f"Inventory not found: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        try:
            entries = [ModelEntry(**item) for item in raw.get("models", [])]
        except ValidationError as exc:
            raise InventoryValidationError(str(exc)) from exc
        seen = set()
        for e in entries:
            key = (e.model_name, e.model_version)
            if key in seen:
                raise InventoryValidationError(f"Duplicate inventory entry: {key}")
            seen.add(key)
        return cls(entries)

    def get(self, model_name: str, model_version: str | None = None) -> ModelEntry | None:
        matches = [e for e in self.entries if e.model_name == model_name
                   and (model_version is None or e.model_version == model_version)]
        return matches[-1] if matches else None


def can_serve(entry: ModelEntry | None, *, allow_unreviewed: bool = False,
              as_of: date | None = None) -> tuple[bool, str]:
    """Return (allowed, reason) for serving a model under its inventory record."""
    if entry is None:
        return False, "model has no inventory entry"
    as_of = as_of or datetime.now(timezone.utc).date()
    if entry.status is ModelStatus.ACTIVE:
        if entry.next_review_date and as_of > entry.next_review_date:
            if allow_unreviewed:
                return True, f"review overdue since {entry.next_review_date}; override enabled"
            return False, f"review overdue since {entry.next_review_date}"
        return True, "active"
    if allow_unreviewed and entry.status in _OVERRIDABLE:
        return True, f"status {entry.status.value}; override enabled"
    return False, f"status {entry.status.value} is not servable"
