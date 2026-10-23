"""Post-model adjustment (PMA) register: overlays applied on top of model outputs.

Each adjustment records what is changed (type, direction, magnitude, unit), why
(rationale, evidence), when (effective, review and expiry dates) and who (owner, approver,
approval state). Controls implemented here:

* dates must satisfy effective_date <= review_date <= expiry_date;
* an ``approved`` adjustment needs an approver who is not the owner;
* :meth:`PMARegister.active`, :meth:`overdue_reviews` and :meth:`expired` report status
  as of a given date, so expired or unreviewed overlays are visible.

The register is a plain CSV (``governance/pma_log.csv``) and must not be stored in Git LFS.
Model-change events (training, validation, inventory edits) are a different artefact and
live in the hash-chained change log, see :func:`credit_risk_model.governance.audit.record_change`.
"""
from __future__ import annotations

import csv
import uuid
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from credit_risk_model.config import get_settings


class AdjustmentType(str, Enum):
    OVERLAY = "overlay"
    SCALAR = "scalar"
    FLOOR = "floor"
    CAP = "cap"
    SEGMENT_OVERRIDE = "segment_override"


class Direction(str, Enum):
    INCREASE = "increase"
    DECREASE = "decrease"


class Unit(str, Enum):
    PD_ABSOLUTE = "pd_absolute"          # e.g. +0.005 added to PD
    PD_RELATIVE_PCT = "pd_relative_pct"  # e.g. +10 (%) relative uplift
    SCORE_POINTS = "score_points"


class ApprovalState(str, Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


COLUMNS = [
    "pma_id", "model_name", "model_version", "adjustment_type", "direction", "magnitude",
    "unit", "rationale", "effective_date", "review_date", "expiry_date", "owner",
    "approver", "approval_state", "evidence_reference", "created_at_utc",
]


class PMARegisterError(ValueError):
    pass


class Adjustment(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=(), use_enum_values=False)

    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    adjustment_type: AdjustmentType
    direction: Direction
    magnitude: float = Field(ge=0)
    unit: Unit
    rationale: str = Field(min_length=1)
    effective_date: date
    review_date: date
    expiry_date: date
    owner: str = Field(min_length=1)
    approver: str | None = None
    approval_state: ApprovalState = ApprovalState.PROPOSED
    evidence_reference: str = Field(min_length=1)
    pma_id: str = Field(default_factory=lambda: f"PMA-{uuid.uuid4().hex[:12].upper()}")
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @model_validator(mode="before")
    @classmethod
    def _blank_to_none(cls, values):
        if isinstance(values, dict) and values.get("approver") == "":
            values = {**values, "approver": None}
        return values

    @model_validator(mode="after")
    def _controls(self):
        if not self.effective_date <= self.review_date <= self.expiry_date:
            raise ValueError("dates must satisfy effective_date <= review_date <= expiry_date")
        if self.approval_state is ApprovalState.APPROVED:
            if not self.approver:
                raise ValueError("an approved adjustment needs an approver")
            if self.approver == self.owner:
                raise ValueError("approver must be independent of the adjustment owner")
        return self

    def is_active(self, as_of: date) -> bool:
        return (self.approval_state is ApprovalState.APPROVED
                and self.effective_date <= as_of < self.expiry_date)

    def as_row(self) -> dict[str, str]:
        row = self.model_dump(mode="json")
        row["approver"] = row["approver"] or ""
        return {c: str(row[c]) for c in COLUMNS}


class PMARegister:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else get_settings().pma_register_path

    def _check_header(self) -> None:
        text = self.path.read_text(encoding="utf-8")
        if text.startswith("version https://git-lfs"):
            raise PMARegisterError(f"{self.path} is a Git LFS pointer, not the register; "
                                   "governance files must not be stored in LFS")
        first = text.splitlines()[0] if text.strip() else ""
        if first.split(",") != COLUMNS:
            raise PMARegisterError(f"{self.path} has an unexpected header; expected "
                                   + ",".join(COLUMNS))

    def _ensure_file(self) -> None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=COLUMNS).writeheader()
        self._check_header()

    def load(self) -> list[Adjustment]:
        if not self.path.exists():
            return []
        self._check_header()
        with self.path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        try:
            return [Adjustment(**row) for row in rows]
        except ValidationError as exc:
            raise PMARegisterError(f"invalid row in {self.path}: {exc}") from exc

    def append(self, adjustment: Adjustment, *, record_change_event: bool = True) -> str:
        self._ensure_file()
        if adjustment.pma_id in {a.pma_id for a in self.load()}:
            raise PMARegisterError(f"Duplicate pma_id: {adjustment.pma_id}")
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=COLUMNS).writerow(adjustment.as_row())
        if record_change_event:
            from credit_risk_model.governance.audit import record_change
            record_change("post_model_adjustment_added", model_name=adjustment.model_name,
                          model_version=adjustment.model_version, actor=adjustment.owner,
                          pma_id=adjustment.pma_id,
                          approval_state=adjustment.approval_state.value)
        return adjustment.pma_id

    def active(self, as_of: date | None = None) -> list[Adjustment]:
        as_of = as_of or date.today()
        return [a for a in self.load() if a.is_active(as_of)]

    def overdue_reviews(self, as_of: date | None = None) -> list[Adjustment]:
        as_of = as_of or date.today()
        return [a for a in self.active(as_of) if a.review_date < as_of]

    def expired(self, as_of: date | None = None) -> list[Adjustment]:
        as_of = as_of or date.today()
        return [a for a in self.load()
                if a.approval_state is ApprovalState.APPROVED and a.expiry_date <= as_of]

    def summary(self, as_of: date | None = None) -> dict:
        as_of = as_of or date.today()
        return {"as_of": str(as_of), "total": len(self.load()),
                "active": [a.pma_id for a in self.active(as_of)],
                "overdue_review": [a.pma_id for a in self.overdue_reviews(as_of)],
                "expired": [a.pma_id for a in self.expired(as_of)]}
