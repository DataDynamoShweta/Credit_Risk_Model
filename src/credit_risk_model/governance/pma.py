"""Post-Model-Analysis log: CSV-backed register of model-change events."""
from __future__ import annotations

import csv
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from credit_risk_model.config import get_settings


class PMAEventType(str, Enum):
    NEW_MODEL_VERSION = "new_model_version"
    CHALLENGER_COMPARISON = "challenger_comparison"
    THRESHOLD_CHANGE = "threshold_change"
    FEATURE_CHANGE = "feature_change"
    MONITORING_ALERT = "monitoring_alert"
    MANUAL_REVIEW = "manual_review"
    MODEL_RETIREMENT = "model_retirement"


class PMAStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    CLOSED = "closed"
    CANCELLED = "cancelled"


COLUMNS = [
    "pma_id", "timestamp_utc", "model_name", "model_version",
    "event_type", "trigger", "description", "owner", "status",
    "related_validation_report", "related_monitoring_run",
    "approved_by", "next_action",
]


@dataclass(slots=True)
class PMAEvent:
    model_name: str
    model_version: str
    event_type: PMAEventType
    trigger: str
    description: str
    owner: str
    status: PMAStatus = PMAStatus.OPEN
    related_validation_report: str = ""
    related_monitoring_run: str = ""
    approved_by: str = ""
    next_action: str = ""
    pma_id: str = ""
    timestamp_utc: str = ""

    def __post_init__(self) -> None:
        if not self.pma_id:
            self.pma_id = f"PMA-{uuid.uuid4().hex[:12].upper()}"
        if not self.timestamp_utc:
            self.timestamp_utc = datetime.now(
                timezone.utc).isoformat(timespec="seconds")

    def as_row(self) -> dict[str, str]:
        row = asdict(self)
        row["event_type"] = self.event_type.value
        row["status"] = self.status.value
        return row


class PMALog:
    def __init__(self, path: Path | None = None):
        self._path = path or get_settings().pma_log_path
        if not self._path.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=COLUMNS).writeheader()

    def append(self, event: PMAEvent) -> str:
        existing = {r["pma_id"] for r in self._read_rows()}
        if event.pma_id in existing:
            raise ValueError(f"Duplicate pma_id: {event.pma_id}")
        with self._path.open("a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=COLUMNS).writerow(event.as_row())
        return event.pma_id

    def _read_rows(self) -> list[dict[str, str]]:
        with self._path.open(encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
