"""Durable, append-only JSONL audit store."""
from __future__ import annotations

import json
import sys
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from credit_risk_model.config import get_settings

_LOCK = threading.Lock()


@dataclass(slots=True)
class PredictionEvent:
    request_id: str
    timestamp_utc: str
    model_name: str
    model_version: str
    model_status: str
    probability_of_default: float
    score: float | None
    decision_threshold: float
    decision: str
    input_feature_names: list[str]
    latency_ms: float
    environment: str
    api_user_or_client_id: str | None = None
    event_type: str = "prediction"

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


def new_request_id() -> str:
    return str(uuid.uuid4())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class AuditStore:
    """Writes prediction and system events to daily-rotated JSONL."""

    def __init__(self, root: Path | None = None):
        self._root = root or get_settings().audit_trail_dir
        self._decisions_dir = self._root / "decisions"
        self._changes_dir = self._root / "changes"
        self._monitoring_dir = self._root / "monitoring"
        for d in (self._decisions_dir, self._changes_dir, self._monitoring_dir):
            d.mkdir(parents=True, exist_ok=True)

    def _decision_file(self, when: datetime) -> Path:
        return self._decisions_dir / f"decisions-{when.strftime('%Y-%m-%d')}.jsonl"

    def write_prediction(self, event: PredictionEvent) -> Path:
        when = datetime.now(timezone.utc)
        path = self._decision_file(when)
        self._append(path, event.to_json())
        return path

    def write_change(self, event: dict[str, Any]) -> Path:
        return self._write_generic(self._changes_dir, "changes", event)

    def write_monitoring(self, event: dict[str, Any]) -> Path:
        return self._write_generic(self._monitoring_dir, "monitoring", event)

    def _write_generic(self, base: Path, prefix: str, event: dict) -> Path:
        when = datetime.now(timezone.utc)
        path = base / f"{prefix}-{when.strftime('%Y-%m-%d')}.jsonl"
        event.setdefault("timestamp_utc", _utc_now())
        self._append(path, json.dumps(event, separators=(",", ":")))
        return path

    def _append(self, path: Path, line: str) -> None:
        with _LOCK:
            try:
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(line)
                    fh.write("\n")
                    fh.flush()
            except OSError as exc:
                print(f"[audit] WARN: {exc}", file=sys.stderr)

    def read_decisions(
        self,
        *,
        date_str: str | None = None,
        model_version: str | None = None,
        request_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        if date_str:
            files = [self._decisions_dir / f"decisions-{date_str}.jsonl"]
        else:
            files = sorted(
                self._decisions_dir.glob("decisions-*.jsonl"), reverse=True)
        matched: list[dict] = []
        for fp in files:
            if not fp.exists():
                continue
            for line in fp.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if model_version and event.get("model_version") != model_version:
                    continue
                if request_id and event.get("request_id") != request_id:
                    continue
                matched.append(event)
        return matched[offset:offset + limit], len(matched)
