"""Append-only, hash-chained JSONL audit store.

Every record carries ``prev_hash`` (the previous record's hash in the same daily file) and
``record_hash`` (SHA-256 of the record's canonical JSON without ``record_hash``). Editing,
removing or reordering a line breaks the chain, which :func:`verify_file` detects. This is
tamper *evidence*, not tamper prevention: whoever controls the files can rewrite a whole
chain, and deleting an entire daily file is only visible as a gap in dates. The store is
designed for a single writer process.

Run ``python -m credit_risk_model.governance.audit verify`` to check every file, or
``... prune`` to delete files older than ``AUDIT_RETENTION_DAYS``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from credit_risk_model.config import get_settings

_LOCK = threading.Lock()
GENESIS_HASH = "0" * 64
CATEGORIES = ("decisions", "changes", "monitoring")
_FILE_DATE = re.compile(r"-(\d{4}-\d{2}-\d{2})\.jsonl$")


class AuditWriteError(RuntimeError):
    """Raised when an audit record cannot be persisted and the store is fail-closed."""


@dataclass(slots=True)
class PredictionEvent:
    request_id: str
    timestamp_utc: str
    model_name: str
    model_version: str
    model_status: str
    probability_of_default: float
    score: float | None
    input_feature_names: list[str]
    latency_ms: float
    environment: str
    api_user_or_client_id: str | None = None
    event_type: str = "prediction"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))


def new_request_id() -> str:
    return str(uuid.uuid4())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _canonical(record: dict) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)


def compute_hash(record: dict) -> str:
    body = {k: v for k, v in record.items() if k != "record_hash"}
    return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()


def verify_file(path: Path) -> tuple[bool, str]:
    """Check the hash chain of one JSONL file. Returns (ok, message)."""
    prev = GENESIS_HASH
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            return False, f"{path.name}:{number}: not valid JSON"
        if record.get("prev_hash") != prev:
            return False, f"{path.name}:{number}: prev_hash does not match previous record"
        if record.get("record_hash") != compute_hash(record):
            return False, f"{path.name}:{number}: record_hash does not match contents"
        prev = record["record_hash"]
    return True, f"{path.name}: ok"


class AuditStore:
    """Writes prediction, change and monitoring events to daily-rotated JSONL files."""

    def __init__(self, root: Path | None = None, *, fail_closed: bool | None = None):
        settings = get_settings()
        self._root = Path(root) if root else settings.audit_trail_dir
        self.fail_closed = settings.audit_fail_closed if fail_closed is None else fail_closed

    def _dir(self, category: str) -> Path:
        return self._root / category

    def _file(self, category: str, when: datetime) -> Path:
        prefix = {"decisions": "decisions", "changes": "changes",
                  "monitoring": "monitoring"}[category]
        return self._dir(category) / f"{prefix}-{when.strftime('%Y-%m-%d')}.jsonl"

    def write_prediction(self, event: PredictionEvent) -> Path:
        return self._write("decisions", event.to_dict())

    def write_change(self, event: dict[str, Any]) -> Path:
        return self._write("changes", dict(event))

    def write_monitoring(self, event: dict[str, Any]) -> Path:
        return self._write("monitoring", dict(event))

    @staticmethod
    def _last_hash(path: Path) -> str:
        if not path.exists():
            return GENESIS_HASH
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            return GENESIS_HASH
        return json.loads(lines[-1]).get("record_hash", GENESIS_HASH)

    def _write(self, category: str, record: dict[str, Any]) -> Path:
        path = self._file(category, datetime.now(timezone.utc))
        record.setdefault("timestamp_utc", utc_now())
        with _LOCK:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                record["prev_hash"] = self._last_hash(path)
                record["record_hash"] = compute_hash(record)
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")
                    fh.flush()
            except (OSError, ValueError) as exc:
                if self.fail_closed:
                    raise AuditWriteError(f"audit write failed for {category}") from exc
                # Demo-only fail-open mode: the event is lost, so say so loudly.
                print(f"[audit] ERROR: {category} record NOT persisted: {exc}", file=sys.stderr)
        return path

    def _read(self, category: str, date_str: str | None = None) -> list[dict]:
        if date_str:
            files = [self._file(category, datetime.strptime(date_str, "%Y-%m-%d"))]
        else:
            files = sorted(self._dir(category).glob("*.jsonl"), reverse=True)
        events: list[dict] = []
        for fp in files:
            if not fp.exists():
                continue
            lines = fp.read_text(encoding="utf-8").splitlines()
            for line in reversed(lines):
                if line.strip():
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return events  # newest first

    def read_decisions(self, *, date_str: str | None = None, model_version: str | None = None,
                       request_id: str | None = None, limit: int = 100,
                       offset: int = 0) -> tuple[list[dict], int]:
        events = [e for e in self._read("decisions", date_str)
                  if (not model_version or e.get("model_version") == model_version)
                  and (not request_id or e.get("request_id") == request_id)]
        return events[offset:offset + limit], len(events)

    def read_changes(self, *, model_version: str | None = None, limit: int = 100,
                     offset: int = 0) -> tuple[list[dict], int]:
        events = [e for e in self._read("changes")
                  if not model_version or e.get("model_version") == model_version]
        return events[offset:offset + limit], len(events)

    def read_monitoring(self, *, limit: int = 20) -> list[dict]:
        return self._read("monitoring")[:limit]

    def verify(self) -> list[tuple[str, bool, str]]:
        results = []
        for category in CATEGORIES:
            for fp in sorted(self._dir(category).glob("*.jsonl")):
                ok, message = verify_file(fp)
                results.append((category, ok, message))
        return results

    def prune(self, retention_days: int | None = None, *, today: datetime | None = None) -> list[Path]:
        """Delete daily files older than the retention period; returns deleted paths."""
        days = retention_days or get_settings().audit_retention_days
        cutoff = (today or datetime.now(timezone.utc)).date() - timedelta(days=days)
        removed = []
        for category in CATEGORIES:
            for fp in self._dir(category).glob("*.jsonl"):
                match = _FILE_DATE.search(fp.name)
                if match and datetime.strptime(match.group(1), "%Y-%m-%d").date() < cutoff:
                    fp.unlink()
                    removed.append(fp)
        return removed


def record_change(change_type: str, *, model_name: str | None = None,
                  model_version: str | None = None, actor: str = "pipeline",
                  store: AuditStore | None = None, **details) -> dict:
    """Append a model-change event (training, validation, inventory or monitoring)."""
    event = {"event_type": "model_change", "change_type": change_type,
             "model_name": model_name, "model_version": model_version, "actor": actor,
             "details": details}
    (store or AuditStore()).write_change(event)
    return event


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit trail maintenance")
    parser.add_argument("command", choices=["verify", "prune"])
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--retention-days", type=int, default=None)
    args = parser.parse_args(argv)
    store = AuditStore(args.root)
    if args.command == "verify":
        results = store.verify()
        for category, ok, message in results:
            print(f"[{'OK' if ok else 'FAIL'}] {category}: {message}")
        return 0 if all(ok for _, ok, _ in results) else 1
    for path in store.prune(args.retention_days):
        print(f"deleted {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
