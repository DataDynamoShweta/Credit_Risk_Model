"""Read-only audit and governance endpoints."""
from __future__ import annotations

import csv
import json

from fastapi import APIRouter, Query

from credit_risk_model.config import get_settings
from credit_risk_model.governance.audit import AuditStore

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("/decisions")
def decisions(
    date: str | None = Query(None, description="YYYY-MM-DD"),
    model_version: str | None = None,
    request_id: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    store = AuditStore()
    events, total = store.read_decisions(
        date_str=date, model_version=model_version,
        request_id=request_id, limit=limit, offset=offset)
    return {"total": total, "limit": limit, "offset": offset, "events": events}


@router.get("/changes")
def changes(
    model_version: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    path = get_settings().pma_log_path
    if not path.exists():
        return {"total": 0, "events": []}
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if model_version:
        rows = [r for r in rows if r.get("model_version") == model_version]
    return {"total": len(rows), "limit": limit, "offset": offset,
            "events": rows[offset:offset + limit]}


@router.get("/monitoring")
def monitoring(limit: int = Query(20, ge=1, le=500)):
    base = get_settings().audit_trail_dir / "monitoring"
    if not base.exists():
        return {"events": []}
    files = sorted(base.glob("monitoring-*.jsonl"), reverse=True)
    events: list[dict] = []
    for fp in files:
        for line in reversed(fp.read_text(encoding="utf-8").splitlines()):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(events) >= limit:
                break
        if len(events) >= limit:
            break
    return {"total": len(events), "events": events}
