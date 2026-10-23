"""Read-only audit and governance endpoints.

Disabled unless ``AUDIT_API_TOKEN`` is set; then every call needs
``Authorization: Bearer <token>``. Page size is capped by ``AUDIT_MAX_PAGE_SIZE``.
"""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query

from api.security import require_audit_token
from credit_risk_model.config import get_settings
from credit_risk_model.governance.audit import AuditStore
from credit_risk_model.governance.pma import PMARegister, PMARegisterError

router = APIRouter(prefix="/audit", tags=["audit"], dependencies=[Depends(require_audit_token)])


def _limit(limit: int) -> int:
    return min(limit, get_settings().audit_max_page_size)


@router.get("/decisions")
def decisions(
    date: str | None = Query(None, description="YYYY-MM-DD"),
    model_version: str | None = None,
    request_id: str | None = None,
    limit: int = Query(50, ge=1),
    offset: int = Query(0, ge=0),
):
    if date:
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="date must be YYYY-MM-DD") from exc
    limit = _limit(limit)
    events, total = AuditStore().read_decisions(
        date_str=date, model_version=model_version, request_id=request_id,
        limit=limit, offset=offset)
    return {"total": total, "limit": limit, "offset": offset, "events": events}


@router.get("/changes")
def changes(model_version: str | None = None, limit: int = Query(50, ge=1),
            offset: int = Query(0, ge=0)):
    """Model-change events: training, validation, inventory edits, monitoring alerts, PMAs."""
    limit = _limit(limit)
    events, total = AuditStore().read_changes(model_version=model_version, limit=limit,
                                              offset=offset)
    return {"total": total, "limit": limit, "offset": offset, "events": events}


@router.get("/adjustments")
def adjustments():
    """Post-model adjustment register with active, overdue-review and expired IDs."""
    register = PMARegister()
    try:
        records = [a.model_dump(mode="json") for a in register.load()]
        summary = register.summary()
    except PMARegisterError as exc:
        raise HTTPException(status_code=500, detail="PMA register unreadable") from exc
    return {"summary": summary, "adjustments": records}


@router.get("/monitoring")
def monitoring(limit: int = Query(20, ge=1)):
    events = AuditStore().read_monitoring(limit=_limit(limit))
    return {"total": len(events), "events": events}


@router.get("/verify")
def verify():
    """Check the hash chain of every audit file."""
    results = AuditStore().verify()
    return {"ok": all(ok for _, ok, _ in results),
            "files": [{"category": c, "ok": ok, "message": m} for c, ok, m in results]}
