"""Lightweight demo safeguards: bearer-token checks and safe identifiers.

These are not production security. Production deployments need an identity provider
(OIDC/OAuth2), authorisation by role, TLS termination and rate limiting at a gateway.
"""
from __future__ import annotations

import re
import secrets

from fastapi import HTTPException, Request

from credit_risk_model.config import get_settings

SAFE_ID = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")


def safe_identifier(value: str | None) -> str | None:
    """Return the value if it is a safe identifier, otherwise None (never log raw input)."""
    return value if value and SAFE_ID.fullmatch(value) else None


def _bearer(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    return token if scheme.lower() == "bearer" and token else None


def _check(request: Request, expected: str | None, *, disabled_detail: str | None):
    if not expected:
        if disabled_detail:
            raise HTTPException(status_code=403, detail=disabled_detail)
        return
    token = _bearer(request)
    if not token or not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token",
                            headers={"WWW-Authenticate": "Bearer"})


def require_audit_token(request: Request) -> None:
    """Audit endpoints are disabled unless AUDIT_API_TOKEN is set, then need that token."""
    _check(request, get_settings().audit_api_token,
           disabled_detail="Audit endpoints are disabled; set AUDIT_API_TOKEN to enable them")


def require_predict_token(request: Request) -> None:
    """If PREDICT_API_TOKEN is set, /predict requires it; otherwise /predict is open."""
    _check(request, get_settings().predict_api_token, disabled_detail=None)
