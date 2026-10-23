"""Request correlation, request-size limit and privacy-conscious request logging.

One request ID is used everywhere: a valid ``x-request-id`` header is reused, a malformed
one is replaced with a new UUID (the raw value is never logged), and the endpoint may
adopt a valid ``request_id`` from the payload. The final ID is returned in the
``x-request-id`` response header, the response body, the audit record and the log line.
"""
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from api.security import safe_identifier
from credit_risk_model.config import get_settings

logger = logging.getLogger("decision_audit")


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        supplied = request.headers.get("x-request-id")
        request_id = safe_identifier(supplied) or str(uuid.uuid4())
        request.state.request_id = request_id
        if supplied and supplied != request_id:
            logger.warning("api_audit request_id=%s replaced malformed x-request-id header",
                           request_id)
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > get_settings().max_request_bytes:
            return JSONResponse({"detail": "Request body too large", "request_id": request_id},
                                status_code=413, headers={"x-request-id": request_id})
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        request_id = request.state.request_id  # the endpoint may have adopted a payload ID
        response.headers["x-request-id"] = request_id
        logger.info("api_audit request_id=%s method=%s path=%s status=%s duration_ms=%s",
                    request_id, request.method, request.url.path, response.status_code,
                    elapsed_ms)
        return response
