"""Request correlation and privacy-conscious audit metadata."""
import logging
import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("decision_audit")

class AuditLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter()-start)*1000, 2)
        response.headers["x-request-id"] = request_id
        logger.info("api_audit request_id=%s method=%s path=%s status=%s duration_ms=%s",
                    request_id, request.method, request.url.path, response.status_code, elapsed_ms)
        return response
