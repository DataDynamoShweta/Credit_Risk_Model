"""Request correlation and privacy-conscious audit metadata."""
import logging
import time
import uuid
from datetime import datetime, timezone

from starlette.middleware.base import BaseHTTPMiddleware

from credit_risk_model.config import get_settings
from credit_risk_model.governance.audit import AuditStore, PredictionEvent

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
        payload = getattr(request.state, "predict_audit_payload", None)
        if payload is not None:
            _persist_prediction(payload, elapsed_ms)
        return response

def _persist_prediction(payload: dict, latency_ms: float) -> None:
    """Append a PredictionEvent; raw feature values are never stored."""
    from api.model_loader import champion_entry, model_version
    settings = get_settings()
    entry = champion_entry()
    pd_value = payload["probability_of_default"]
    threshold = settings.decision_threshold
    AuditStore().write_prediction(PredictionEvent(
        request_id=payload["request_id"],
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        model_name=entry.model_name if entry else "credit-risk-champion",
        model_version=model_version(),
        model_status=entry.status.value if entry else "unregistered",
        probability_of_default=pd_value, score=payload["score"],
        decision_threshold=threshold,
        decision="decline" if pd_value >= threshold else "approve",
        input_feature_names=payload["input_feature_names"],
        latency_ms=latency_ms, environment=settings.environment))
