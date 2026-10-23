import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from api.logging_middleware import AuditLoggingMiddleware
from api.model_loader import champion_entry, load_champion, model_version
from api.routers import audit as audit_router
from api.schemas import PredictRequest, PredictResponse
from credit_risk_model.config import get_settings
from credit_risk_model.features.scorecard import probability_to_score


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_champion()
    yield

app = FastAPI(title="Credit Risk Model Validation Platform", version="0.1.0", lifespan=lifespan)
app.add_middleware(AuditLoggingMiddleware)
app.include_router(audit_router.router)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": load_champion() is not None,
            "environment": get_settings().environment}

@app.get("/model-card")
def model_card():
    entry = champion_entry()
    card = {"name": "credit-risk-champion", "version": model_version(),
            "purpose": "Demonstration PD estimation", "status": "not-approved-for-production"}
    if entry:
        card["inventory"] = {"model_name": entry.model_name, "model_type": entry.model_type,
                             "owner": entry.owner, "purpose": entry.purpose,
                             "status": entry.status.value, "approved_by": entry.approved_by,
                             "approval_date": entry.approval_date,
                             "next_review_date": entry.next_review_date,
                             "validation_report": entry.validation_report}
    return card

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request):
    model = load_champion()
    if model is None:
        raise HTTPException(status_code=503,
                            detail="Model unavailable: artifact missing or blocked by inventory")
    try:
        pd_value = float(model.predict_proba(payload.features)[0])
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid features: {exc}") from exc
    score = int(probability_to_score(pd_value))
    request_id = payload.request_id or str(uuid.uuid4())
    request.state.predict_audit_payload = {
        "request_id": request_id, "probability_of_default": pd_value, "score": score,
        "input_feature_names": sorted(payload.features)}
    return PredictResponse(request_id=request_id, probability_of_default=pd_value, score=score,
                           model_version=model_version())
