import uuid

from fastapi import FastAPI, HTTPException

from api.logging_middleware import AuditLoggingMiddleware
from api.model_loader import load_model, model_version
from api.schemas import PredictRequest, PredictResponse
from credit_risk_model.features.scorecard import probability_to_score

app = FastAPI(title="Credit Risk Model Validation Platform", version="0.1.0")
app.add_middleware(AuditLoggingMiddleware)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": load_model() is not None}

@app.get("/model-card")
def model_card():
    return {"name": "credit-risk-champion", "version": model_version(),
            "purpose": "Demonstration PD estimation", "status": "not-approved-for-production"}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    model = load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact unavailable; train first")
    try:
        pd_value = float(model.predict_proba(payload.features)[0])
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid features: {exc}") from exc
    score = int(probability_to_score(pd_value))
    return PredictResponse(request_id=payload.request_id or str(uuid.uuid4()),
                           probability_of_default=pd_value, score=score,
                           model_version=model_version())
