from pydantic import BaseModel, Field
from typing import Any

class PredictRequest(BaseModel):
    features: dict[str, Any]
    request_id: str | None = None

class PredictResponse(BaseModel):
    request_id: str
    probability_of_default: float = Field(ge=0, le=1)
    score: int
    model_version: str
