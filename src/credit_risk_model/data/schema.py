"""Data contracts used at ingestion and scoring boundaries."""
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LoanRecord(BaseModel):
    """Minimal example contract; extend with the features in your approved data dictionary."""
    model_config = ConfigDict(extra="allow")
    applicant_id: str | None = None
    target: int | None = Field(default=None, ge=0, le=1)

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    features: dict[str, Any]
    request_id: str | None = None

class PredictionResponse(BaseModel):
    request_id: str
    probability_of_default: float = Field(ge=0.0, le=1.0)
    score: int | None = None
    model_version: str
