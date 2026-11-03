from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

REQUEST_ID_PATTERN = r"^[A-Za-z0-9._:-]{1,64}$"
FeatureValue = StrictInt | StrictFloat | StrictStr


class PredictRequest(BaseModel):
    """Feature values are checked against the model's feature contract before scoring."""
    model_config = ConfigDict(extra="forbid")

    features: Annotated[dict[Annotated[str, Field(max_length=64)], FeatureValue],
                        Field(min_length=1, max_length=100)]
    request_id: Annotated[str, Field(pattern=REQUEST_ID_PATTERN)] | None = None

    @model_validator(mode="before")
    @classmethod
    def wrap_flat_features(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # If features is explicitly provided, leave untouched for strict validation
            if "features" in data:
                return data
            # Otherwise, support flat applicant payload for interactive UI scoring
            req_id = data.get("request_id")
            feats = {k: v for k, v in data.items() if k != "request_id"}
            return {"features": feats, "request_id": req_id}
        return data


class PredictResponse(BaseModel):
    request_id: str
    probability_of_default: float = Field(ge=0, le=1)
    score: int
    model_name: str
    model_version: str
