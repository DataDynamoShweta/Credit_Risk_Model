from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr

REQUEST_ID_PATTERN = r"^[A-Za-z0-9._:-]{1,64}$"
FeatureValue = StrictInt | StrictFloat | StrictStr


class PredictRequest(BaseModel):
    """Feature values are checked against the model's feature contract before scoring."""
    model_config = ConfigDict(extra="forbid")

    features: Annotated[dict[Annotated[str, Field(max_length=64)], FeatureValue],
                        Field(min_length=1, max_length=100)]
    request_id: Annotated[str, Field(pattern=REQUEST_ID_PATTERN)] | None = None


class PredictResponse(BaseModel):
    request_id: str
    probability_of_default: float = Field(ge=0, le=1)
    score: int
    model_name: str
    model_version: str
