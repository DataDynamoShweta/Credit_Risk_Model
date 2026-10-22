"""Load a local champion artifact; production deployments can use MLflow registry."""
import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_model():
    path = Path(os.getenv("MODEL_PATH", "artifacts/champion.joblib"))
    if not path.exists():
        return None
    from credit_risk_model.models.champion import ChampionModel
    return ChampionModel.load(path)

def model_version():
    return os.getenv("MODEL_VERSION", "local-unregistered")
