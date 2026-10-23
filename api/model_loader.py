"""Load the champion artifact, gated by the governance model inventory."""
import logging
import os
from functools import lru_cache
from pathlib import Path

from credit_risk_model.config import PROJECT_ROOT, get_settings
from credit_risk_model.governance.inventory import (
    InventoryValidationError,
    ModelEntry,
    ModelInventory,
    can_serve,
)

CHAMPION_NAME = "credit_risk_champion"
logger = logging.getLogger("model_loader")


@lru_cache(maxsize=1)
def champion_entry() -> ModelEntry | None:
    try:
        return ModelInventory.load(get_settings().inventory_path).get(CHAMPION_NAME)
    except InventoryValidationError as exc:
        logger.warning("Model inventory unavailable: %s", exc)
        return None


@lru_cache(maxsize=1)
def load_champion():
    """Return the champion model, or None if the artifact is missing or the inventory blocks it."""
    entry = champion_entry()
    allowed, reason = can_serve(entry, allow_unreviewed=get_settings().allow_unreviewed_model)
    if not allowed:
        logger.warning("Champion not served: %s", reason)
        return None
    path = Path(os.getenv("MODEL_PATH") or entry.artifact_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return None
    from credit_risk_model.models.champion import ChampionModel
    return ChampionModel.load(path)


# Backwards-compatible name used by existing callers.
load_model = load_champion


def model_version():
    entry = champion_entry()
    return entry.model_version if entry else os.getenv("MODEL_VERSION", "local-unregistered")
