"""Load the champion artefact, gated by the governance model inventory.

Only successful loads are cached, keyed on the artefact's path, size and modification
time, so a model trained after the API started becomes available without a restart and a
retrained artefact replaces the old one. The inventory is re-read when its file changes.

Pickle-based artefacts (joblib) execute code when loaded: only load artefacts you built.
Set ``MODEL_SHA256`` (or ``artifact_sha256`` in the inventory) to refuse any artefact
whose hash differs.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from credit_risk_model.config import get_settings, project_root
from credit_risk_model.governance.inventory import (
    InventoryValidationError,
    ModelEntry,
    ModelInventory,
    can_serve,
    file_sha256,
)

logger = logging.getLogger("model_loader")
_lock = threading.Lock()
_inventory_cache: dict = {}
_model_cache: dict = {}


@dataclass
class LoadState:
    model: object | None
    entry: ModelEntry | None
    reason: str

    @property
    def loaded(self) -> bool:
        return self.model is not None


def _file_key(path: Path):
    stat = path.stat()
    return (str(path), stat.st_size, stat.st_mtime_ns)


def champion_entry() -> ModelEntry | None:
    settings = get_settings()
    path = settings.inventory_path
    if not path.exists():
        logger.warning("Model inventory not found: %s", path)
        return None
    key = _file_key(path)
    with _lock:
        if _inventory_cache.get("key") == key:
            return _inventory_cache["entry"]
    try:
        entry = ModelInventory.load(path).get(settings.champion_name)
    except InventoryValidationError as exc:
        logger.warning("Model inventory invalid: %s", exc)
        return None
    with _lock:
        _inventory_cache.update(key=key, entry=entry)
    return entry


def artefact_path(entry: ModelEntry | None) -> Path | None:
    settings = get_settings()
    if settings.model_path:
        return settings.model_path
    if entry is None:
        return None
    path = Path(entry.artifact_path)
    return path if path.is_absolute() else project_root() / path


def load_state() -> LoadState:
    """Current serving state; never caches a failure."""
    settings = get_settings()
    entry = champion_entry()
    allowed, reason = can_serve(entry, allow_unreviewed=settings.allow_unreviewed_model)
    if not allowed:
        return LoadState(None, entry, f"blocked by inventory: {reason}")
    path = artefact_path(entry)
    if path is None or not path.exists():
        return LoadState(None, entry, "model artefact not found; run pipelines/train.py")
    key = _file_key(path)
    with _lock:
        cached = _model_cache.get("key") == key
        if cached:
            return LoadState(_model_cache["model"], entry, reason)
    expected = settings.model_sha256 or (entry.artifact_sha256 if entry else None)
    if expected and file_sha256(path) != expected.lower():
        return LoadState(None, entry, "artefact SHA-256 does not match the expected hash")
    from credit_risk_model.models.champion import ChampionModel
    model = ChampionModel.load(path)
    artefact_version = getattr(model, "metadata", {}).get("model_version")
    if entry and artefact_version not in (None, entry.model_version):
        return LoadState(None, entry, f"artefact version {artefact_version} does not match "
                                      f"inventory version {entry.model_version}")
    with _lock:
        _model_cache.update(key=key, model=model)
    return LoadState(model, entry, reason)


def load_champion():
    """Return the champion model, or None if it cannot be served right now."""
    return load_state().model


load_model = load_champion  # backwards-compatible name


def model_version() -> str:
    entry = champion_entry()
    return entry.model_version if entry else "unregistered"


def clear_cache() -> None:
    with _lock:
        _inventory_cache.clear()
        _model_cache.clear()
