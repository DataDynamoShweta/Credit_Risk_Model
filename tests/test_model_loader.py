"""Model loading: no negative caching, integrity and version checks (G4.6, G10)."""
import shutil

from fastapi.testclient import TestClient

from api import model_loader
from api.main import app
from credit_risk_model.config import reset_settings_cache


def test_model_becomes_available_after_training_without_restart(project, trained_artifacts):
    client = TestClient(app)
    first = client.get("/health").json()
    assert first["model_loaded"] is False
    assert "not found" in first["reason"]
    shutil.copytree(trained_artifacts, project / "artifacts")  # "train" while running
    assert client.get("/health").json()["model_loaded"] is True


def test_inventory_edit_is_picked_up(trained_project):
    assert model_loader.load_state().loaded
    inventory = trained_project / "governance" / "model_inventory.yaml"
    inventory.write_text(inventory.read_text().replace("status: active", "status: suspended", 1))
    state = model_loader.load_state()
    assert not state.loaded and "suspended" in state.reason


def test_sha256_mismatch_refused(trained_project, monkeypatch):
    monkeypatch.setenv("MODEL_SHA256", "0" * 64)
    reset_settings_cache()
    state = model_loader.load_state()
    assert not state.loaded and "SHA-256" in state.reason


def test_sha256_match_accepted(trained_project, monkeypatch):
    from credit_risk_model.governance.inventory import file_sha256
    monkeypatch.setenv("MODEL_SHA256", file_sha256(trained_project / "artifacts" / "champion.joblib"))
    reset_settings_cache()
    assert model_loader.load_state().loaded


def test_version_mismatch_refused(trained_project):
    inventory = trained_project / "governance" / "model_inventory.yaml"
    inventory.write_text(inventory.read_text().replace("model_version: 1.3.0", "model_version: 9.9.9", 1))
    state = model_loader.load_state()
    assert not state.loaded and "does not match inventory version" in state.reason


def test_missing_inventory_blocks(trained_project):
    (trained_project / "governance" / "model_inventory.yaml").unlink()
    state = model_loader.load_state()
    assert not state.loaded and "no inventory entry" in state.reason
