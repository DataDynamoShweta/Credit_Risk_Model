"""MLflow alias adapter (mocked) and configuration path handling (G7, G15)."""
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from credit_risk_model.config import Settings, reset_settings_cache
from credit_risk_model.models.registry import ModelRegistry


def test_set_alias_uses_alias_api():
    client = MagicMock()
    ModelRegistry("file:./mlruns", client=client).set_alias("m", 3, "champion")
    client.set_registered_model_alias.assert_called_once_with("m", "champion", "3")
    client.transition_model_version_stage.assert_not_called()


def test_version_for_alias_and_missing_alias():
    client = MagicMock()
    client.get_model_version_by_alias.return_value = "v3"
    assert ModelRegistry("x", client=client).version_for_alias("m") == "v3"
    client.get_model_version_by_alias.side_effect = RuntimeError("RESOURCE_DOES_NOT_EXIST")
    with pytest.raises(LookupError):
        ModelRegistry("x", client=client).version_for_alias("m", "gone")


def test_tracking_uri_read_from_settings(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.local")
    reset_settings_cache()
    try:
        assert ModelRegistry().tracking_uri == "http://mlflow.local"
    finally:
        reset_settings_cache()


def test_paths_resolve_against_cwd_and_project_root(tmp_path, monkeypatch):
    monkeypatch.delenv("PROJECT_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    assert Settings().artifacts_dir == tmp_path.resolve() / "artifacts"
    other = tmp_path / "other"
    monkeypatch.setenv("PROJECT_ROOT", str(other))
    settings = Settings()
    assert settings.inventory_path == other.resolve() / "governance" / "model_inventory.yaml"
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "abs"))
    assert Settings().reports_dir == tmp_path / "abs"


def test_loading_settings_and_app_creates_no_directories(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PROJECT_ROOT": str(tmp_path),
           "PYTHONPATH": os.pathsep.join([str(repo), str(repo / "src")])}
    code = "import api.main; from credit_risk_model.config import get_settings; get_settings()"
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, check=True)
    assert list(tmp_path.iterdir()) == []
