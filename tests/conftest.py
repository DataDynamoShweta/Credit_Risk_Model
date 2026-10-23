"""Shared fixtures: isolated project roots so tests never touch the working tree."""
import os
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _reset_caches():
    from api import model_loader
    from credit_risk_model.config import reset_settings_cache
    reset_settings_cache()
    model_loader.clear_cache()


def make_project(root: Path) -> Path:
    """Copy the governance files and report template into an empty project root."""
    (root / "governance").mkdir(parents=True, exist_ok=True)
    (root / "reports" / "templates").mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "governance" / "model_inventory.yaml", root / "governance")
    shutil.copy(REPO / "governance" / "pma_log.csv", root / "governance")
    shutil.copy(REPO / "reports" / "templates" / "validation_report.md.j2",
                root / "reports" / "templates")
    return root


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Empty project root (no model artefact) used as PROJECT_ROOT and cwd."""
    root = make_project(tmp_path / "project")
    for var in ("MODEL_PATH", "MODEL_SHA256", "ALLOW_UNREVIEWED_MODEL", "AUDIT_API_TOKEN",
                "PREDICT_API_TOKEN", "AUDIT_FAIL_CLOSED", "MAX_REQUEST_BYTES"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PROJECT_ROOT", str(root))
    monkeypatch.chdir(root)
    _reset_caches()
    yield root
    _reset_caches()


@pytest.fixture(scope="session")
def trained_artifacts(tmp_path_factory):
    """Train once per session (small synthetic sample) and return the artefacts dir."""
    root = make_project(tmp_path_factory.mktemp("trained"))
    old = os.environ.get("PROJECT_ROOT")
    os.environ["PROJECT_ROOT"] = str(root)
    _reset_caches()
    try:
        from pipelines import train
        assert train.main(["--rows", "4000"]) == 0
    finally:
        if old is None:
            os.environ.pop("PROJECT_ROOT", None)
        else:
            os.environ["PROJECT_ROOT"] = old
        _reset_caches()
    return root / "artifacts"


@pytest.fixture
def trained_project(project, trained_artifacts):
    """Isolated project root that already contains trained artefacts."""
    shutil.copytree(trained_artifacts, project / "artifacts")
    shutil.copytree(trained_artifacts.parent / "governance" / "audit_trail",
                    project / "governance" / "audit_trail")
    _reset_caches()
    return project
