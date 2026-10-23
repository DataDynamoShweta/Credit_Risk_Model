"""End-to-end: synthetic data -> train -> validate -> monitor -> API (G5, G7, G11, G13).

Runs from an isolated project root that is the current working directory, so it also
checks that outputs land in the runtime directory rather than the source tree.
"""
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from credit_risk_model.data import synthetic
from credit_risk_model.governance.audit import AuditStore
from credit_risk_model.governance.inventory import ModelInventory
from pipelines import monitor, train, validate
from tests.helpers import FEATURES_OK


@pytest.fixture
def pipeline_run(project):
    assert train.main(["--rows", "4000"]) == 0
    assert validate.main(["--bootstrap", "50"]) == 0
    return project


def test_training_outputs_and_metadata(pipeline_run):
    artifacts = pipeline_run / "artifacts"
    for name in ("champion.joblib", "challenger.joblib", "feature_list.json",
                 "training_metadata.json", "training_metrics.json"):
        assert (artifacts / name).exists(), name
    meta = json.loads((artifacts / "training_metadata.json").read_text())
    champion = meta["champion"]
    assert champion["class_weight"] is None and champion["model_version"] == "1.3.0"
    assert champion["split"]["date_column"] == "application_date"
    assert len(champion["artifact_sha256"]) == 64 and len(champion["full_data_sha256"]) == 64
    assert meta["challenger"]["algorithm"] == "histgb"
    features = json.loads((artifacts / "feature_list.json").read_text())["features"]
    assert [f["name"] for f in features] == list(FEATURES_OK)


def test_validation_uses_held_out_data_only(pipeline_run):
    report = (pipeline_run / "reports" / "validation_report.md").read_text(encoding="utf-8")
    meta = json.loads((pipeline_run / "artifacts" / "training_metadata.json").read_text())
    sizes = meta["champion"]["split"]["sizes"]
    assert f"train {sizes['train']:,}, test {sizes['test']:,}, OOT {sizes['oot']:,}" in report
    assert "Validation data matches recorded training data | ✅ pass" in report
    assert meta["champion"]["training_rows"] == sizes["train"]
    for section in ("Methodology", "Data review", "Model implementation review",
                    "Discrimination", "Calibration", "Stability", "Out-of-time",
                    "Champion–challenger", "Sensitivity", "Observations", "Findings",
                    "Use restrictions", "Not assessed", "Sign-off"):
        assert section in report, section
    assert "Conceptual soundness | Requires" in report
    assert "{'" not in report and "[{" not in report  # no raw Python dicts/lists


def test_inventory_references_exist_after_pipeline(pipeline_run):
    inventory = ModelInventory.load(pipeline_run / "governance" / "model_inventory.yaml")
    assert inventory.missing_references(pipeline_run) == []


def test_change_log_records_training_validation_and_inventory(pipeline_run):
    events, _ = AuditStore().read_changes()
    types = [e["change_type"] for e in events]
    assert {"model_trained", "validation_run", "inventory_changed"} <= set(types)
    assert all(ok for _, ok, _ in AuditStore().verify())


def test_monitoring_exit_codes_and_outputs(pipeline_run):
    data = pipeline_run / "data"
    data.mkdir()
    synthetic.generate(4000, seed=101).to_csv(data / "reference.csv", index=False)
    synthetic.generate(4000, seed=102).to_csv(data / "stable.csv", index=False)
    synthetic.generate(4000, seed=103, drift=0.3).to_csv(data / "drifted.csv", index=False)
    args = ["--reference", "data/reference.csv", "--current"]
    assert monitor.main([*args, "data/stable.csv"]) == 0
    assert monitor.main([*args, "data/drifted.csv"]) in (1, 2)
    summary = json.loads((pipeline_run / "artifacts" / "monitoring" /
                          "monitoring_summary.json").read_text())
    assert summary["overall_level"] in ("investigate", "escalate")
    assert "target" not in summary["features_exceeding_thresholds"]
    assert summary["performance"]["status"] == "assessed"
    metrics = pd.read_csv(pipeline_run / "artifacts" / "monitoring" / "psi_csi_metrics.csv")
    assert set(metrics["feature"]) == set(FEATURES_OK)
    assert (pipeline_run / "reports" / "monitoring_report.md").exists()
    events, _ = AuditStore().read_changes()
    assert events[0]["change_type"] == "monitoring_alert"


def test_monitoring_fails_on_incompatible_data(pipeline_run):
    data = pipeline_run / "data"
    data.mkdir()
    pd.DataFrame({"income": [1, 2], "age": [30, 40]}).to_csv(data / "bad.csv", index=False)
    assert monitor.main(["--reference", "data/bad.csv", "--current", "data/bad.csv"]) == 3


def test_api_serves_trained_model(pipeline_run):
    with TestClient(__import__("api.main", fromlist=["app"]).app) as client:
        assert client.get("/health").json()["model_loaded"] is True
        r = client.post("/predict", json={"features": FEATURES_OK})
        assert r.status_code == 200
    decisions, total = AuditStore().read_decisions()
    assert total == 1 and decisions[0]["model_version"] == "1.3.0"
