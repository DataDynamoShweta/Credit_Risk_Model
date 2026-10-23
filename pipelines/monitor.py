"""Scheduled drift job: compare reference and current feature CSV files."""
import argparse
import json
from pathlib import Path

import pandas as pd

from credit_risk_model.config import get_settings
from credit_risk_model.governance.audit import AuditStore
from credit_risk_model.monitoring.alerts import build_alerts
from credit_risk_model.monitoring.drift import drift_summary
from credit_risk_model.monitoring.report import alert_level, write_summary
from credit_risk_model.validation.stability import psi


def population_psi(reference, current, model_path):
    """PSI of predicted PD when the model's features are present, else the largest feature PSI."""
    if Path(model_path).exists():
        from credit_risk_model.models.champion import ChampionModel
        model = ChampionModel.load(model_path)
        needed = set(model.feature_names)
        if needed <= set(reference.columns) and needed <= set(current.columns):
            return psi(model.predict_proba(reference), model.predict_proba(current)), "predicted_pd"
    return None, "max_feature_psi"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--current", required=True)
    p.add_argument("--model-name", default="credit-risk-champion")
    p.add_argument("--model-version", default="unversioned")
    p.add_argument("--model", default="artifacts/champion.joblib",
                   help="Champion artifact used for score-level PSI, if present")
    p.add_argument("--output", default="reports/drift_summary.json")
    args=p.parse_args()
    settings=get_settings()
    reference, current = pd.read_csv(args.reference), pd.read_csv(args.current)
    summary=drift_summary(reference, current,
                          thresholds={"warning": settings.csi_investigate,
                                      "critical": settings.csi_alert})
    result={"summary":summary,"alerts":build_alerts(summary,model_name=args.model_name)}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2))

    # Summary JSON, per-feature CSV and Markdown report, plus an audit-trail event.
    feature_metrics=[{"feature": name, "psi": v["psi"], "csi": v["psi"],
                      "csi_level": alert_level(v["psi"], settings.csi_investigate,
                                               settings.csi_alert)}
                     for name, v in summary["features"].items()]
    pop_psi, basis = population_psi(reference, current, args.model)
    if pop_psi is None:
        pop_psi = max((f["psi"] for f in feature_metrics), default=0.0)
    monitoring=write_summary(
        settings.artifacts_dir / "monitoring", model_name=args.model_name,
        model_version=args.model_version,
        reference_meta={"path": args.reference, "rows": len(reference)},
        current_meta={"path": args.current, "rows": len(current),
                      "population_psi_basis": basis},
        population_psi=pop_psi, feature_metrics=feature_metrics,
        thresholds={"psi_investigate": settings.psi_investigate,
                    "psi_alert": settings.psi_alert,
                    "csi_investigate": settings.csi_investigate,
                    "csi_alert": settings.csi_alert})
    AuditStore().write_monitoring({"event_type": "monitoring_run", **monitoring})
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
