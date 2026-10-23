"""Drift and performance monitoring: compare a current sample with a reference sample.

Only the champion's input features are monitored (the target and date columns are
excluded). Population PSI is computed on predicted PD; the run fails if the data lacks
any model feature. Performance (AUROC, PD level) is compared when both samples have labels.

Exit codes: 0 = ok, 1 = investigate, 2 = escalate, 3 = input error.

Example:
  python scripts/generate_synthetic.py --rows 5000 --seed 42 --out data/reference.csv
  python scripts/generate_synthetic.py --rows 5000 --seed 43 --drift 0.15 --out data/current.csv
  python pipelines/monitor.py --reference data/reference.csv --current data/current.csv
"""
import argparse
import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from credit_risk_model.config import get_settings, resolve_path
from credit_risk_model.data.loader import load_csv
from credit_risk_model.governance.audit import AuditStore, record_change
from credit_risk_model.models.champion import ChampionModel
from credit_risk_model.monitoring.alerts import build_alerts
from credit_risk_model.monitoring.drift import alert_level, drift_summary
from credit_risk_model.monitoring.report import write_summary
from credit_risk_model.validation.stability import psi

EXIT_CODES = {"ok": 0, "investigate": 1, "escalate": 2}
INPUT_ERROR = 3


def performance(model, reference, current, target, settings) -> dict:
    if target not in reference or target not in current:
        return {"status": "not_assessed", "reason": "target column missing", "level": "ok"}
    y_ref, y_cur = reference[target].astype(int), current[target].astype(int)
    if y_ref.nunique() < 2 or y_cur.nunique() < 2:
        return {"status": "not_assessed", "reason": "a sample has only one outcome class",
                "level": "ok"}
    p_ref, p_cur = model.predict_proba(reference), model.predict_proba(current)
    ref_auc, cur_auc = float(roc_auc_score(y_ref, p_ref)), float(roc_auc_score(y_cur, p_cur))
    drop = ref_auc - cur_auc
    return {"status": "assessed", "reference_auroc": ref_auc, "current_auroc": cur_auc,
            "auroc_drop": drop, "reference_mean_pd": float(np.mean(p_ref)),
            "current_mean_pd": float(np.mean(p_cur)),
            "reference_default_rate": float(y_ref.mean()),
            "current_default_rate": float(y_cur.mean()),
            "level": alert_level(drop, settings.auroc_drop_investigate,
                                 settings.auroc_drop_escalate)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--reference", required=True)
    p.add_argument("--current", required=True)
    p.add_argument("--model", default=None, help="Defaults to <artifacts>/champion.joblib")
    p.add_argument("--target", default="target")
    p.add_argument("--output", default=None, help="Defaults to <reports>/drift_summary.json")
    args = p.parse_args(argv)
    settings = get_settings()
    settings.ensure_dirs()
    model_path = resolve_path(args.model) if args.model else settings.artifacts_dir / "champion.joblib"
    if not model_path.exists():
        print(f"[monitor] model artefact not found: {model_path}", file=sys.stderr)
        return INPUT_ERROR
    model = ChampionModel.load(model_path)
    reference, current = load_csv(resolve_path(args.reference)), load_csv(resolve_path(args.current))
    features = list(model.feature_names)
    for name, sample in (("reference", reference), ("current", current)):
        missing = [f for f in features if f not in sample]
        if missing:
            print(f"[monitor] {name} data lacks model features: {missing}", file=sys.stderr)
            return INPUT_ERROR

    thresholds = {"psi_investigate": settings.psi_investigate,
                  "psi_escalate": settings.psi_escalate,
                  "csi_investigate": settings.csi_investigate,
                  "csi_escalate": settings.csi_escalate}
    summary = drift_summary(reference, current, features=features,
                            thresholds={"investigate": settings.csi_investigate,
                                        "escalate": settings.csi_escalate})
    model_name = model.metadata.get("model_name", settings.champion_name)
    model_version = model.metadata.get("model_version", "unregistered")
    alerts = build_alerts(summary, model_name=model_name)
    output = resolve_path(args.output) if args.output else settings.reports_dir / "drift_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "alerts": alerts}, indent=2))

    feature_metrics = [{"feature": f, "csi": v["psi"], "level": v["level"]}
                       for f, v in summary["features"].items()]
    population_psi = psi(model.predict_proba(reference), model.predict_proba(current))
    perf = performance(model, reference, current, args.target, settings)
    result = write_summary(
        settings.artifacts_dir / "monitoring", settings.reports_dir / "monitoring_report.md",
        model_name=model_name, model_version=model_version,
        reference_meta={"path": args.reference, "rows": len(reference)},
        current_meta={"path": args.current, "rows": len(current)},
        population_psi=population_psi, feature_metrics=feature_metrics,
        thresholds=thresholds, performance=perf)

    store = AuditStore()
    store.write_monitoring({"event_type": "monitoring_run", **result})
    if result["overall_level"] != "ok":
        record_change("monitoring_alert", model_name=model_name, model_version=model_version,
                      store=store, level=result["overall_level"],
                      features=result["features_exceeding_thresholds"],
                      population_psi=result["population_psi"])
    print(json.dumps({"overall_level": result["overall_level"],
                      "population_psi": result["population_psi"],
                      "features_exceeding_thresholds": result["features_exceeding_thresholds"],
                      "performance_level": perf["level"]}, indent=2))
    return EXIT_CODES[result["overall_level"]]


if __name__ == "__main__":
    sys.exit(main())
