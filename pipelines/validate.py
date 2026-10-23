"""Validate the champion on held-out data and write a Markdown (optionally PDF) report.

Leakage-safe by default: without --data, the synthetic training data is regenerated from
the parameters recorded in the model's metadata, its fingerprint is checked against the
training fingerprint, and the same split is rebuilt so only the held-out test and OOT
samples are scored. With --data, the file is split the same way and only its test/OOT
portions are scored; independence from the training data is reported as unverified unless
its fingerprint matches the training data.

Examples:
  python pipelines/validate.py
  python pipelines/validate.py --data loan_data_2015.csv --adapter lending_club
"""
import argparse
import json
import sys

import joblib

from credit_risk_model.config import get_settings, resolve_path
from credit_risk_model.data.sources import ADAPTERS, load_dataset
from credit_risk_model.governance.audit import record_change
from credit_risk_model.governance.inventory import (
    InventoryValidationError,
    ModelInventory,
    record_inventory_snapshot,
)
from credit_risk_model.governance.pma import PMARegister, PMARegisterError
from credit_risk_model.governance.report_builder import build_markdown
from credit_risk_model.models.champion import ChampionModel, data_fingerprint
from credit_risk_model.validation.suite import run_validation, split_dataset


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", help="Labelled CSV; omit to rebuild the synthetic hold-out")
    p.add_argument("--adapter", choices=ADAPTERS, default="none")
    p.add_argument("--target", default="target")
    p.add_argument("--date-column", default=None,
                   help="Defaults to the date column recorded at training time")
    p.add_argument("--model", default=None, help="Defaults to <artifacts>/champion.joblib")
    p.add_argument("--challenger-model", default=None,
                   help="Defaults to <artifacts>/challenger.joblib if present")
    p.add_argument("--report", default=None, help="Defaults to <reports>/validation_report.md")
    p.add_argument("--export-pdf", default=None, help="Optional PDF output path")
    p.add_argument("--bootstrap", type=int, default=200)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    settings.ensure_dirs()
    model_path = resolve_path(args.model) if args.model else settings.artifacts_dir / "champion.joblib"
    report_path = resolve_path(args.report) if args.report else settings.reports_dir / "validation_report.md"
    model = ChampionModel.load(model_path)
    meta = model.metadata
    split = meta.get("split")
    if not split:
        print("[validate] model has no split metadata; retrain with pipelines/train.py",
              file=sys.stderr)
        return 2

    if args.data is None:
        source = meta["data_source"]
        if source.get("kind") != "synthetic":
            print("[validate] model was trained on external data; pass --data", file=sys.stderr)
            return 2
        frame, source = load_dataset(data=None, rows=source["rows"], seed=source["seed"],
                                     drift=source.get("drift", 0.0))
    else:
        frame, source = load_dataset(data=args.data, target=args.target,
                                     date_column=args.date_column or split["date_column"],
                                     adapter=args.adapter)
    matches = data_fingerprint(frame) == meta.get("full_data_sha256")
    data_sha256_matches = True if matches else (False if args.data is None else None)
    date_column = args.date_column or split["date_column"]
    splits = split_dataset(frame, "target", date_column if date_column in frame else None,
                           test_fraction=split["test_fraction"],
                           oot_fraction=split["oot_fraction"], seed=split["seed"])

    challenger_path = (resolve_path(args.challenger_model) if args.challenger_model
                       else settings.artifacts_dir / "challenger.joblib")
    challengers = {}
    if challenger_path.exists():
        challenger = joblib.load(challenger_path)
        name = getattr(challenger, "challenger_metadata_", {}).get("algorithm", "challenger")
        challengers[f"challenger ({name})"] = challenger

    try:
        pma_summary = PMARegister(settings.pma_register_path).summary()
    except PMARegisterError as exc:
        print(f"[validate] PMA register unreadable: {exc}", file=sys.stderr)
        pma_summary = None

    thresholds = {"psi_investigate": settings.psi_investigate,
                  "psi_escalate": settings.psi_escalate,
                  "csi_investigate": settings.csi_investigate,
                  "csi_escalate": settings.csi_escalate}
    results = run_validation(model, splits, target="target", metadata=meta,
                             thresholds=thresholds, data_source=source["kind"],
                             data_sha256_matches=data_sha256_matches, challengers=challengers,
                             pma_summary=pma_summary, bootstrap=args.bootstrap)

    entry = None
    try:
        entry = ModelInventory.load(settings.inventory_path).get(settings.champion_name)
    except InventoryValidationError as exc:
        print(f"[validate] inventory unavailable: {exc}", file=sys.stderr)
    model_info = {
        "model_name": settings.champion_name,
        "model_version": entry.model_version if entry else meta.get("model_version"),
        "model_type": entry.model_type if entry else meta.get("algorithm"),
        "status": entry.status.value if entry else "unregistered",
        "limitations": entry.limitations if entry else [],
        "use_restrictions": entry.use_restrictions if entry else [],
    }
    if entry and meta.get("model_version") not in (None, entry.model_version):
        print(f"[validate] WARNING: artefact version {meta.get('model_version')} differs "
              f"from inventory version {entry.model_version}", file=sys.stderr)
    data_note = {True: "same data as training; metrics use held-out rows only",
                 False: "regenerated data does NOT match the training fingerprint",
                 None: "external data; independence from training data not verified"}
    build_markdown(report_path, {"model": model_info, "metadata": meta, "results": results,
                                 "thresholds": thresholds, "seed": split["seed"],
                                 "data_note": data_note[data_sha256_matches]})
    if args.export_pdf:
        from credit_risk_model.governance.pdf_export import export_pdf
        try:
            export_pdf(report_path, resolve_path(args.export_pdf))
        except (RuntimeError, OSError) as exc:
            print(f"[validate] PDF skipped: {exc}", file=sys.stderr)

    record_change("validation_run", model_name=settings.champion_name,
                  model_version=model_info["model_version"], report=str(report_path),
                  data_source=source["kind"], test_auroc=results["discrimination"]["auroc"],
                  calibration_ratio=results["calibration"]["calibration_ratio"],
                  automated_findings=len(results["findings"]))
    if settings.inventory_path.exists():
        record_inventory_snapshot(settings.inventory_path)
    summary = {"report": str(report_path), "test_auroc": results["discrimination"]["auroc"],
               "calibration_ratio": results["calibration"]["calibration_ratio"],
               "oot": (results["oot"] or {}).get("auroc"),
               "automated_findings": [f["title"] for f in results["findings"]]}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
