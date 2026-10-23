"""Train the champion (and optionally a challenger) with complete, reproducible metadata.

Outputs (in --output-dir, default ``artifacts/``):
  champion.joblib, challenger.joblib (optional), feature_list.json,
  training_metadata.json, training_metrics.json
and a ``model_trained`` event in the hash-chained change log.

Examples:
  python pipelines/train.py                                   # synthetic data
  python pipelines/train.py --data loan_data_2015.csv --adapter lending_club
"""
import argparse
import json
import sys

import joblib
from sklearn.metrics import roc_auc_score

from credit_risk_model.config import get_settings, resolve_path
from credit_risk_model.data.sources import ADAPTERS, load_dataset
from credit_risk_model.governance.audit import record_change
from credit_risk_model.governance.inventory import (
    InventoryValidationError,
    ModelInventory,
    file_sha256,
    record_inventory_snapshot,
)
from credit_risk_model.models import predict_pd
from credit_risk_model.models.challenger import SUPPORTED_ALGORITHMS, build_challenger
from credit_risk_model.models.champion import ChampionModel, data_fingerprint
from credit_risk_model.validation.suite import split_dataset


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", help="CSV path; omit to use the canonical synthetic data")
    p.add_argument("--adapter", choices=ADAPTERS, default="none")
    p.add_argument("--target", default="target")
    p.add_argument("--date-column", default="application_date")
    p.add_argument("--rows", type=int, default=20000, help="Synthetic rows")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-fraction", type=float, default=0.25)
    p.add_argument("--oot-fraction", type=float, default=0.2)
    p.add_argument("--class-weight", choices=["none", "balanced"], default="none",
                   help="'balanced' distorts PDs; use only for ranking experiments")
    p.add_argument("--challenger", choices=[*SUPPORTED_ALGORITHMS, "none"], default="histgb")
    p.add_argument("--output-dir", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    settings.ensure_dirs()
    out = resolve_path(args.output_dir) if args.output_dir else settings.artifacts_dir
    out.mkdir(parents=True, exist_ok=True)

    frame, source = load_dataset(data=args.data, target=args.target,
                                 date_column=args.date_column, adapter=args.adapter,
                                 rows=args.rows, seed=args.seed)
    date_column = source.get("date_column") if source.get("date_column") in frame else None
    splits = split_dataset(frame, "target", date_column, test_fraction=args.test_fraction,
                           oot_fraction=args.oot_fraction, seed=args.seed)

    try:
        inventory = ModelInventory.load(settings.inventory_path)
    except InventoryValidationError as exc:
        print(f"[train] inventory unavailable ({exc}); versions recorded as 'unregistered'")
        inventory = None
    champion_entry = inventory.get(settings.champion_name) if inventory else None
    version = champion_entry.model_version if champion_entry else "unregistered"

    class_weight = None if args.class_weight == "none" else args.class_weight
    exclude = (date_column,) if date_column else ()
    model = ChampionModel.fit(splits.train, "target", random_state=args.seed,
                              class_weight=class_weight, exclude=exclude,
                              model_name=settings.champion_name, model_version=version)
    test_X = splits.test.drop(columns=["target", *exclude])
    holdout_auroc = float(roc_auc_score(splits.test["target"], model.predict_proba(test_X)))
    model.metadata.update({"data_source": source, "split": {
        "date_column": date_column, "test_fraction": args.test_fraction,
        "oot_fraction": args.oot_fraction, "seed": args.seed, "sizes": splits.sizes()},
        "full_data_sha256": data_fingerprint(frame), "holdout_auroc": holdout_auroc})
    model.save(out / "champion.joblib")

    metadata = {"champion": {**model.metadata,
                             "artifact_sha256": file_sha256(out / "champion.joblib")}}
    if args.challenger != "none":
        challenger_entry = inventory.get("credit_risk_challenger") if inventory else None
        challenger = build_challenger(splits.train, "target", args.challenger,
                                      random_state=args.seed, exclude=exclude)
        challenger.challenger_metadata_.update({
            "model_name": "credit_risk_challenger",
            "model_version": challenger_entry.model_version if challenger_entry else "unregistered",
            "holdout_auroc": float(roc_auc_score(splits.test["target"],
                                                 predict_pd(challenger, test_X)))})
        joblib.dump(challenger, out / "challenger.joblib")
        metadata["challenger"] = {**challenger.challenger_metadata_,
                                  "artifact_sha256": file_sha256(out / "challenger.joblib")}

    (out / "feature_list.json").write_text(json.dumps(
        {"model_name": settings.champion_name, "model_version": version,
         "features": model.metadata["features"]}, indent=2), encoding="utf-8")
    (out / "training_metadata.json").write_text(json.dumps(metadata, indent=2, default=str),
                                                encoding="utf-8")
    (out / "training_metrics.json").write_text(json.dumps(
        {"holdout_auroc": holdout_auroc,
         "holdout_mean_pd": float(model.predict_proba(test_X).mean()),
         "holdout_default_rate": float(splits.test["target"].mean())}, indent=2),
        encoding="utf-8")

    record_change("model_trained", model_name=settings.champion_name, model_version=version,
                  artifact_sha256=metadata["champion"]["artifact_sha256"],
                  data_source=source["kind"], class_weight=class_weight,
                  holdout_auroc=holdout_auroc, challenger=args.challenger)
    if settings.inventory_path.exists():
        record_inventory_snapshot(settings.inventory_path)
    print(f"Saved {out / 'champion.joblib'} ({settings.champion_name} {version}); "
          f"holdout AUROC={holdout_auroc:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
