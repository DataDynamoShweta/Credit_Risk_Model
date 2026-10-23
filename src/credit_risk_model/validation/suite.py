"""Leakage-safe validation suite used by ``pipelines/validate.py``.

Design:

* :func:`split_dataset` is shared by training and validation. With a date column, the
  latest ``oot_fraction`` of applications (cut on a date boundary) is the out-of-time
  sample and the rest is split stratified into train/test. Without a date column only a
  stratified train/test split is made and OOT is reported as not assessed.
* Validation metrics are computed on the held-out test sample (and OOT sample); the
  training sample is used only as the stability reference and for WoE/IV review.

Automated findings are produced by simple rules and are labelled as automated: they are
inputs for a validator, not validation conclusions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..features.scorecard import probability_to_score
from ..features.woe_binning import calculate_woe_iv
from ..models import predict_pd
from ..monitoring.drift import alert_level
from .calibration import calibration
from .challenger_bench import benchmark
from .discrimination import discrimination
from .oot import evaluate_sample, oot_split
from .stability import psi
from .vintage import vintage_performance

ASSESSED, PARTIAL, NOT_ASSESSED = "Assessed", "Partially assessed", "Not assessed"

ALWAYS_NOT_ASSESSED = [
    ("Conceptual soundness", ("Requires expert review of model design, variable selection "
     "and business rationale; not automatable.")),
    ("Data lineage", ("No source-system lineage or data-quality controls are available for "
     "the input data.")),
    ("ECL / IRB-specific calibration", ("PDs are not calibrated to a through-the-cycle or "
     "point-in-time definition, long-run average default rate or regulatory default "
     "definition.")),
    ("Use test", "Requires evidence of how the model is used in decisions; none exists."),
]


@dataclass
class Splits:
    train: pd.DataFrame
    test: pd.DataFrame
    oot: pd.DataFrame | None
    date_column: str | None

    def sizes(self) -> dict:
        return {"train": len(self.train), "test": len(self.test),
                "oot": 0 if self.oot is None else len(self.oot)}


def split_dataset(frame: pd.DataFrame, target: str, date_column: str | None = None, *,
                  test_fraction: float = .25, oot_fraction: float = .2,
                  seed: int = 42) -> Splits:
    if date_column and date_column in frame:
        development, oot = oot_split(frame, date_column, test_fraction=oot_fraction)
    else:
        development, oot, date_column = frame, None, None
    train, test = train_test_split(development, test_size=test_fraction, random_state=seed,
                                   stratify=development[target])
    return Splits(train.copy(), test.copy(), None if oot is None else oot.copy(), date_column)


def _features(frame: pd.DataFrame, target: str, date_column: str | None) -> pd.DataFrame:
    return frame.drop(columns=[c for c in (target, date_column) if c and c in frame])


def data_review(splits: Splits, target: str, features: list[str]) -> dict:
    samples = {"train": splits.train, "test": splits.test, "oot": splits.oot}
    per_split = []
    for name, sample in samples.items():
        if sample is None:
            continue
        row = {"sample": name, "rows": len(sample), "defaults": int(sample[target].sum()),
               "default_rate": float(sample[target].mean())}
        if splits.date_column:
            dates = pd.to_datetime(sample[splits.date_column])
            row["start_date"], row["end_date"] = str(dates.min().date()), str(dates.max().date())
        per_split.append(row)
    full = pd.concat([s for s in samples.values() if s is not None])
    columns = []
    for f in features:
        series = full[f]
        columns.append({"feature": f, "dtype": str(series.dtype),
                        "missing_rate": float(series.isna().mean()),
                        "distinct": int(series.nunique()),
                        "constant": bool(series.nunique(dropna=True) <= 1)})
    return {"status": ASSESSED, "splits": per_split, "features": columns,
            "duplicate_rows": int(full.duplicated().sum())}


def implementation_checks(model, test: pd.DataFrame, target: str, date_column: str | None,
                          metadata: dict, data_sha256_matches: bool | None) -> dict:
    X = _features(test, target, date_column)
    checks = []
    names = list(getattr(model, "feature_names", []))
    checks.append(("Feature contract matches validation data",
                   set(names) <= set(X.columns), f"{len(names)} model features"))
    p1 = predict_pd(model, X)
    checks.append(("Predicted PDs within [0, 1]", bool(np.all((p1 >= 0) & (p1 <= 1))), ""))
    checks.append(("Scoring is deterministic", bool(np.array_equal(p1, predict_pd(model, X))), ""))
    scores = probability_to_score(np.sort(p1))
    checks.append(("Score decreases as PD increases", bool(np.all(np.diff(scores) <= 0)),
                   "PDO convention: higher PD, lower score"))
    checks.append(("Class weighting disabled (PDs not re-weighted)",
                   metadata.get("class_weight") is None,
                   f"class_weight={metadata.get('class_weight')!r}"))
    if data_sha256_matches is None:
        checks.append(("Validation data matches recorded training data", None,
                       "no training data fingerprint to compare (external data)"))
    else:
        checks.append(("Validation data matches recorded training data", data_sha256_matches,
                       "regenerated data fingerprint compared with training metadata"))
    rows = [{"check": c, "passed": ok, "detail": d} for c, ok, d in checks]
    status = ASSESSED if all(r["passed"] is not None for r in rows) else PARTIAL
    return {"status": status, "checks": rows}


def sensitivity(model, test: pd.DataFrame, target: str, date_column: str | None,
                specs, shift: float = 0.10) -> dict:
    """Mean PD change when each numeric feature is scaled by (1 +/- shift)."""
    X = _features(test, target, date_column)
    base = float(predict_pd(model, X).mean())
    rows = []
    for spec in specs:
        if spec.kind != "numeric":
            continue
        up, down = X.copy(), X.copy()
        up[spec.name] = up[spec.name] * (1 + shift)
        down[spec.name] = down[spec.name] * (1 - shift)
        rows.append({"feature": spec.name,
                     "mean_pd_up": float(predict_pd(model, up).mean()) - base,
                     "mean_pd_down": float(predict_pd(model, down).mean()) - base})
    rows.sort(key=lambda r: abs(r["mean_pd_up"]) + abs(r["mean_pd_down"]), reverse=True)
    return {"status": ASSESSED, "shift": shift, "base_mean_pd": base, "features": rows,
            "note": "Categorical features are not perturbed."}


def stability(model, splits: Splits, target: str, features: list[str], thresholds: dict) -> dict:
    ref = splits.train
    comparisons = [("test", splits.test)] + ([("oot", splits.oot)] if splits.oot is not None else [])
    ref_pd = predict_pd(model, _features(ref, target, splits.date_column))
    rows, score_psi = [], {}
    for name, sample in comparisons:
        X = _features(sample, target, splits.date_column)
        value = psi(ref_pd, predict_pd(model, X))
        score_psi[name] = {"psi": value, "level": alert_level(
            value, thresholds["psi_investigate"], thresholds["psi_escalate"])}
        for f in features:
            csi = psi(ref[f], sample[f])
            rows.append({"feature": f, "comparison": f"train→{name}", "csi": csi,
                         "level": alert_level(csi, thresholds["csi_investigate"],
                                              thresholds["csi_escalate"])})
    status = ASSESSED if splits.oot is not None else PARTIAL
    note = ("" if splits.oot is not None else
            "No date column: stability measured train→test only, which tests sampling "
            "noise rather than population change.")
    return {"status": status, "score_psi": score_psi, "features": rows, "note": note}


def woe_iv_review(train: pd.DataFrame, target: str, features: list[str]) -> dict:
    rows = []
    for f in features:
        try:
            r = calculate_woe_iv(train[f], train[target])
            rows.append({"feature": f, "iv": r.iv, "bins": len(r.table), "kind": r.kind,
                         "monotonic": r.monotonic})
        except ValueError as exc:
            rows.append({"feature": f, "iv": None, "bins": 0, "kind": "n/a",
                         "monotonic": None, "error": str(exc)})
    rows.sort(key=lambda r: -(r["iv"] or 0))
    return {"status": ASSESSED, "features": rows,
            "note": "Univariate review only; the champion uses standardised raw features, "
                    "not WoE-transformed inputs."}


def automated_findings(results: dict, data_source: str) -> list[dict]:
    findings = []

    def add(severity, title, description):
        findings.append({"id": f"AF-{len(findings) + 1:02d}", "severity": severity,
                         "title": title, "description": description})

    if data_source == "synthetic":
        add("high", "Synthetic data only",
            "All results use generated data; they say nothing about a real portfolio.")
    for check in results["implementation"]["checks"]:
        if check["passed"] is False:
            add("high", f"Implementation check failed: {check['check']}", check["detail"])
    if results["implementation"]["checks"][-1]["passed"] is None:
        add("medium", "Independence of validation data not verified",
            "External data was supplied; overlap with training data could not be checked.")
    ratio = results["calibration"].get("calibration_ratio")
    if ratio is not None and not 0.8 <= ratio <= 1.25:
        add("high", "Calibration-in-the-large outside tolerance",
            f"Mean predicted PD / observed default rate = {ratio:.2f} (tolerance 0.80–1.25).")
    auroc = results["discrimination"]["auroc"]
    if auroc < 0.6:
        add("medium", "Weak discrimination", f"Held-out AUROC {auroc:.3f} is below 0.60.")
    oot = results.get("oot") or {}
    if oot.get("status") == ASSESSED and auroc - oot["auroc"] > 0.05:
        add("medium", "Performance deterioration out of time",
            f"AUROC falls from {auroc:.3f} (test) to {oot['auroc']:.3f} (OOT).")
    for row in results["stability"]["features"]:
        if row["level"] == "escalate":
            add("medium", f"Population shift in {row['feature']}",
                f"CSI {row['csi']:.3f} ({row['comparison']}) is above the escalate threshold.")
    for row in results.get("challenger") or []:
        if row["model"] != "champion" and row["auroc"] - auroc > 0.02:
            add("low", f"Challenger {row['model']} outperforms champion",
                f"Challenger AUROC {row['auroc']:.3f} vs champion {auroc:.3f}.")
    pma = results.get("pma") or {}
    if pma.get("overdue_review") or pma.get("expired"):
        add("medium", "Post-model adjustments need attention",
            f"Overdue review: {pma.get('overdue_review')}; expired: {pma.get('expired')}.")
    return findings


def run_validation(model, splits: Splits, *, target: str, metadata: dict,
                   thresholds: dict, data_source: str, data_sha256_matches: bool | None,
                   challengers: dict | None = None, pma_summary: dict | None = None,
                   bootstrap: int = 200) -> dict:
    features = list(model.feature_names)
    test_X = _features(splits.test, target, splits.date_column)
    y_test, p_test = splits.test[target].astype(int), predict_pd(model, test_X)
    results: dict = {
        "data_source": data_source, "split_sizes": splits.sizes(),
        "data_review": data_review(splits, target, features),
        "implementation": implementation_checks(model, splits.test, target, splits.date_column,
                                                metadata, data_sha256_matches),
        "discrimination": {"status": ASSESSED,
                           **discrimination(y_test, p_test, bootstrap=bootstrap)},
        "calibration": {"status": ASSESSED, **calibration(y_test, p_test)},
        "stability": stability(model, splits, target, features, thresholds),
        "sensitivity": sensitivity(model, splits.test, target, splits.date_column, model.specs),
        "woe_iv": woe_iv_review(splits.train, target, features),
        "pma": pma_summary,
    }
    if splits.oot is not None:
        oot_X = _features(splits.oot, target, splits.date_column)
        results["oot"] = evaluate_sample(splits.oot[target], predict_pd(model, oot_X),
                                         splits.oot[splits.date_column])
        results["oot"]["status"] = (ASSESSED if results["oot"]["status"] == "assessed"
                                    else NOT_ASSESSED)
        scored = pd.concat([splits.test, splits.oot])
        results["vintage"] = vintage_performance(
            scored[splits.date_column], scored[target],
            predict_pd(model, _features(scored, target, splits.date_column)))
    else:
        results["oot"], results["vintage"] = None, None
    if challengers:
        predictions = {"champion": p_test}
        predictions.update({name: predict_pd(m, test_X) for name, m in challengers.items()})
        results["challenger"] = benchmark(y_test, predictions)
    else:
        results["challenger"] = None

    assessment = [
        ("Data review", ASSESSED, "Row counts, default rates, missingness, duplicates."),
        ("Implementation verification", results["implementation"]["status"],
         "Contract, PD bounds, determinism, score direction, class weighting, data fingerprint."),
        ("Discrimination", ASSESSED, "AUROC, Gini, KS with bootstrap CI on the held-out test sample."),
        ("Calibration", ASSESSED, ("Brier, log loss, calibration-in-the-large, grouped "
         "Hosmer–Lemeshow-style statistic (no p-value) on the held-out test sample.")),
        ("Stability (PSI/CSI)", results["stability"]["status"], results["stability"]["note"]
         or "Score PSI and feature CSI, train→test and train→OOT."),
        ("Out-of-time", results["oot"]["status"] if results["oot"] else NOT_ASSESSED,
         (results["oot"] or {}).get("reason", "Latest applications by date.")
         if results["oot"] else "No date column in the data."),
        ("Vintage performance", ASSESSED if results["vintage"] else NOT_ASSESSED,
         "Quarterly cohorts of test + OOT applications." if results["vintage"]
         else "No date column in the data."),
        ("Champion–challenger", ASSESSED if results["challenger"] else NOT_ASSESSED,
         "Same held-out test cohort." if results["challenger"] else "No challenger artefact."),
        ("Sensitivity", ASSESSED, "±10% shift of each numeric feature."),
        ("WoE / IV review", ASSESSED, "Univariate, training sample."),
        ("Post-model adjustments", ASSESSED if pma_summary is not None else NOT_ASSESSED,
         "Register status as of the validation date." if pma_summary is not None
         else "Register unavailable."),
        ("Real-data validation", NOT_ASSESSED if data_source == "synthetic" else PARTIAL,
         "Synthetic data only." if data_source == "synthetic"
         else "Real data supplied; representativeness and lineage not reviewed."),
    ]
    results["assessment"] = [{"area": a, "status": s, "basis": b} for a, s, b in assessment]
    results["assessment"] += [{"area": a, "status": NOT_ASSESSED, "basis": r}
                              for a, r in ALWAYS_NOT_ASSESSED]
    results["not_assessed"] = [r for r in results["assessment"] if r["status"] == NOT_ASSESSED]
    results["findings"] = automated_findings(results, data_source)
    return results
