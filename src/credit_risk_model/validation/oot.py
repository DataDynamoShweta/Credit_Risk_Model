"""Chronological out-of-time evaluation."""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from ..models import predict_pd


def _dates(frame: pd.DataFrame, date_column: str) -> pd.Series:
    if date_column not in frame:
        raise ValueError(f"Missing date column: {date_column}")
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"Date column {date_column!r} has {int(dates.isna().sum())} "
                         "missing or unparseable values")
    return dates


def oot_split(frame: pd.DataFrame, date_column: str, *, test_fraction: float = .2):
    """Split into earlier and later samples without letting one date fall on both sides.

    Rows are sorted by date (the input may be unsorted). The cut-off is the date at the
    requested fraction; every row on that date goes to the later (test) sample.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if len(frame) < 2:
        raise ValueError("Need at least two rows for an out-of-time split")
    data = frame.copy()
    data[date_column] = _dates(data, date_column)
    data = data.sort_values(date_column, kind="stable")
    position = max(1, min(len(data) - 1, int(len(data) * (1 - test_fraction))))
    cutoff = data[date_column].iloc[position]
    train, test = data[data[date_column] < cutoff], data[data[date_column] >= cutoff]
    if train.empty:
        raise ValueError("All rows share the cut-off date; cannot form an out-of-time split")
    return train.copy(), test.copy()


def evaluate_sample(y, p, dates: pd.Series | None = None) -> dict:
    """Metrics for one labelled sample; reports 'not_assessed' when AUROC is undefined."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    result = {"rows": len(y), "default_rate": float(y.mean()) if len(y) else None,
              "mean_pd": float(p.mean()) if len(p) else None}
    if dates is not None and len(dates):
        result["start_date"] = str(pd.Timestamp(dates.min()).date())
        result["end_date"] = str(pd.Timestamp(dates.max()).date())
    if len(y) == 0:
        return {**result, "status": "not_assessed", "reason": "no rows in window"}
    result["brier"] = float(brier_score_loss(y, p))
    if len(np.unique(y)) < 2:
        return {**result, "status": "not_assessed",
                "reason": "only one outcome class in window; AUROC undefined"}
    return {**result, "status": "assessed", "auroc": float(roc_auc_score(y, p))}


def evaluate_oot(model, frame, target: str, date_column: str, *, test_fraction: float = .2):
    """Score the later part of ``frame`` with ``model`` (ChampionModel or sklearn-like)."""
    _, test = oot_split(frame, date_column, test_fraction=test_fraction)
    X = test.drop(columns=[target, date_column])
    p = predict_pd(model, X)
    return evaluate_sample(test[target], p, test[date_column])
