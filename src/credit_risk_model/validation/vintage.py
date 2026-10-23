"""Vintage-level performance: outcomes and predictions grouped by origination period."""
import numpy as np
import pandas as pd

from .oot import evaluate_sample


def vintage_performance(dates, y_true, y_prob, *, freq: str = "Q") -> list[dict]:
    """One row per origination period with rows, default rate, mean PD and AUROC/Brier.

    Periods with a single outcome class are reported with status ``not_assessed``.
    """
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True), errors="coerce")
    if dates.isna().any():
        raise ValueError("Vintage dates contain missing or unparseable values")
    frame = pd.DataFrame({"period": dates.dt.to_period(freq).astype(str),
                          "y": np.asarray(y_true).astype(int), "p": np.asarray(y_prob, float),
                          "date": dates})
    rows = []
    for period, group in frame.groupby("period", sort=True):
        rows.append({"vintage": period, **evaluate_sample(group["y"], group["p"], group["date"])})
    return rows
