"""Calibration metrics and data for reliability diagrams."""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

def calibration(y_true, y_prob, *, groups: int = 10):
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-8, 1-1e-8)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("Inputs must have equal non-zero length")
    bins = pd.qcut(p, q=min(groups, len(np.unique(p))), duplicates="drop")
    table = pd.DataFrame({"y": y, "p": p, "bin": bins}).groupby("bin", observed=False).agg(
        count=("y", "size"), observed_rate=("y", "mean"), mean_pd=("p", "mean")).reset_index()
    # Hosmer-Lemeshow-style grouped statistic; asymptotic p-value intentionally omitted.
    expected_bad = table["count"] * table["mean_pd"]
    expected_good = table["count"] * (1-table["mean_pd"])
    observed_bad = table["count"] * table["observed_rate"]
    observed_good = table["count"] - observed_bad
    denom_bad = expected_bad.clip(lower=1e-8)
    denom_good = expected_good.clip(lower=1e-8)
    hl = float((((observed_bad-expected_bad)**2/denom_bad) +
                ((observed_good-expected_good)**2/denom_good)).sum())
    return {"brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0,1])),
            "hosmer_lemeshow_statistic": hl,
            "reliability": table.drop(columns="bin").to_dict(orient="records")}
