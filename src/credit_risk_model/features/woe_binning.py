"""Transparent numeric/categorical WoE and IV utilities.

Numeric features use quantile bins kept in numeric order, so monotonicity is assessed over
correctly ordered bins. Categorical features are ordered by label and their monotonicity is
reported as ``None`` (not applicable). The implementation does not force monotonic bins or
claim optimal binning.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WoEBinResult:
    table: pd.DataFrame
    iv: float
    monotonic: bool | None  # None for categorical features, where order has no meaning
    kind: str = "numeric"


def _numeric_bins(x: pd.Series, bins: int) -> pd.Series:
    """Ordered categorical bins: quantile intervals, or the distinct values if qcut fails."""
    if x.nunique() > 1:
        try:
            return pd.qcut(x, q=min(bins, x.nunique()), duplicates="drop")
        except ValueError:
            pass
    levels = sorted(x.unique())
    return pd.Categorical(x, categories=levels, ordered=True)


def calculate_woe_iv(x: pd.Series, y: pd.Series, *, bins: int = 5,
                     smoothing: float = 0.5) -> WoEBinResult:
    if len(x) != len(y):
        raise ValueError("x and y must have equal length")
    data = pd.DataFrame({"x": pd.Series(x).to_numpy(), "y": pd.Series(y).to_numpy()}).dropna()
    if data.empty or data["y"].nunique() != 2:
        raise ValueError("Need non-empty data and a binary target with both classes")
    if not set(data["y"].unique()).issubset({0, 1}):
        raise ValueError("Target must be encoded as 0/1")
    numeric = pd.api.types.is_numeric_dtype(data["x"]) and not pd.api.types.is_bool_dtype(data["x"])
    if numeric:
        data["bin"] = _numeric_bins(data["x"], bins)
    else:
        labels = data["x"].astype(str)
        data["bin"] = pd.Categorical(labels, categories=sorted(labels.unique()), ordered=True)
    # Grouping on an ordered categorical keeps bins in their natural (numeric) order.
    grouped = data.groupby("bin", observed=True)["y"].agg(["count", "sum"])
    grouped = grouped.rename(columns={"sum": "bad"})
    grouped["good"] = grouped["count"] - grouped["bad"]
    total_bad, total_good = grouped["bad"].sum(), grouped["good"].sum()
    n = len(grouped)
    grouped["bad_rate"] = grouped["bad"] / grouped["count"]
    grouped["dist_bad"] = (grouped["bad"] + smoothing) / (total_bad + smoothing*n)
    grouped["dist_good"] = (grouped["good"] + smoothing) / (total_good + smoothing*n)
    grouped["woe"] = np.log(grouped["dist_good"] / grouped["dist_bad"])
    grouped["iv_component"] = (grouped["dist_good"] - grouped["dist_bad"]) * grouped["woe"]
    monotonic = None
    if numeric:
        diffs = np.diff(grouped["bad_rate"].to_numpy())
        monotonic = bool(len(diffs) == 0 or np.all(diffs >= -1e-12) or np.all(diffs <= 1e-12))
    table = grouped.reset_index()
    table["bin"] = table["bin"].astype(str)
    return WoEBinResult(table, float(grouped["iv_component"].sum()), monotonic,
                        "numeric" if numeric else "categorical")
