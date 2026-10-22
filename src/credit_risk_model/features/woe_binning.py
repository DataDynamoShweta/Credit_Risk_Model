"""Transparent numeric/categorical WoE and IV utilities.

Bins are quantile-based for numeric features. Monotonicity is assessed and reported;
the implementation does not silently force monotonic bins or claim optimal binning.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class WoEBinResult:
    table: pd.DataFrame
    iv: float
    monotonic: bool

def calculate_woe_iv(x: pd.Series, y: pd.Series, *, bins: int = 5,
                     smoothing: float = 0.5) -> WoEBinResult:
    if len(x) != len(y):
        raise ValueError("x and y must have equal length")
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    if data.empty or data["y"].nunique() != 2:
        raise ValueError("Need non-empty data and a binary target with both classes")
    if not set(data["y"].unique()).issubset({0, 1, False, True}):
        raise ValueError("Target must be encoded as 0/1")
    if pd.api.types.is_numeric_dtype(data["x"]) and data["x"].nunique() > 1:
        try:
            data["bin"] = pd.qcut(data["x"], q=min(bins, data["x"].nunique()),
                                  duplicates="drop").astype(str)
        except ValueError:
            data["bin"] = data["x"].astype(str)
    else:
        data["bin"] = data["x"].astype(str)
    grouped = data.groupby("bin", observed=False)["y"].agg(["count", "sum"])
    grouped = grouped.rename(columns={"sum": "bad"})
    grouped["good"] = grouped["count"] - grouped["bad"]
    total_bad, total_good = grouped["bad"].sum(), grouped["good"].sum()
    n = len(grouped)
    grouped["dist_bad"] = (grouped["bad"] + smoothing) / (total_bad + smoothing*n)
    grouped["dist_good"] = (grouped["good"] + smoothing) / (total_good + smoothing*n)
    grouped["woe"] = np.log(grouped["dist_good"] / grouped["dist_bad"])
    grouped["iv_component"] = (grouped["dist_good"] - grouped["dist_bad"]) * grouped["woe"]
    rates = (grouped["bad"] / grouped["count"]).replace([np.inf, -np.inf], np.nan).dropna()
    diffs = np.diff(rates.to_numpy())
    monotonic = bool(len(diffs) == 0 or np.all(diffs >= -1e-12) or np.all(diffs <= 1e-12))
    return WoEBinResult(grouped.reset_index(), float(grouped["iv_component"].sum()), monotonic)
