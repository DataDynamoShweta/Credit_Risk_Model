"""SHAP-based explainability — optional dependency, aggregated outputs."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False


def is_available() -> bool:
    return _HAS_SHAP


def global_importance(
    model, X: pd.DataFrame, top_n: int = 15,
) -> list[dict[str, Any]]:
    if not _HAS_SHAP:
        raise RuntimeError("SHAP not installed. `pip install -e '.[explain]'`.")
    explainer = shap.Explainer(model, X)
    values = explainer(X).values
    if values.ndim == 3:
        values = values.mean(axis=2)
    importance = np.abs(values).mean(axis=0)
    order = np.argsort(importance)[::-1][:top_n]
    return [
        {"feature": str(X.columns[i]),
         "mean_abs_shap": round(float(importance[i]), 6)}
        for i in order
    ]


def local_explanation(
    model, X_row: pd.DataFrame, top_n: int = 8,
) -> list[dict[str, Any]]:
    if not _HAS_SHAP:
        raise RuntimeError("SHAP not installed. `pip install -e '.[explain]'`.")
    explainer = shap.Explainer(model, X_row)
    sv = explainer(X_row)
    vals = sv.values[0]
    if vals.ndim > 1:
        vals = vals[:, -1]
    contributions = [
        {"feature": str(col), "shap_value": round(float(v), 6),
         "direction": "increases_risk" if v > 0 else "decreases_risk"}
        for col, v in zip(X_row.columns, vals)
    ]
    contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
    return contributions[:top_n]
