"""SHAP-based explainability (optional dependency) for the champion model.

For the logistic-regression ChampionModel the explanation is computed in the model's
transformed (scaled, one-hot) space with a linear explainer and then summed back to the
original feature names. Contributions are on the log-odds scale.

SHAP values describe how this model's output moves relative to a background sample. They
are not adverse-action reason codes: using them for customer-facing reasons would need a
documented, reviewed methodology, which this project does not provide.
"""
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


def _require():
    if not _HAS_SHAP:
        raise RuntimeError("SHAP not installed. `pip install -e '.[explain]'`.")


def _original_feature_index(model, transformed_names: list[str]) -> list[str]:
    """Map each transformed column name (e.g. 'cat__home_ownership_RENT') to its source."""
    mapping = []
    for name in transformed_names:
        bare = name.split("__", 1)[-1]
        match = next((f for f in sorted(model.feature_names, key=len, reverse=True)
                      if bare == f or bare.startswith(f + "_")), bare)
        mapping.append(match)
    return mapping


def _shap_by_feature(model, background: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    _require()
    pre, est = model.pipeline.named_steps["preprocess"], model.pipeline.named_steps["model"]
    to_dense = (lambda m: m.toarray() if hasattr(m, "toarray") else np.asarray(m))
    bg, data = to_dense(pre.transform(background[model.feature_names])), \
        to_dense(pre.transform(X[model.feature_names]))
    explainer = shap.LinearExplainer(est, bg)
    values = np.asarray(explainer.shap_values(data))
    columns = _original_feature_index(model, list(pre.get_feature_names_out()))
    return pd.DataFrame(values, columns=columns).T.groupby(level=0).sum().T


def global_importance(model, X: pd.DataFrame, top_n: int = 15) -> list[dict[str, Any]]:
    """Mean absolute SHAP value per original feature, largest first."""
    _require()
    values = _shap_by_feature(model, X, X)
    importance = values.abs().mean().sort_values(ascending=False).head(top_n)
    return [{"feature": f, "mean_abs_shap": round(float(v), 6)} for f, v in importance.items()]


def local_explanation(model, background: pd.DataFrame, X_row: pd.DataFrame,
                      top_n: int = 8) -> list[dict[str, Any]]:
    """Per-feature contributions for one row, relative to ``background``."""
    _require()
    values = _shap_by_feature(model, background, X_row).iloc[0]
    contributions = [{"feature": f, "shap_value": round(float(v), 6),
                      "direction": "increases_pd" if v > 0 else "decreases_pd"}
                     for f, v in values.items()]
    contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
    return contributions[:top_n]
