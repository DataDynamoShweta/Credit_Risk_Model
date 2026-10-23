"""Population Stability Index (PSI) and per-feature Characteristic Stability Index (CSI).

Continuous numeric features are binned on reference quantiles. Binary, low-cardinality
numeric and categorical features are compared on observed category proportions, because
quantile edges collapse to a single bin for them and would hide real shifts. Categories
seen in only one sample are kept and floored at a small epsilon.
"""
import numpy as np
import pandas as pd

EPSILON = 1e-6


def _proportions(values: pd.Series, levels: list) -> np.ndarray:
    dist = values.value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy()
    return np.clip(dist, EPSILON, None)


def _psi_from(r: np.ndarray, c: np.ndarray) -> float:
    r, c = r / r.sum(), c / c.sum()
    return float(np.sum((c - r) * np.log(c / r)))


def is_discrete(reference: pd.Series, current: pd.Series, *, max_levels: int = 20) -> bool:
    if not (pd.api.types.is_numeric_dtype(reference) and pd.api.types.is_numeric_dtype(current)):
        return True
    if pd.api.types.is_bool_dtype(reference) or pd.api.types.is_bool_dtype(current):
        return True
    return pd.concat([reference, current]).nunique() <= max_levels


def psi(reference, current, *, bins: int = 10, max_levels: int = 20) -> float:
    ref, cur = pd.Series(reference).dropna(), pd.Series(current).dropna()
    if ref.empty or cur.empty:
        raise ValueError("Reference and current samples must be non-empty")
    if is_discrete(ref, cur, max_levels=max_levels):
        ref_s, cur_s = ref.astype(str), cur.astype(str)
        levels = sorted(set(ref_s) | set(cur_s))
        return _psi_from(_proportions(ref_s, levels), _proportions(cur_s, levels))
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.clip(np.histogram(ref, bins=edges)[0] / len(ref), EPSILON, None)
    c = np.clip(np.histogram(cur, bins=edges)[0] / len(cur), EPSILON, None)
    return _psi_from(r, c)


def csi_by_feature(reference: pd.DataFrame, current: pd.DataFrame, *,
                   features: list[str] | None = None) -> dict:
    """PSI per feature. ``features`` restricts the comparison (e.g. to model inputs)."""
    shared = [c for c in reference.columns if c in current.columns]
    if features is not None:
        missing = [f for f in features if f not in shared]
        if missing:
            raise ValueError("features missing from reference or current data: "
                             + ", ".join(missing))
        shared = list(features)
    return {c: psi(reference[c], current[c]) for c in shared}
