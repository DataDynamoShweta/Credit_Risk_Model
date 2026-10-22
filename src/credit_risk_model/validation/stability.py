"""Population Stability Index (PSI) and simple Characteristic Stability Index (CSI)."""
import numpy as np
import pandas as pd


def _distribution(values, edges):
    values = pd.Series(values).dropna()
    counts = np.histogram(values, bins=edges)[0].astype(float)
    dist = counts / max(counts.sum(), 1)
    return np.clip(dist, 1e-6, None)

def psi(reference, current, *, bins: int = 10) -> float:
    ref, cur = pd.Series(reference).dropna(), pd.Series(current).dropna()
    if ref.empty or cur.empty:
        raise ValueError("Reference and current samples must be non-empty")
    if not pd.api.types.is_numeric_dtype(ref) or not pd.api.types.is_numeric_dtype(cur):
        levels = sorted(set(ref.astype(str)) | set(cur.astype(str)))
        r = ref.astype(str).value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy()
        c = cur.astype(str).value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy()
        r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins+1)))
        if len(edges) < 2:
            edges = np.array([ref.min()-0.5, ref.max()+0.5])
        else:
            edges[0], edges[-1] = -np.inf, np.inf
        r, c = _distribution(ref, edges), _distribution(cur, edges)
    r, c = r/r.sum(), c/c.sum()
    return float(np.sum((c-r) * np.log(c/r)))

def csi_by_feature(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    shared = [c for c in reference.columns if c in current.columns]
    return {c: psi(reference[c], current[c]) for c in shared}
