"""Discrimination metrics with optional percentile bootstrap confidence intervals."""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def discrimination(y_true, y_prob, *, bootstrap: int = 0, confidence: float = .95,
                   random_state: int = 42):
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    if len(y) != len(p) or len(y) == 0 or len(np.unique(y)) != 2:
        raise ValueError("Inputs must have equal non-zero length and both target classes")
    auc = float(roc_auc_score(y, p))
    fpr, tpr, _ = roc_curve(y, p)
    ks = float(np.max(tpr - fpr))
    result = {"auroc": auc, "gini": 2*auc - 1, "ks": ks}
    if bootstrap > 0:
        rng = np.random.default_rng(random_state)
        values = []
        for _ in range(bootstrap):
            idx = rng.integers(0, len(y), len(y))
            if len(np.unique(y[idx])) == 2:
                values.append(roc_auc_score(y[idx], p[idx]))
        alpha = (1-confidence)/2
        if values:
            result["auroc_ci"] = [float(np.quantile(values, alpha)),
                                  float(np.quantile(values, 1-alpha))]
            result["bootstrap_valid_samples"] = len(values)
    return result
