"""Model wrappers and helpers."""
import numpy as np


def predict_pd(model, X) -> np.ndarray:
    """Positive-class probabilities from a ChampionModel, sklearn estimator or wrapper.

    Accepts both 1-D outputs (the project's ChampionModel) and the 2-D ``(n, 2)`` output
    of scikit-learn estimators.
    """
    names = getattr(model, "feature_names_", None)
    if names is not None and hasattr(X, "loc"):
        X = X[names]
    proba = np.asarray(model.predict_proba(X), dtype=float)
    if proba.ndim == 2:
        if proba.shape[1] != 2:
            raise ValueError("predict_proba must return two class columns for a binary model")
        proba = proba[:, 1]
    if proba.ndim != 1:
        raise ValueError("Unexpected predict_proba output shape")
    return proba
