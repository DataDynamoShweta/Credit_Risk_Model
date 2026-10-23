"""Interpretable logistic-regression champion model.

Class weighting is off by default. Re-weighting classes ("balanced") shifts every predicted
probability towards 0.5, so the outputs stop being usable as probabilities of default;
it is available only as an explicit training option and is recorded in the metadata.
"""
from __future__ import annotations

import hashlib
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .contract import FeatureSpec, infer_specs, validate_frame, validate_record


def data_fingerprint(frame: pd.DataFrame) -> str:
    """Order-sensitive SHA-256 of a DataFrame's values and column names."""
    digest = hashlib.sha256()
    digest.update("|".join(map(str, frame.columns)).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def library_versions() -> dict[str, str]:
    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__, "numpy": np.__version__}


def build_preprocessor(specs: list[FeatureSpec], *, scale: bool = True,
                       dense: bool = False) -> ColumnTransformer:
    numeric = [s.name for s in specs if s.kind == "numeric"]
    categorical = [s.name for s in specs if s.kind == "categorical"]
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense))])
    return ColumnTransformer([("num", Pipeline(numeric_steps), numeric),
                              ("cat", categorical_pipe, categorical)])


def split_xy(frame: pd.DataFrame, target: str, exclude: tuple[str, ...] = ()):
    if target not in frame:
        raise ValueError(f"Missing target column: {target}")
    y = frame[target]
    if y.isna().any():
        raise ValueError("Target contains missing values")
    if y.nunique() != 2 or not set(y.unique()) <= {0, 1}:
        raise ValueError("Target must contain exactly two classes encoded 0/1")
    X = frame.drop(columns=[target, *[c for c in exclude if c in frame]])
    if X.shape[1] == 0:
        raise ValueError("No feature columns left after removing target/excluded columns")
    return X, y.astype(int)


@dataclass
class ChampionModel:
    pipeline: Pipeline
    feature_names: list[str]
    feature_specs: list[FeatureSpec] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, frame: pd.DataFrame, target: str = "target", *, random_state: int = 42,
            class_weight: str | dict | None = None, exclude: tuple[str, ...] = (),
            model_name: str = "credit_risk_champion", model_version: str = "unregistered"):
        X, y = split_xy(frame, target, exclude)
        specs = infer_specs(X)
        pipe = Pipeline([("preprocess", build_preprocessor(specs)),
                         ("model", LogisticRegression(max_iter=2000, class_weight=class_weight,
                                                      random_state=random_state))])
        pipe.fit(X, y)
        metadata = {
            "model_name": model_name, "model_version": model_version,
            "algorithm": "logistic_regression", "class_weight": class_weight,
            "random_state": random_state, "features": [s.to_dict() for s in specs],
            "training_rows": len(X), "training_default_rate": float(y.mean()),
            "training_data_sha256": data_fingerprint(frame[[*X.columns, target]]),
            "library_versions": library_versions(),
            "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        return cls(pipe, list(X.columns), specs, metadata)

    @property
    def specs(self) -> list[FeatureSpec]:
        # Artefacts saved before feature specs existed fall back to "all numeric".
        return self.feature_specs or [FeatureSpec(n, "numeric") for n in self.feature_names]

    def predict_proba(self, X, *, allow_extra: bool = False) -> np.ndarray:
        """Return a 1-D array of probabilities of the positive (default) class."""
        if isinstance(X, dict):
            validate_record(X, self.specs, allow_extra=allow_extra)
            X = pd.DataFrame([X])
        X = validate_frame(pd.DataFrame(X), self.specs)
        return self.pipeline.predict_proba(X)[:, 1]

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
