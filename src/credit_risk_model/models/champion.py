"""Interpretable logistic-regression champion model."""
from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class ChampionModel:
    pipeline: Pipeline
    feature_names: list[str]

    @classmethod
    def fit(cls, frame: pd.DataFrame, target: str = "target", *, random_state: int = 42):
        if target not in frame:
            raise ValueError(f"Missing target column: {target}")
        X, y = frame.drop(columns=[target]), frame[target]
        if y.nunique() != 2:
            raise ValueError("Target must contain exactly two classes")
        numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical = [c for c in X.columns if c not in numeric]
        numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                                 ("scaler", StandardScaler())])
        categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                      ("onehot", OneHotEncoder(handle_unknown="ignore"))])
        transform = ColumnTransformer([
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical)
        ])
        pipe = Pipeline([("preprocess", transform),
                         ("model", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                       random_state=random_state))])
        pipe.fit(X, y.astype(int))
        return cls(pipe, list(X.columns))

    def predict_proba(self, X):
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        X = X.reindex(columns=self.feature_names)
        return self.pipeline.predict_proba(X)[:, 1]

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
