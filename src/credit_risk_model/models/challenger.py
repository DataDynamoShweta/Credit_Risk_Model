"""Optional tree challenger; XGBoost/LightGBM are used only when installed."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_challenger(frame: pd.DataFrame, target: str = "target", algorithm: str = "histgb"):
    if target not in frame:
        raise ValueError(f"Missing target column: {target}")
    X, y = frame.drop(columns=[target]), frame[target].astype(int)
    num = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)
    ])
    if algorithm == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=.05,
                              eval_metric="logloss", random_state=42)
    elif algorithm == "lightgbm":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(n_estimators=200, max_depth=-1, learning_rate=.05,
                               random_state=42, verbosity=-1)
    else:
        model = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                       class_weight="balanced", random_state=42)
    pipe = Pipeline([("preprocess", prep), ("model", model)])
    pipe.fit(X, y)
    return pipe
