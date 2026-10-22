"""Chronological out-of-time evaluation."""
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


def oot_split(frame: pd.DataFrame, date_column: str, *, test_fraction: float = .2):
    if date_column not in frame:
        raise ValueError(f"Missing date column: {date_column}")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    data = frame.copy()
    data[date_column] = pd.to_datetime(data[date_column], errors="raise")
    data = data.sort_values(date_column)
    split = max(1, min(len(data)-1, int(len(data)*(1-test_fraction))))
    return data.iloc[:split].copy(), data.iloc[split:].copy()

def evaluate_oot(model, frame, target: str, date_column: str, *, test_fraction: float = .2):
    _, test = oot_split(frame, date_column, test_fraction=test_fraction)
    X = test.drop(columns=[target, date_column])
    p = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict_proba(X)
    y = test[target].astype(int)
    return {"rows": len(test), "auroc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "start_date": str(test[date_column].min()),
            "end_date": str(test[date_column].max())}
