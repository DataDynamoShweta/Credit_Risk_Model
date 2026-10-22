"""Small, explicit CSV loading helpers."""
from pathlib import Path

import pandas as pd


def load_csv(path: str | Path, *, target: str | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("Dataset is empty")
    if target and target not in frame.columns:
        raise ValueError(f"Target column {target!r} is missing")
    if frame.columns.duplicated().any():
        raise ValueError("Dataset contains duplicate column names")
    return frame
