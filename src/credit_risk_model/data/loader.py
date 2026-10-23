"""Small, explicit CSV loading helpers."""
import csv
from pathlib import Path

import pandas as pd


def _header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return next(csv.reader(fh), [])


def load_csv(path: str | Path, *, target: str | None = None, **read_kwargs) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    header = _header(path)
    if not header:
        raise ValueError("Dataset is empty")
    # Checked on the raw header: pandas would silently rename duplicates to "a.1".
    duplicates = sorted({c for c in header if header.count(c) > 1})
    if duplicates:
        raise ValueError(f"Dataset contains duplicate column names: {duplicates}")
    try:
        frame = pd.read_csv(path, **read_kwargs)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("Dataset is empty") from exc
    if frame.empty:
        raise ValueError("Dataset is empty")
    if target and target not in frame.columns:
        raise ValueError(f"Target column {target!r} is missing")
    return frame
