"""Dataset loading shared by the train, validate and monitor pipelines."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import synthetic
from .loader import load_csv
from .loan_data_adapter import adapt

ADAPTERS = ("none", "lending_club")


def load_dataset(*, data: str | None, target: str = "target", date_column: str | None = None,
                 adapter: str = "none", rows: int = 20000, seed: int = 42,
                 drift: float = 0.0) -> tuple[pd.DataFrame, dict]:
    """Return (frame, source) where the frame has a 0/1 ``target`` column.

    ``source`` describes where the data came from so it can be recorded in metadata.
    """
    if data is None:
        frame = synthetic.generate(rows, seed=seed, drift=drift)
        return frame, {"kind": "synthetic", "rows": rows, "seed": seed, "drift": drift,
                       "date_column": synthetic.DATE_COLUMN, "target": synthetic.TARGET}
    if adapter == "lending_club":
        raw = load_csv(data, low_memory=False)
        frame, report = adapt(raw)
        return frame, {"kind": "lending_club", "path": str(Path(data)), "adapter": report,
                       "date_column": "application_date", "target": "target"}
    if adapter != "none":
        raise ValueError(f"Unknown adapter {adapter!r}; choose from {ADAPTERS}")
    frame = load_csv(data, target=target)
    if target != "target":
        frame = frame.rename(columns={target: "target"})
    if date_column and date_column in frame:
        frame[date_column] = pd.to_datetime(frame[date_column], errors="raise")
    return frame, {"kind": "csv", "path": str(Path(data)), "date_column": date_column,
                   "target": "target"}
