"""Canonical synthetic credit dataset used by training, validation, the API and monitoring.

The data is invented. It has a dated application column so that out-of-time and vintage
analysis can run, a default rate of roughly 5%, and a mix of numeric and categorical
features. ``drift`` shifts interest rate, DTI and the home-ownership mix to simulate a
changed population.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "target"
DATE_COLUMN = "application_date"
NUMERIC_FEATURES = ["loan_amnt", "int_rate", "annual_inc", "dti", "emp_length"]
CATEGORICAL_FEATURES = ["home_ownership"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def generate(rows: int = 5000, *, seed: int = 42, drift: float = 0.0,
             start_date: str = "2018-01-01", months: int = 36) -> pd.DataFrame:
    if rows < 10:
        raise ValueError("rows must be at least 10")
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start_date)
    offsets = rng.integers(0, max(months, 1) * 30, rows)
    dates = (start + pd.to_timedelta(offsets, unit="D")).normalize()
    home_p = np.array([0.45, 0.40, 0.15]) + drift * np.array([0.3, -0.2, -0.1])
    home_p = np.clip(home_p, 0.01, None)
    frame = pd.DataFrame({
        DATE_COLUMN: dates,
        "loan_amnt": rng.normal(15000, 5000, rows).clip(1000, 40000).round(0),
        "int_rate": rng.normal(11 + drift * 10, 3, rows).clip(4, 25).round(2),
        "annual_inc": rng.lognormal(11.0, 0.35, rows).clip(15000, 250000).round(0),
        "dti": rng.normal(18 + drift * 10, 6, rows).clip(0, 60).round(2),
        "emp_length": rng.integers(0, 11, rows),
        "home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN"], rows, p=home_p / home_p.sum()),
    })
    logit = (-3.05 + 0.00002 * (frame["loan_amnt"] - 15000)
             + 0.2 * (frame["int_rate"] - 11)
             - 0.000012 * (frame["annual_inc"] - 62000)
             + 0.04 * (frame["dti"] - 18)
             - 0.04 * frame["emp_length"]
             + frame["home_ownership"].map({"RENT": 0.35, "MORTGAGE": -0.2, "OWN": 0.0}))
    probability = 1 / (1 + np.exp(-logit))
    frame[TARGET] = (rng.random(rows) < probability).astype(int)
    return frame.sort_values(DATE_COLUMN, kind="stable").reset_index(drop=True)
