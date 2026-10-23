"""Generate synthetic credit data for demo & CI."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--drift", type=float, default=0.0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    n = args.rows
    df = pd.DataFrame({
        "loan_amnt": rng.normal(15000, 5000, n).clip(1000, 40000),
        "int_rate": rng.normal(11 + args.drift * 10, 3, n).clip(4, 25),
        "annual_inc": rng.normal(62000, 20000, n).clip(15000, 200000),
        "dti": rng.normal(18 + args.drift * 10, 6, n).clip(0, 60),
        "emp_length": rng.integers(0, 11, n),
    })
    logit = (-3.0 + 0.00002 * df["loan_amnt"]
             + 0.10 * (df["int_rate"] - 11)
             - 0.00001 * (df["annual_inc"] - 62000)
             + 0.05 * (df["dti"] - 18))
    pd_prob = 1 / (1 + np.exp(-logit))
    df["default"] = (rng.random(n) < pd_prob).astype(int)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {n} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
