"""Write the canonical synthetic credit dataset to CSV (same generator as training)."""
from __future__ import annotations

import argparse
from pathlib import Path

from credit_risk_model.data import synthetic


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--drift", type=float, default=0.0,
                   help="Population shift (0 = none); shifts int_rate, dti, home_ownership")
    p.add_argument("--start-date", default="2018-01-01")
    p.add_argument("--months", type=int, default=36)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    frame = synthetic.generate(args.rows, seed=args.seed, drift=args.drift,
                               start_date=args.start_date, months=args.months)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"Wrote {len(frame)} rows ({frame[synthetic.TARGET].mean():.2%} defaults) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
