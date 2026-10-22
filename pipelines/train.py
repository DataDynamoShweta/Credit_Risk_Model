"""Train a reproducible demonstration model; accepts a user-supplied CSV."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from credit_risk_model.models.champion import ChampionModel

def demo_data(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    income = rng.lognormal(10.5, .5, n)
    debt_ratio = rng.beta(2, 5, n)
    age = rng.integers(21, 70, n)
    logit = -3.5 + 3.0*debt_ratio - .000012*income + .018*(35-age).clip(min=0)
    pd_ = 1/(1+np.exp(-logit))
    return pd.DataFrame({"income": income, "debt_ratio": debt_ratio, "age": age,
                         "employment_type": rng.choice(["salaried","self-employed","other"], n),
                         "target": rng.binomial(1, pd_)})

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", help="Optional CSV path")
    p.add_argument("--target", default="target")
    p.add_argument("--output-dir", default="artifacts")
    args = p.parse_args()
    frame = pd.read_csv(args.data) if args.data else demo_data()
    if args.target != "target" and args.target in frame:
        frame = frame.rename(columns={args.target:"target"})
    train, test = train_test_split(frame, test_size=.25, random_state=42, stratify=frame["target"])
    model = ChampionModel.fit(train, "target")
    X, y = test.drop(columns="target"), test["target"]
    auc = roc_auc_score(y, model.predict_proba(X))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    model.save(out/"champion.joblib")
    (out/"training_metrics.json").write_text(__import__("json").dumps({"holdout_auroc":float(auc)}, indent=2))
    print(f"Saved {out/'champion.joblib'}; holdout AUROC={auc:.4f}")
if __name__ == "__main__":
    main()
