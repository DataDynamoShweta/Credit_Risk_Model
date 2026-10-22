"""Scheduled drift job: compare reference and current feature CSV files."""
import argparse
import json

import pandas as pd

from credit_risk_model.monitoring.alerts import build_alerts
from credit_risk_model.monitoring.drift import drift_summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--current", required=True)
    p.add_argument("--model-name", default="credit-risk-champion")
    p.add_argument("--output", default="reports/drift_summary.json")
    args=p.parse_args()
    summary=drift_summary(pd.read_csv(args.reference), pd.read_csv(args.current))
    result={"summary":summary,"alerts":build_alerts(summary,model_name=args.model_name)}
    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
