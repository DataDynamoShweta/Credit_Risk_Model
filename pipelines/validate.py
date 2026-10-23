"""Run validation against a labelled CSV and write a Markdown report."""
import argparse
import json
from pathlib import Path

import pandas as pd

from credit_risk_model.governance.report_builder import build_markdown
from credit_risk_model.models.champion import ChampionModel
from credit_risk_model.validation.calibration import calibration
from credit_risk_model.validation.discrimination import discrimination


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data")
    p.add_argument("--model", default="artifacts/champion.joblib")
    p.add_argument("--target", default="target")
    p.add_argument("--report", default="reports/validation_report.md")
    p.add_argument("--export-pdf", default=None, help="Optional PDF output path")
    args=p.parse_args()
    if args.data:
        frame=pd.read_csv(args.data)
    else:
        from train import demo_data
        frame=demo_data()
    model=ChampionModel.load(args.model)
    y=frame[args.target].astype(int)
    probs=model.predict_proba(frame.drop(columns=[args.target]))
    results={"discrimination":discrimination(y,probs,bootstrap=200),
             "calibration":calibration(y,probs)}
    build_markdown(args.report, model_name="Logistic regression champion",
                   purpose="Probability-of-default estimation", results=results)
    if args.export_pdf:
        from credit_risk_model.governance.pdf_export import export_pdf
        try:
            export_pdf(Path(args.report), Path(args.export_pdf))
        except Exception as exc:  # noqa: BLE001 - PDF export is best-effort
            print(f"[validate] PDF skipped: {exc}")
    print(json.dumps(results, indent=2))
if __name__ == "__main__":
    main()
