# Credit Risk Model Validation Platform

A modular, testable credit-risk scorecard and independent model-validation toolkit designed around the responsibilities in the Skipton Model Validation / Model Risk role. It includes a logistic-regression champion, optional tree-based challenger, scorecard scaling, discrimination/calibration/stability/Out-of-Time (OOT) checks, a FastAPI scoring layer, audit logging, monitoring hooks, and governance reporting.

> **Important:** This repository is an engineering demonstration, not evidence of regulatory approval or compliance. SS1/23 alignment is a design objective; compliance requires documented policies, evidence, independent review, and approval by the relevant institution. Standardised/IRB capital treatment and IFRS 9 requirements depend on model purpose and regulatory context.

## Repository structure

```text
credit-risk-model/
├── src/credit_risk_model/
│   ├── data/                 # CSV loading and Pydantic contracts
│   ├── features/             # WoE/IV binning and PDO score scaling
│   ├── models/               # Champion, optional challenger, MLflow registry
│   ├── validation/           # Discrimination, calibration, PSI/CSI, OOT, benchmark
│   ├── monitoring/            # Drift metrics and alert rules
│   └── governance/           # Markdown/PDF report and model inventory
├── api/                      # FastAPI serving and decision audit middleware
├── pipelines/                # Train, validate, monitor entrypoints
├── reports/                  # Report output and templates
├── tests/
├── notebooks/                # Original exploration notebook location
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

The Python package lives under `src/credit_risk_model/` (the importable package name), while the top-level `src/` folder remains the library root.

## Quick start

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Windows cmd:       .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Create a small synthetic dataset and train:

```bash
python pipelines/train.py --output-dir artifacts
python pipelines/validate.py --model artifacts/champion.joblib --report reports/validation_report.md
uvicorn api.main:app --reload
```

The API exposes `GET /health`, `GET /model-card`, and `POST /predict`. API payload fields must match the configured feature list. For a real dataset, supply a CSV with a binary target column (default `target`) and numeric/categorical predictors; do not commit sensitive customer data.

## Validation scope

- **Discrimination:** AUROC, Gini, KS; optional bootstrap confidence intervals.
- **Calibration:** Brier score, log loss, Hosmer–Lemeshow grouping statistic and reliability-curve data.
- **Stability:** PSI for numeric/categorical distributions and CSI-style feature contribution reporting.
- **OOT/vintage:** temporal holdout evaluation when a date column is supplied.
- **Champion–challenger:** common split and comparable metric table; challenger is optional if XGBoost/LightGBM is installed.
- **Governance:** inventory record, risk tier, limitations, validation findings and action log.

## Model-risk and regulatory framing

The platform provides evidence artefacts that can support a model-risk process: model inventory metadata, reproducible validation metrics, assumptions/limitations, audit logs, and a generated report. It does **not** automatically certify compliance with PRA SS1/23, IFRS 9, or IRB/Standardised capital requirements. Validation should be independent of development, challenge conceptual soundness and data quality, review implementation and use, assess outcomes and limitations, document findings and remediation, and obtain accountable approval.

## Data, privacy and security

- Do not commit customer-level data, secrets, model artefacts, or production logs.
- Configure authentication, authorisation, TLS, rate limits, retention and access controls before production deployment.
- Audit events include a generated request ID, timestamp, model version and outcome metadata; avoid logging raw PII.
- Model registry calls are optional and require a configured MLflow tracking URI.

## Original notebook

Place the repository's original `Final.ipynb` at `notebooks/Final.ipynb` to preserve the exploratory work. The original notebook could not be retrieved by the available repository reader during bundle generation, so this bundle does not fabricate its contents. A short placeholder is included if no notebook is supplied.

## Tests

```bash
pytest
```

## Disclaimer

This code is a starting point for technical demonstration and should be reviewed, tested, secured, and independently validated before any business or regulated use.
