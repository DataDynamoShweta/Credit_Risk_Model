<div align="center">

# 🏦 Credit Risk Model Validation Platform

### *A modular, testable toolkit for credit-risk scorecard development, independent validation, and model governance.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://pytest.org/)

[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?style=flat-square&logo=github-actions&logoColor=white)](.github/workflows/ci.yml)
![Status](https://img.shields.io/badge/Status-Active_Development-yellow?style=flat-square)

</div>

---

## 📖 Table of Contents

- [✨ Overview](#-overview)
- [🎯 Key Features](#-key-features)
- [🏗️ Repository Structure](#️-repository-structure)
- [🚀 Quick Start](#-quick-start)
- [🔌 API Reference](#-api-reference)
- [📊 Validation Scope](#-validation-scope)
- [🛡️ Governance Artefacts](#️-governance-artefacts)
- [🔄 MLOps Pipeline](#-mlops-pipeline)
- [📈 Monitoring & Stability](#-monitoring--stability)
- [🧪 Testing](#-testing)
- [🔐 Data, Privacy & Security](#-data-privacy--security)
- [🗺️ Roadmap](#️-roadmap)
- [📓 Original Notebook](#-original-notebook)
- [⚠️ Disclaimer](#️-disclaimer)

---

## ✨ Overview

> A complete, end-to-end **credit-risk modelling and validation platform** that goes beyond a notebook — from feature engineering to a live scoring API, with automated validation reports and governance hooks.

This repository demonstrates how a traditional credit-risk scorecard can be productionised responsibly:

| 🧩 Component | 🎯 What it does |
|---|---|
| 🧮 **Champion Model** | Logistic-regression scorecard with WoE binning and PDO scaling |
| 🌲 **Challenger Model** | Optional XGBoost / LightGBM benchmark for champion–challenger testing |
| 🔬 **Validation Suite** | Discrimination, calibration, stability, OOT / vintage, and benchmark testing |
| 🚀 **FastAPI Service** | Real-time scoring, gated by the model inventory, with a persistent audit trail |
| 📑 **Report Generator** | Auto-generated validation reports with inventory metadata and findings |
| 📉 **Monitoring** | PSI / CSI drift metrics, alert levels and a nightly GitHub Actions run |
| 🛡️ **Governance** | YAML model inventory, PMA log, audit trail and code-owner review |

> **ℹ️ Note:** This is an **engineering demonstration**. It is not evidence of regulatory approval or compliance. Model-risk processes require documented policies, independent review, and accountable sign-off by the deploying institution.

---

## 🎯 Key Features

<table>
<tr>
<td width="50%">

### 🧠 Modelling
- ✅ Logistic-regression champion
- ✅ WoE / IV feature binning
- ✅ PDO scorecard scaling
- ✅ Optional tree-based challenger
- ✅ MLflow model registry integration
- ✅ Optional SHAP explainability

</td>
<td width="50%">

### 🔬 Validation
- ✅ AUROC, Gini, KS
- ✅ Bootstrap confidence intervals
- ✅ Brier score & log loss
- ✅ Hosmer–Lemeshow calibration
- ✅ Out-of-time (OOT) testing

</td>
</tr>
<tr>
<td width="50%">

### 🚀 Serving
- ✅ FastAPI REST endpoints
- ✅ Pydantic request contracts
- ✅ `/health`, `/model-card`, `/predict`
- ✅ Read-only `/audit/*` endpoints
- ✅ Inventory-gated model loading
- ✅ Dockerised deployment, published to GHCR

</td>
<td width="50%">

### 📊 Governance
- ✅ Auto-generated validation reports (optional PDF)
- ✅ Persistent model inventory YAML
- ✅ PMA log CSV
- ✅ Daily-rotated JSONL audit trail
- ✅ CODEOWNERS review for governance changes

</td>
</tr>
</table>

---

## 🏗️ Repository Structure

```text
credit-risk-model/
│
├── 📦 src/credit_risk_model/          # Importable Python package
│   ├── 📂 data/                       # CSV loading and Pydantic contracts
│   ├── 📂 features/                   # WoE / IV binning and PDO score scaling
│   ├── 📂 models/                     # Champion, challenger, MLflow registry
│   ├── 📂 validation/                 # Discrimination, calibration, PSI/CSI, OOT
│   ├── 📂 monitoring/                 # Drift metrics, alert rules, monitoring report
│   ├── 📂 governance/                 # Inventory, PMA log, audit store, reports, PDF export
│   ├── 📂 explainability/             # Optional SHAP helpers
│   └── ⚙️ config.py                   # Settings loaded from environment / .env
│
├── 🔌 api/                            # FastAPI serving + audit middleware
│   └── 📂 routers/                    # Read-only /audit endpoints
├── ⚙️ pipelines/                      # train.py, validate.py, monitor.py
├── 🛡️ governance/                     # Governance records
│   ├── 📋 model_inventory.yaml        # Model register (status, approval, review dates)
│   ├── 📝 pma_log.csv                 # Model-change event log
│   └── 📂 audit_trail/                # decisions/, changes/, monitoring/ (JSONL)
├── 📚 docs/                           # Governance, audit trail, monitoring, CI/CD notes
├── 🧰 scripts/generate_synthetic.py   # Synthetic data for demos and CI
├── 📄 reports/                        # Generated reports + Jinja2 templates
│   ├── 📂 templates/
│   └── 📝 validation_report.md
├── 🧪 tests/                          # pytest suite
├── 📓 notebooks/Final.ipynb           # Original exploration notebook
├── 🔁 .github/
│   ├── 📂 workflows/                  # ci.yml, cd.yml, monitoring.yml
│   └── 👥 CODEOWNERS                  # Required reviewers
├── 💾 loan_data_2015.csv              # LendingClub 2015 loan data (Git LFS)
├── 🪟 setup_and_run.bat               # Windows one-click setup, train, validate, serve
├── 🔧 .env.example                    # Configuration template
├── 🐳 Dockerfile
├── 🐳 docker-compose.yml
├── ⚙️ pyproject.toml
└── 📋 requirements.txt
```

---

## 🚀 Quick Start

> **Prerequisite:** Python 3.10+

### 1️⃣ Clone and set up the environment

```bash
git clone https://github.com/DataDynamoShweta/Credit_Risk_Model.git
cd Credit_Risk_Model

python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Windows cmd:        .venv\Scripts\activate.bat
# macOS / Linux:      source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[dev]"
cp .env.example .env    # optional: override paths, thresholds and flags
```

> 💡 On Windows, `setup_and_run.bat` runs the setup, training, validation and API steps in one go.

### 2️⃣ Train a champion model

```bash
python pipelines/train.py --output-dir artifacts
```

> 💡 Uses synthetic data if no CSV is supplied.

### 3️⃣ Run the validation suite

```bash
python pipelines/validate.py \
  --model artifacts/champion.joblib \
  --report reports/validation_report.md
```

### 4️⃣ Start the scoring API

```bash
uvicorn api.main:app --reload
```

Then open 👉 <http://localhost:8000/docs> for the interactive Swagger UI.

> 🚪 The API only serves the champion if `governance/model_inventory.yaml` allows it: the entry must be `active` and its `next_review_date` must not have passed. Otherwise `/health` reports `"model_loaded": false` and `/predict` returns `503`. For local demos only, `ALLOW_UNREVIEWED_MODEL=true` bypasses the check for `draft` and `under_review` models and overdue reviews. Suspended and retired models are never served.

### 5️⃣ Run drift monitoring

```bash
python scripts/generate_synthetic.py --rows 5000 --seed 42 --out data/reference.csv
python scripts/generate_synthetic.py --rows 5000 --seed 43 --drift 0.15 --out data/current.csv
python pipelines/monitor.py --reference data/reference.csv --current data/current.csv
```

### 6️⃣ (Optional) Run with Docker

```bash
docker compose up --build
```

---

## 🔌 API Reference

| 🛣️ Endpoint | 🔧 Method | 📋 Purpose |
|---|---|---|
| `/health` | `GET` | ❤️ Liveness check, model-loaded flag and environment |
| `/model-card` | `GET` | 📇 Model name, version and status, plus its inventory entry |
| `/predict` | `POST` | 🎯 Score a single application (writes an audit event) |
| `/audit/decisions` | `GET` | 🧾 Prediction audit events; filter by `date`, `model_version`, `request_id` |
| `/audit/changes` | `GET` | 📝 PMA log entries; filter by `model_version` |
| `/audit/monitoring` | `GET` | 📉 Recent monitoring-run events |

### 📥 Example request

```http
POST /predict
Content-Type: application/json

{
  "features": {
    "income": 52000,
    "debt_ratio": 0.32,
    "age": 41,
    "employment_type": "salaried"
  },
  "request_id": "optional-client-id"
}
```

### 📤 Example response

```json
{
  "request_id": "optional-client-id",
  "probability_of_default": 0.083,
  "score": 642,
  "model_version": "1.3.0"
}
```

> ⚠️ The keys in `features` must match the feature list the model was trained on (the example uses the synthetic demo features). If `request_id` is omitted, a UUID is generated. The `/audit` list endpoints accept `limit` and `offset`.

---

## 📊 Validation Scope

<table>
<tr>
<td width="50%">

### 🎯 Discrimination
- AUROC
- Gini coefficient
- KS statistic
- Bootstrap confidence intervals *(optional)*

### 📐 Calibration
- Brier score
- Log loss
- Hosmer–Lemeshow grouping statistic
- Reliability-curve data

</td>
<td width="50%">

### 📉 Stability
- PSI for numeric features
- PSI for categorical features
- CSI-style feature contribution

### ⏳ Out-of-Time (OOT)
- Temporal holdout evaluation
- Vintage-level performance tracking
- *(Enabled when a date column is supplied)*

### 🥊 Champion–Challenger
- Common-split metric comparison
- Optional XGBoost / LightGBM challenger

</td>
</tr>
</table>

---

## 🛡️ Governance Artefacts

| 📦 Artefact | 📁 Location | 🟢 Status |
|---|---|---|
| Validation Report | `reports/validation_report.md` (generated from `reports/templates/`; PDF via `--export-pdf`) | ✅ Implemented |
| Model Inventory | `governance/model_inventory.yaml`, validated by `src/credit_risk_model/governance/inventory.py` | ✅ Implemented — gates model loading |
| Audit Trail | `governance/audit_trail/{decisions,changes,monitoring}/*.jsonl` | ✅ Implemented — daily files, read via `/audit/*` |
| PMA Log | `governance/pma_log.csv`, written by `src/credit_risk_model/governance/pma.py` | ✅ Implemented |

### 👥 Ownership and review

Ownership is set in [`.github/CODEOWNERS`](.github/CODEOWNERS):

| Area | Owner |
|---|---|
| Governance (`governance/`, governance code and docs) | [@DwaipayanDutta](https://github.com/DwaipayanDutta) |
| Independent validation (validation code, `reports/`) | [@DataDynamoShweta](https://github.com/DataDynamoShweta) |
| Model inventory changes | Both — the validator reviews and approves |

A model moves from `draft` to `active` only after validation, with `approved_by` and `approval_date` recorded in the inventory. See [docs/GOVERNANCE.md](docs/GOVERNANCE.md), [docs/AUDIT_TRAIL.md](docs/AUDIT_TRAIL.md) and [docs/MONITORING.md](docs/MONITORING.md).

---

## 🔄 MLOps Pipeline

```text
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   📥 Data    │───▶│  🧮 Train    │───▶│  🔬 Validate │───▶│  🚀 Serve    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                           │                    │                    │
                           ▼                    ▼                    ▼
                    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
                    │  📊 MLflow   │    │  📄 Report   │    │  🛡️ Audit    │
                    │  Registry    │    │  Generator   │    │  Logging     │
                    └──────────────┘    └──────────────┘    └──────────────┘
                                                                 │
                                                                 ▼
                                                        ┌──────────────┐
                                                        │  📉 Monitor  │
                                                        │  PSI / CSI   │
                                                        └──────────────┘
```

**GitHub Actions workflows** (details in [docs/CI_CD.md](docs/CI_CD.md)):

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | Every push and pull request | 🧹 `ruff check` and 🧪 `pytest` on Python 3.10, 3.11 and 3.12 |
| `cd.yml` | Push to `main`, `v*` tags | 🐳 Builds the image, smoke-tests `/health`, publishes to GHCR |
| `monitoring.yml` | Nightly 02:00 UTC, or manual | 📉 Runs drift monitoring; opens an issue on an `escalate` alert |

---

## 📈 Monitoring & Stability

### 📉 Population Stability Index (PSI)

```text
PSI = Σ (Actual% − Expected%) × ln(Actual% / Expected%)
```

| 🚦 Threshold | 🎯 Interpretation | 🛠️ Action |
|---|---|---|
| `< 0.10` | 🟢 No significant shift | None |
| `0.10 – 0.25` | 🟡 Moderate shift | Investigate |
| `≥ 0.25` | 🔴 Major shift | Escalate to validation |

### 📊 Characteristic Stability Index (CSI)

Applied per feature to isolate which inputs are driving drift — essential for root-cause analysis and stakeholder communication.

### 📤 Monitoring outputs

| File | Purpose |
|---|---|
| `artifacts/monitoring/monitoring_summary.json` | Overall and population alert levels, recommended action |
| `artifacts/monitoring/psi_csi_metrics.csv` | Per-feature PSI / CSI and level |
| `reports/monitoring_report.md` | Human-readable report |
| `reports/drift_summary.json` | Feature drift and alert records |

Thresholds come from `.env` (`PSI_INVESTIGATE`, `PSI_ALERT`, `CSI_INVESTIGATE`, `CSI_ALERT`). Population PSI is calculated on predicted PD when the data has the model's features; otherwise the largest feature PSI is used.

---

## 🧪 Testing

```bash
pytest
```

Test coverage includes:

- ✅ API endpoints, including `/audit/*`
- ✅ Discrimination, stability and calibration metrics
- ✅ WoE / IV, score scaling and OOT split
- ✅ Model inventory validation and serving gate
- ✅ Audit store, PMA log and PDF export
- ✅ Explainability fallback when SHAP is not installed

---

## 🔐 Data, Privacy & Security

> 🚨 **Critical reminders before any deployment:**

- ❌ Do not commit customer-level data, secrets, model artefacts, or production logs.
- 🔒 Configure authentication, authorisation, TLS, rate limits, retention, and access controls.
- 🕵️ Audit events include a request ID, timestamp, model version, PD, score, decision and feature *names* — raw feature values and PII are never logged.
- 🚫 Never set `ALLOW_UNREVIEWED_MODEL=true` outside local demos.
- 🔗 MLflow registry calls are optional and require a configured tracking URI.

---

## 🗺️ Roadmap

### 🎯 Governance Hardening

- [x] 📋 Add `governance/model_inventory.yaml` — persistent, auditable inventory register
- [x] 📝 Add `governance/pma_log.csv` + `src/credit_risk_model/governance/pma.py`
- [x] 🗂️ Add `governance/audit_trail/` with daily-rotated decision logs
- [x] 🔍 Add read-only endpoints: `/audit/decisions`, `/audit/changes`, `/audit/monitoring`
- [x] 🚪 Gate model loading against inventory status and review dates
- [ ] 🧹 Enforce audit-log retention (`AUDIT_RETENTION_DAYS` is not applied yet)
- [ ] 🔐 Add authentication and authorisation to the API

### 🚀 Platform Enhancements

- [ ] 📊 Evidently AI dashboard integration
- [x] 🌲 SHAP-based explainability helpers *(optional `explain` extra)*
- [x] 🔁 Automated nightly monitoring via GitHub Actions cron
- [x] 🧾 PDF export for validation reports *(optional `pdf` extra)*
- [ ] 🧮 Connect `loan_data_2015.csv` to the training pipeline

---

## 📓 Original Notebook

The original exploratory analysis is preserved at:

```text
📂 notebooks/Final.ipynb
```

Useful for understanding the modelling journey that led to the productionised pipeline.

---

## ⚠️ Disclaimer

<div align="center">

**This code is a technical demonstration.**

It should be reviewed, tested, secured, and independently validated before any business use.

*No warranty is provided. Use at your own risk.*

</div>

---

<div align="center">

💡 **Built with care for responsible credit-risk modelling**

⭐ If this project helped you, consider giving it a star! ⭐

[⬆ Back to Top](#-credit-risk-model-validation-platform)

</div>
