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
| 🚀 **FastAPI Service** | Real-time scoring with request-level audit logging |
| 📑 **Report Generator** | Auto-generated validation reports with inventory metadata and findings |
| 📉 **Monitoring** | PSI / CSI drift metrics with configurable alert thresholds |

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
- ✅ Request-ID audit logging
- ✅ Dockerised deployment

</td>
<td width="50%">

### 📊 Governance
- ✅ Auto-generated validation reports
- ✅ Model inventory (code-level)
- ✅ Audit-trail middleware
- 🚧 Persistent inventory YAML *(roadmap)*
- 🚧 PMA log CSV *(roadmap)*

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
│   ├── 📂 monitoring/                 # Drift metrics and alert rules
│   └── 📂 governance/                 # Report builder and inventory logic
│
├── 🔌 api/                            # FastAPI serving + audit middleware
├── ⚙️ pipelines/                      # train.py, validate.py, monitor.py
├── 📄 reports/                        # Generated reports + Jinja2 templates
│   ├── 📂 templates/
│   └── 📝 validation_report.md
├── 🧪 tests/                          # pytest suite
├── 📓 notebooks/Final.ipynb           # Original exploration notebook
├── 🔁 .github/workflows/ci.yml        # Lint + test on Python 3.10–3.12
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
```

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

### 5️⃣ (Optional) Run with Docker

```bash
docker compose up --build
```

---

## 🔌 API Reference

| 🛣️ Endpoint | 🔧 Method | 📋 Purpose |
|---|---|---|
| `/health` | `GET` | ❤️ Liveness check and model-loaded flag |
| `/model-card` | `GET` | 📇 Model name, version, purpose and status |
| `/predict` | `POST` | 🎯 Score a single application (writes an audit event) |

### 📥 Example request

```http
POST /predict
Content-Type: application/json

{
  "features": {
    "loan_amnt": 15000,
    "int_rate": 11.5,
    "annual_inc": 62000,
    "dti": 18.2,
    "emp_length": 5
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

> ⚠️ The keys in `features` must match the feature list the model was trained on. If `request_id` is omitted, a UUID is generated.

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
| Validation Report | `reports/validation_report.md` (generated from `reports/templates/`) | ✅ Implemented |
| Model Inventory | `src/credit_risk_model/governance/inventory.py` | ⚠️ Code-level metadata only — no persistent YAML yet |
| Audit Logging | `api/logging_middleware.py` | ⚠️ Request-level events — no persistent store or query endpoints yet |
| PMA Log | — | ❌ Not yet implemented |

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

**CI via GitHub Actions** runs on every push and pull request, on Python 3.10, 3.11 and 3.12:

- 🧹 **Lint** — `ruff check src api pipelines tests`
- 🧪 **Test** — `pytest`

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

---

## 🧪 Testing

```bash
pytest
```

Test coverage includes:

- ✅ API contract tests (Pydantic schemas)
- ✅ Discrimination metric correctness
- ✅ Stability metric correctness
- ✅ End-to-end validation pipeline

---

## 🔐 Data, Privacy & Security

> 🚨 **Critical reminders before any deployment:**

- ❌ Do not commit customer-level data, secrets, model artefacts, or production logs.
- 🔒 Configure authentication, authorisation, TLS, rate limits, retention, and access controls.
- 🕵️ Audit events include a generated request ID, timestamp, model version, and outcome metadata — raw PII is never logged.
- 🔗 MLflow registry calls are optional and require a configured tracking URI.

---

## 🗺️ Roadmap

### 🎯 Governance Hardening

- [ ] 📋 Add `governance/model_inventory.yaml` — persistent, auditable inventory register
- [ ] 📝 Add `governance/pma_log.csv` + `src/credit_risk_model/governance/pma.py`
- [ ] 🗂️ Add `governance/audit_trail/` with daily-rotated decision logs
- [ ] 🔍 Add read-only endpoints: `/audit/decisions`, `/audit/changes`, `/audit/monitoring`
- [ ] 🚪 Gate model loading against inventory status and review dates

### 🚀 Platform Enhancements

- [ ] 📊 Evidently AI dashboard integration
- [ ] 🌲 SHAP-based challenger explainability
- [ ] 🔁 Automated nightly monitoring via GitHub Actions cron
- [ ] 🧾 PDF export for validation reports

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
