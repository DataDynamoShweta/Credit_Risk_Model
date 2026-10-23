<div align="center">

# 🏦 Credit Risk Model Validation Platform

### *A modular, testable toolkit for credit-risk PD modelling, validation evidence and model governance.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://pytest.org/)

[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?style=flat-square&logo=github-actions&logoColor=white)](.github/workflows/ci.yml)
![Status](https://img.shields.io/badge/Status-Demonstration-yellow?style=flat-square)

</div>

---

## 📖 Table of Contents

- [✨ Overview](#-overview)
- [🎯 What is implemented](#-what-is-implemented)
- [🏗️ Repository Structure](#️-repository-structure)
- [🚀 Quick Start](#-quick-start)
- [🔌 API Reference](#-api-reference)
- [📊 Validation](#-validation)
- [🛡️ Governance](#️-governance)
- [📈 Monitoring](#-monitoring)
- [🔄 CI/CD and Docker](#-cicd-and-docker)
- [🧪 Testing](#-testing)
- [🔐 Data, Privacy & Security](#-data-privacy--security)
- [🚧 Not assessed / known limitations](#-not-assessed--known-limitations)
- [🗺️ Roadmap](#️-roadmap)
- [📓 Original Notebook](#-original-notebook)
- [⚠️ Disclaimer](#️-disclaimer)

---

## ✨ Overview

> An end-to-end **probability-of-default (PD) modelling and validation platform**: synthetic or
> real data in, a held-out validation report, a scoring API gated by a model inventory, a
> tamper-evident audit trail, and drift/performance monitoring.

| 🧩 Component | 🎯 What it does |
|---|---|
| 🧮 **Champion model** | Logistic regression on standardised and one-hot encoded features, with PDO score scaling. Unweighted by default so outputs stay usable as PDs. |
| 🌲 **Challenger model** | HistGradientBoosting by default (Random Forest, XGBoost, LightGBM optional), compared on the same held-out cohort. |
| 🔬 **Validation suite** | Data review, implementation checks, discrimination, calibration, PSI/CSI, out-of-time, vintage, champion–challenger, sensitivity, WoE/IV review, and an explicit not-assessed list. |
| 🚀 **FastAPI service** | Strict feature contract, inventory-gated model loading, safe errors, one request ID end to end. |
| 🛡️ **Governance** | Strict YAML model inventory, post-model adjustment register, hash-chained change log, CODEOWNERS review. |
| 📉 **Monitoring** | Feature CSI, score PSI and performance drift, with exit codes and a nightly workflow. |

> **ℹ️ Note:** This is an **engineering demonstration** on synthetic data. It is not evidence
> of model approval, independent validation, regulatory compliance or production readiness.

---

## 🎯 What is implemented

<table>
<tr>
<td width="50%">

### 🧠 Modelling
- ✅ Logistic-regression champion (`class_weight=None` by default)
- ✅ PDO score scaling (higher PD → lower score)
- ✅ Tree-based challengers with explicit algorithm choice
- ✅ Training metadata: seed, class weight, data fingerprint, library versions, feature list
- ✅ MLflow registry adapter using model aliases
- ✅ Optional SHAP explanations (log-odds, not reason codes)

</td>
<td width="50%">

### 🔬 Validation
- ✅ Held-out test + out-of-time samples, never training rows
- ✅ AUROC, Gini, KS with bootstrap CI
- ✅ Brier, log loss, calibration-in-the-large, grouped HL-style statistic
- ✅ PSI / CSI, out-of-time and quarterly vintage performance
- ✅ Champion–challenger, sensitivity, WoE/IV review (univariate)
- ✅ Report from one Jinja2 template, with sign-off placeholders

</td>
</tr>
<tr>
<td width="50%">

### 🚀 Serving
- ✅ `/health`, `/model-card`, `/predict`
- ✅ Missing, unexpected and wrongly typed features rejected
- ✅ No raw input echoed in errors; safe 500 handler
- ✅ Inventory gate, version check, optional SHA-256 check
- ✅ Token-protected `/audit/*` endpoints

</td>
<td width="50%">

### 📊 Governance & operations
- ✅ Strict model inventory (unknown fields rejected)
- ✅ Post-model adjustment register with expiry and approval controls
- ✅ Hash-chained audit trail with verify and prune commands
- ✅ Docker image that trains a demo model and checks readiness
- ✅ CI with end-to-end pipeline and container smoke tests

</td>
</tr>
</table>

> The WoE/IV module is used for univariate review in the validation report; the champion
> does **not** use WoE-transformed inputs.

---

## 🏗️ Repository Structure

```text
Credit_Risk_Model/
│
├── 📦 src/credit_risk_model/
│   ├── 📂 data/            # synthetic generator, CSV loader, LendingClub adapter, sources
│   ├── 📂 features/        # PDO score scaling, WoE / IV
│   ├── 📂 models/          # champion, challenger, feature contract, MLflow alias adapter
│   ├── 📂 validation/      # discrimination, calibration, stability, OOT, vintage, suite
│   ├── 📂 monitoring/      # drift levels, alerts, monitoring report
│   ├── 📂 governance/      # inventory, PMA register, audit store, report builder, PDF
│   ├── 📂 explainability/  # optional SHAP helpers
│   └── ⚙️ config.py        # settings from environment / .env
│
├── 🔌 api/                 # FastAPI app, model loader, security, middleware, /audit router
├── ⚙️ pipelines/           # train.py, validate.py, monitor.py
├── 🛡️ governance/
│   ├── 📋 model_inventory.yaml   # model register (the source of model identity)
│   ├── 📝 pma_log.csv            # post-model adjustment register (empty)
│   └── 📂 audit_trail/           # decisions/, changes/, monitoring/ (JSONL, not committed)
├── 📄 reports/templates/   # validation_report.md.j2 (generated reports are not committed)
├── 📚 docs/                # GOVERNANCE, AUDIT_TRAIL, MONITORING, CI_CD, DATA
├── 🧰 scripts/             # generate_synthetic.py, smoke_test.py
├── 🧪 tests/               # unit, API, governance and end-to-end tests
├── 📓 notebooks/           # original Final.ipynb + README of its known flaws
├── 🔁 .github/             # ci.yml, cd.yml, monitoring.yml, CODEOWNERS
├── 💾 loan_data_2015.csv   # LendingClub 2015 data (Git LFS, 175 MB)
├── 🐳 Dockerfile, docker-compose.yml, .dockerignore
├── 🪟 setup_and_run.bat    # Windows one-click setup, train, validate, serve
├── 🔧 .env.example         # every supported setting
└── ⚙️ pyproject.toml       # single source of dependencies (requirements.txt installs it)
```

Generated at run time and git-ignored: `artifacts/`, `reports/*.md` (except templates),
`reports/drift_summary.json`, `data/`, `governance/audit_trail/**/*.jsonl`.

---

## 🚀 Quick Start

> **Prerequisite:** Python 3.10+. Run commands from the repository root (paths resolve
> against the current directory, or `PROJECT_ROOT` if set).

### 1️⃣ Set up

```bash
git clone https://github.com/DataDynamoShweta/Credit_Risk_Model.git
cd Credit_Risk_Model
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1   |   macOS / Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
cp .env.example .env    # optional
```

> 💡 On Windows, `setup_and_run.bat` runs set-up, training, validation and the API in one go.

### 2️⃣ Train

```bash
python pipelines/train.py
```

Trains the champion and a HistGradientBoosting challenger on 20,000 synthetic applications
and writes `artifacts/champion.joblib`, `challenger.joblib`, `feature_list.json`,
`training_metadata.json` and `training_metrics.json`.

### 3️⃣ Validate

```bash
python pipelines/validate.py                       # add --export-pdf reports/validation_report.pdf
```

Rebuilds the training split from the recorded metadata, scores only held-out rows and writes
`reports/validation_report.md`.

### 4️⃣ Monitor

```bash
python scripts/generate_synthetic.py --rows 5000 --seed 42 --out data/reference.csv
python scripts/generate_synthetic.py --rows 5000 --seed 43 --drift 0.15 --out data/current.csv
python pipelines/monitor.py --reference data/reference.csv --current data/current.csv
```

Exit code `0` ok, `1` investigate, `2` escalate, `3` input error.

### 5️⃣ Serve

```bash
uvicorn api.main:app --reload
python scripts/smoke_test.py --url http://localhost:8000   # in another terminal
```

Swagger UI: <http://localhost:8000/docs>.

> 🚪 The API serves the champion only if `governance/model_inventory.yaml` allows it (status
> `active` and `next_review_date` not passed) and the artefact's version matches the inventory.
> Otherwise `/health` reports `model_loaded: false` with the reason and `/predict` returns 503.
> `ALLOW_UNREVIEWED_MODEL=true` bypasses the status and review-date checks for **local demos
> only**; suspended and retired models are never served.

### 6️⃣ Real data (optional)

```bash
git lfs pull
python pipelines/train.py    --data loan_data_2015.csv --adapter lending_club
python pipelines/validate.py --data loan_data_2015.csv --adapter lending_club
```

See [docs/DATA.md](docs/DATA.md) for the target definition, leakage controls and caveats.

### 7️⃣ Docker

```bash
docker compose up --build
python scripts/smoke_test.py --url http://localhost:8000
```

---

## 🔌 API Reference

| 🛣️ Endpoint | 🔧 Method | 📋 Purpose |
|---|---|---|
| `/health` | `GET` | Liveness, `model_loaded`, reason, model name/version, environment |
| `/model-card` | `GET` | Inventory entry (status, intended use, limitations, use restrictions, risk tier), training metadata and feature list |
| `/predict` | `POST` | Score one application; writes an audit record (fail-closed) |
| `/audit/decisions` | `GET` | Prediction records; filter by `date`, `model_version`, `request_id` |
| `/audit/changes` | `GET` | Model-change events (training, validation, inventory, monitoring, PMA) |
| `/audit/adjustments` | `GET` | Post-model adjustment register with active / overdue / expired IDs |
| `/audit/monitoring` | `GET` | Recent monitoring runs |
| `/audit/verify` | `GET` | Hash-chain verification of every audit file |

`/audit/*` is disabled unless `AUDIT_API_TOKEN` is set; then it needs
`Authorization: Bearer <token>`. If `PREDICT_API_TOKEN` is set, `/predict` needs it too.

### 📥 Example request

```http
POST /predict
Content-Type: application/json
x-request-id: demo-001

{
  "features": {
    "loan_amnt": 15000.0,
    "int_rate": 11.5,
    "annual_inc": 62000.0,
    "dti": 18.2,
    "emp_length": 5,
    "home_ownership": "RENT"
  }
}
```

### 📤 Example response

```json
{
  "request_id": "demo-001",
  "probability_of_default": 0.058,
  "score": 568,
  "model_name": "credit_risk_champion",
  "model_version": "1.3.0"
}
```

All six features are required, numbers must be numbers and `home_ownership` a string;
anything else returns 422 listing feature names only (submitted values are never echoed).
The `x-request-id` header (or `request_id` in the body) is used for the response, the
`x-request-id` response header, the audit record and the log line. No approve/decline
decision is returned or recorded: there is no approved decision policy.

---

## 📊 Validation

`pipelines/validate.py` → `src/credit_risk_model/validation/suite.py` →
`reports/templates/validation_report.md.j2`.

| Area | How |
|---|---|
| Split | Latest 20% of applications by date = out-of-time; the rest split 75/25 stratified into train/test. Shared with training. |
| Data review | Rows, defaults and default rate per sample, missingness, distinct values, constant features, duplicates |
| Implementation | Feature contract, PD bounds, determinism, score direction, class weighting off, data fingerprint matches training |
| Discrimination | AUROC, Gini, KS, bootstrap 95% CI (test sample) |
| Calibration | Brier, log loss, mean PD vs observed rate, reliability table, grouped HL-style statistic (no p-value) |
| Stability | Score PSI and feature CSI, train→test and train→OOT |
| Out-of-time / vintage | OOT metrics; quarterly cohorts of test + OOT applications |
| Champion–challenger | Same held-out test cohort |
| Sensitivity | Mean PD change for ±10% on each numeric feature |
| WoE / IV | Univariate review on the training sample |
| Findings | Rule-based automated findings, labelled as such, for a validator to confirm or reject |
| Not assessed | Conceptual soundness, data lineage, ECL/IRB calibration, use test, real-data validation (when synthetic) |

Reference run (synthetic, 20,000 rows): held-out AUROC ≈ 0.71, OOT AUROC ≈ 0.72,
calibration ratio ≈ 1.01. Discrimination and calibration are reported separately.

---

## 🛡️ Governance

| 📦 Artefact | 📁 Location | Status |
|---|---|---|
| Model inventory | `governance/model_inventory.yaml` (schema in `governance/inventory.py`) | ✅ Strict schema; gates serving |
| Post-model adjustment register | `governance/pma_log.csv` (`governance/pma.py`) | ✅ Implemented; no adjustments recorded |
| Model-change log | `governance/audit_trail/changes/` | ✅ Written by train, validate, monitor and PMA changes |
| Prediction audit trail | `governance/audit_trail/decisions/` | ✅ Hash-chained, fail-closed |
| Validation report | `reports/validation_report.md` (generated) | ✅ Draft evidence pack; needs validator sign-off |

### 👥 Ownership and review

| Area | Owner |
|---|---|
| Governance (`governance/`, governance code and docs) | [@DwaipayanDutta](https://github.com/DwaipayanDutta) |
| Validation (validation code, `reports/`) | [@DataDynamoShweta](https://github.com/DataDynamoShweta) |
| Model inventory changes | Both |

CODEOWNERS only takes effect if `main` has branch protection requiring code-owner review.

> The champion's inventory entry is `active` with approval fields filled in. The model has
> only been evaluated on synthetic data and its limitations and use restrictions say so;
> those approval fields are not evidence of independent validation. See
> [docs/GOVERNANCE.md](docs/GOVERNANCE.md).

---

## 📈 Monitoring

| 🚦 PSI / CSI | Level | 🛠️ Action |
|---|---|---|
| `< 0.10` | 🟢 `ok` | None |
| `0.10 – 0.25` | 🟡 `investigate` | Investigate |
| `≥ 0.25` | 🔴 `escalate` | Escalate to validation |

Only model input features are monitored (never the target). Population PSI is on predicted
PD. With labels, AUROC drop above 0.03 / 0.05 raises `investigate` / `escalate`. Thresholds
come from `.env` only. See [docs/MONITORING.md](docs/MONITORING.md).

---

## 🔄 CI/CD and Docker

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | Every push and PR | Ruff and pytest on 3.10–3.12, CLI pipeline smoke test, Docker build + container smoke test |
| `cd.yml` | `main`, `v*` tags | Build, smoke test (model must load and score), publish to GHCR |
| `monitoring.yml` | Nightly 02:00 UTC | Train, monitor drifted synthetic data, issue on `escalate` |

The Docker image runs as a non-root user, trains the demo model during the build, and has
a `HEALTHCHECK` that requires `model_loaded: true`. `docker-compose.yml` keeps the audit
trail in a named volume. Smoke tests never set `ALLOW_UNREVIEWED_MODEL`. Details:
[docs/CI_CD.md](docs/CI_CD.md).

---

## 🧪 Testing

```bash
pytest
ruff check src api pipelines tests scripts
python -m credit_risk_model.governance.audit verify
```

The suite covers every defect fixed from the gap analysis, including calibration with the
default training settings, WoE bin order, OOT with the project's own model, binary-feature
PSI, challenger algorithm choice, strict payloads, no negative model caching, the shipped
inventory with today's date, audit tampering, and an end-to-end train → validate → monitor
→ API run from a separate working directory.

---

## 🔐 Data, Privacy & Security

- ❌ Do not commit customer-level data, secrets, model artefacts or audit logs (all git-ignored).
- 🕵️ Audit records hold the request ID, model version, PD, score and feature *names*; raw
  feature values and identifiers are never stored or logged.
- 🔒 Demo safeguards only: optional bearer tokens, request-size limit, sanitised request IDs.
  Production needs an identity provider, role-based authorisation, TLS and rate limiting at
  a gateway.
- 📦 Model artefacts are pickles: load only artefacts you built, and set `MODEL_SHA256`.
- 🚫 Never set `ALLOW_UNREVIEWED_MODEL=true` outside local demos.

---

## 🚧 Not assessed / known limitations

- Conceptual soundness, data lineage, use testing and ECL/IRB-specific calibration.
- Real-data results come from resolved 2015 LendingClub loans only, which is a biased sample
  ([docs/DATA.md](docs/DATA.md)).
- Docker builds and GitHub Actions runs have not been executed from this environment.
- MLflow registry integration is tested with a mocked client only; WeasyPrint PDF output is
  tested with mocks only.
- Audit tamper evidence does not stop a party with write access from rewriting a whole file.

---

## 🗺️ Roadmap

- [x] Persistent model inventory, PMA register, audit trail, `/audit/*` endpoints
- [x] Inventory-gated serving, nightly monitoring, PDF export, SHAP helpers
- [x] Real-data adapter for `loan_data_2015.csv`
- [ ] Independent validation of the champion and genuine sign-off
- [ ] Authentication / authorisation through an identity provider
- [ ] Scheduled audit-log retention and external anchoring of audit hashes
- [ ] Evidently dashboard integration
- [ ] Policy-approved decision thresholds (none are applied today)

---

## 📓 Original Notebook

`notebooks/Final.ipynb` is the original analysis, kept unchanged.
[notebooks/README.md](notebooks/README.md) lists its flaws (preprocessing leakage, immature
loans counted as good, post-origination leakage, reversed score direction, no stratified
split) and how the package addresses each.

---

## ⚠️ Disclaimer

<div align="center">

**This code is a technical demonstration.**

It should be reviewed, tested, secured, and independently validated before any business use.

*No warranty is provided. Use at your own risk.*

</div>

---

<div align="center">

[⬆ Back to Top](#-credit-risk-model-validation-platform)

</div>
