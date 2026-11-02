<div align="center">
  <img src="767c34801b26a840dbb16b6156837670.jpg" alt="Credit Risk Model Banner" width="100%" style="border-radius: 8px; margin-bottom: 20px;" />

  # ⚡ Credit Risk Model Validation Platform
  
  *A modular, testable toolkit for credit-risk Probability of Default (PD) modelling, validation evidence, model governance, and agentic orchestration.*

  [![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
  [![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
  [![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white)](https://mlflow.org/)
  [![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
  [![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
  [![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white)](https://pytest.org/)
</div>

---

## 🧭 Project Roadmap & Pipeline Timeline

```
       [ 🚀 CREDIT RISK PLATFORM WORKFLOW ]
                       │
                       ▼
               ┌───────────────┐
               │    START      │
               └───────┬───────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│  01. INTAKE     │◄───────►│  02. TRAINING   │
│  & PROFILING    │         │  & SCALING      │
└────────┬────────┘         └────────┬────────┘
         │                           │
         ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│  03. VALIDATION │◄───────►│  04. GOVERNANCE │
│  & METRICS      │         │  & APPROVAL     │
└────────┬────────┘         └────────┬────────┘
         │                           │
         └─────────────┬─────────────┘
                       │
                       ▼
               ┌───────────────┐
               │  05. DEPLOY   │
               │  & MONITORING │
               └───────────────┘
```

---

### 01 📊 DATA INTAKE & PROFILING
> *Validating input datasets, schema integrity, and security fingerprints.*

* **Dataset Verification:** Automatically checks existence, headers, and target column properties.
* **Fingerprinting:** Computes secure SHA-256 hashes for complete data provenance.
* **Cohort Separation:** Strictly isolates held-out test and out-of-time datasets from training runs.
* **State Initialization:** Sets up typed dictionaries (`CreditRiskState`) for robust downstream tracking.

---

### 02 ⚙️ MODEL TRAINING & SCALING
> *Constructing robust Champion and Challenger baseline algorithms.*

* **Champion Model:** Standardised Logistic Regression using one-hot encoded features with PDO score scaling (higher PD maps to lower scores, unweighted by default).
* **Challenger Model:** Tree-based algorithms (HistGradientBoosting default; optional Random Forest, XGBoost, or LightGBM).
* **Metadata Logging:** Tracks random seeds, class weights, feature lists, and library dependency versions.

---

### 03 📈 VALIDATION & METRICS
> *Rigorous testing, performance scoring, and commentary generation.*

* **Discriminatory Power:** Computes AUROC, Gini, and KS coefficients with bootstrap confidence intervals.
* **Calibration Analysis:** Measures Brier score, log loss, calibration-in-the-large, and Hosmer-Lemeshow style statistics.
* **Stability Checks:** Evaluates Population Stability Index (PSI), Characteristic Stability Index (CSI), and quarterly vintage charts.
* **LLM Integration:** Translates metric dictionaries into structured prose commentary via a local model (< 3B).

---

### 04 ⚖️ GOVERNANCE & APPROVAL GATE
> *Human-in-the-loop control and conditional decision gates.*

* **Threshold Routing:** Automatically halts execution if performance metrics fall below strict floors ($\text{AUROC} < 0.70$ or $\text{KS} < 0.30$).
* **Durable Checkpoints:** Utilizes SQLite checkpointing via LangGraph to freeze state during review.
* **Human Sign-Off:** Enforces that agents may only recommend, while authorized human analysts must approve production promotion via API endpoints.

---

### 05 🚀 SCORING API & DEPLOYMENT
> *Inventory-gated serving, hash-chained logs, and continuous monitoring.*

* **FastAPI Service:** Strict feature contract enforcement and dynamic inventory model loading.
* **Audit Trails:** Tamper-evident, hash-chained change logs and explicit YAML model inventory records.
* **Automated Monitoring:** Nightly check scripts assessing feature CSI and score PSI with error exit codes.

---

## 🏗️ Repository Structure

```text
Credit_Risk_Model/
│
├── src/credit_risk_model/          # Core model logic (data generator, features, MLflow)
│   ├── data/                       # Synthetic generator, CSV loader, LendingClub adapter
│   ├── features/                   # PDO score scaling, WoE / IV calculators
│   └── models/                     # Champion, challenger, feature contract, registry adapters
│
├── pipelines/
│   ├── train.py                    # Training script
│   ├── validate.py                 # Validation suite runner
│   └── monitor.py                  # Drift and performance monitor
│
├── credit_risk_agents/             # Agentic orchestration layer
│   ├── state.py                    # CreditRiskState TypedDict definition
│   ├── graph.py                    # StateGraph topology and conditional edges
│   ├── nodes/                      # Intake, training, validation, governance nodes
│   ├── tools/                      # Subprocess pipeline runner & LLM client (Azure OpenAI / local Hugging Face)
│   └── checkpoints.sqlite          # Durable local SQLite checkpointer
│
├── api/
│   ├── main.py                     # FastAPI application entrypoint
│   └── routers/
│       └── workflows.py            # Workflow orchestration endpoints (/workflows/credit-risk/*)
│
├── artifacts/                      # Generated model binaries and manifests
├── reports/                        # Validation reports and metric JSON outputs
├── governance/                     # Model inventory and hash-chained audit trails
├── docs/                           # Supplementary documentation
├── tests/                          # pytest test suite
├── requirements.txt                # Unified project dependencies
├── Dockerfile                      # Container build configuration
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Setup (Windows, one double-click)
Double-click `setup_and_run.bat`. It creates `.venv`, installs `requirements.txt`, downloads
`Qwen/Qwen2.5-3B-Instruct` from Hugging Face into `models/` (~6 GB), and copies
`.env.example` to `.env`, trains and validates the model, then starts the API and opens the frontend
(`/planner`) and Swagger docs in your browser. It is safe to re-run.

### 2. Local LLM
Without the downloaded model, agent commentary falls back to rule-based text.

### 3. Run Pipelines Manually
```bash
python -m pipelines.train --dataset data/synthetic.csv --target default --seed 42
python -m pipelines.validate --dataset data/synthetic.csv --target default --seed 42
```

### 4. Run the Agentic Workflow via API
Start the FastAPI server:
```bash
uvicorn api.main:app --reload
```

Trigger an orchestrated run:
```bash
curl -X POST http://localhost:8000/workflows/credit-risk/run \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_path": "data/synthetic.csv",
    "target_column": "default",
    "random_seed": 42
  }'
```

Check run status:
```bash
curl http://localhost:8000/workflows/credit-risk/crw-<run_id>
```

Submit human approval to advance the workflow:
```bash
curl -X POST http://localhost:8000/workflows/credit-risk/crw-<run_id>/approval \
  -H "Content-Type: application/json" \
  -d '{
    "decision": "APPROVE",
    "comment": "Validation metrics and evidence reviewed successfully.",
    "reviewer_id": "risk_analyst@example.com"
  }'
```

---

## 📡 API Reference

| Domain | Endpoint | Method & Description |
| :--- | :--- | :--- |
| **Scoring** | `/predict` | `POST` — Inventory-gated model scoring under strict feature contracts. |
| **Health** | `/health` | `GET` — Liveness and readiness system check. |
| **Model Card** | `/model-card` | `GET` — Returns model metadata and active feature contracts. |
| **Agents** | `/workflows/credit-risk/run` | `POST` — Initialize a new orchestrator run thread. |
| **Agents** | `/workflows/credit-risk/{run_id}` | `GET` — Inspect stage progress, metrics, and LLM prose commentary. |
| **Agents** | `/workflows/credit-risk/{run_id}/approval` | `POST` — Submit governance sign-off or rejection decision. |

---

## 🧪 Testing & CI/CD

Run the test suite using `pytest`:
```bash
pytest tests/
```
Tests cover graph topology and edge branching (`test_agent_graph.py`), safe subprocess execution (`test_pipeline_runner.py`), and API workflow state handling (`test_workflow_api.py`).

---

## ⚠️ Disclaimer
*This repository is an engineering demonstration. It does not constitute a fully validated commercial credit risk system, nor does it guarantee regulatory compliance or model approval. The agentic workflow layer serves to illustrate modern orchestration patterns and does not substitute for independent model risk management controls.*