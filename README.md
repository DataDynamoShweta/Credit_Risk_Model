# Credit Risk Model Validation Platform

> *A modular, testable toolkit for credit-risk Probability of Default (PD) modelling, validation evidence, and model governance — enhanced with agentic orchestration.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white)](https://pytest.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
![Status](https://img.shields.io/badge/Status-Demonstration-yellow?style=flat-square)

---

## 📌 Table of Contents
1. [✨ Overview](#-overview)
2. [🔄 End-to-End Workflow Pipeline](#-end-to-end-workflow-pipeline)
3. [🤖 Agentic Orchestration Layer](#-agentic-orchestration-layer)
4. [✅ Core Capabilities Implemented](#-core-capabilities-implemented)
5. [🏗️ Repository Structure](#️-repository-structure)
6. [🚀 Quick Start Guide](#-quick-start-guide)
7. [📡 API Reference](#-api-reference)
8. [📊 Validation & Monitoring](#-validation--monitoring)
9. [⚖️ Governance & Security](#️-governance--security)
10. [🧪 Testing & CI/CD](#-testing--cicd)
11. [🗺️ Roadmap & Limitations](#️-roadmap--limitations)
12. [⚠️ Disclaimer](#️-disclaimer)

---

## ✨ Overview

An end-to-end **Probability of Default (PD) modelling and validation platform**: ingest data, train champion and challenger models, generate held-out validation reports, serve scores via a gated FastAPI service, track a tamper-evident audit trail, and monitor for data/score drift.

| Component | Description |
|---|---|
| **Champion Model** | Logistic Regression on standardised and one-hot encoded features with PDO score scaling (unweighted by default to keep outputs valid as true PDs). |
| **Challenger Model** | HistGradientBoosting by default (Random Forest, XGBoost, LightGBM optional), benchmarked on identical held-out cohorts. |
| **Validation Suite** | Data review, implementation checks, discrimination (AUROC, Gini, KS), calibration (Brier, Log Loss, HL-style stats), PSI/CSI, out-of-time, and vintage testing. |
| **FastAPI Service** | Strict feature contracts, inventory-gated model loading, safe error handling, and end-to-end request tracing. |
| **Governance & Agents** | YAML model inventory, post-model adjustment registers, hash-chained change logs, and a LangGraph agentic workflow with a human approval gate. |

> **ℹ️ Note:** This is an **engineering demonstration** built on synthetic data. It is not evidence of regulatory compliance, independent model validation, or immediate production readiness.

---

## 🔄 End-to-End Workflow Pipeline

Inspired by structural multi-stage operational flows, the platform processes data through a sequential, milestone-gated lifecycle:

```text
  [QUEUED] ➔ [01 INTAKE & PROFILE] ➔ [02 TRAINING & ARTIFACTS] ➔ [03 VALIDATION & METRICS] 
                                                                           │
                                                                 (AUROC / KS Thresholds)
                                                                           ▼
                                                                [04 GOVERNANCE GATE] 
                                                                    (Interrupt)
                                                                           │
                                                         ┌─────────────────┴─────────────────┐
                                                    [APPROVED]                           [REJECTED]
                                                         ▼                                   ▼
                                                  [MODEL DEPLOYED]                     [PIPELINE HALT]
```

---

## 🤖 Agentic Orchestration Layer

A lightweight LangGraph coordinator layer sits on top of standard pipelines. **It does not compute metrics**; instead, it orchestrates transitions, packages evidence, and enforces review gates.

* **Typed Workflow State:** Managed via `CreditRiskState` (`TypedDict`) tracking run identity, stages, metrics, and governance outcomes.
* **Safe Subprocess Execution:** `PipelineRunner` invokes `train.py` and `validate.py` safely via modular scripts, capturing exit codes and artifacts.
* **LLM Commentary Integration:** Uses a lightweight local LLM (<3B parameters via Ollama, e.g., `qwen2.5:3b`) to translate pre-computed metric JSON into clear, zero-shot risk commentary.
* **Runtime Human-in-the-Loop:** Uses LangGraph `interrupt()` combined with SQLite checkpointing to pause execution at the governance stage until explicit human approval or rejection is received via API.

---

## ✅ Core Capabilities Implemented

### 📈 Modelling & Scoring
- Logistic regression champion model (`class_weight=None` default).
- PDO score scaling where higher PD maps to lower scores.
- Tree-based challenger models with strict algorithm parameters.
- Comprehensive training metadata tracking seeds, class weights, library versions, and data fingerprints.

### 🧪 Validation Framework
- Held-out test and out-of-time datasets completely separated from training rows.
- Bootstrapped confidence intervals for AUROC, Gini, and Kolmogorov-Smirnov (KS) statistics.
- Brier score, log loss, calibration-in-the-large, and population stability indexes (PSI / CSI).

### ⚙️ Governance & Monitoring
- Runtime-enforced approval gating (`/approval` endpoint resumption).
- YAML model inventory and adjustment registers.
- Nightly automated monitoring workflow checking feature CSI and score PSI with explicit error exit codes.

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
│   ├── tools/                      # Subprocess pipeline runner & Ollama client
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

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
*New agentic dependencies include:* `langgraph>=0.2`, `langgraph-checkpoint-sqlite>=1.0`, `ollama>=0.3`, `pydantic>=2.0`, and `typing-extensions>=4.0`.

### 2. Configure Local LLM (Optional for agent commentary)
```bash
ollama pull qwen2.5:3b
ollama serve
```

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
|---|---|---|
| **Scoring** | `/predict` | `POST` — Inventory-gated model scoring under strict feature contracts. |
| **Health** | `/health` | `GET` — Liveness and readiness system check. |
| **Model Card** | `/model-card` | `GET` — Returns model metadata and active feature contracts. |
| **Agents** | `/workflows/credit-risk/run` | `POST` — Initialize a new orchestrator run thread. |
| **Agents** | `/workflows/credit-risk/{run_id}` | `GET` — Inspect stage progress, metrics, and LLM prose commentary. |
| **Agents** | `/workflows/credit-risk/{run_id}/approval` | `POST` — Submit governance sign-off or rejection decision. |

---

## 📊 Validation & Monitoring

The validation routine outputs rigorous discriminatory power metrics (AUROC, Gini, KS with bootstrap confidence intervals) alongside stability assessments (PSI, CSI). The monitoring script runs nightly to catch performance degradation, triggering automated exit code alerts if stability bounds are breached.

---

## ⚖️ Governance & Security

* **Runtime Enforcement:** The approval gate is enforced programmatically by the graph runtime (`interrupt()`), guaranteeing that workflows cannot proceed past governance without a valid API sign-off payload.
* **Privacy:** Raw consumer rows are never logged or exposed to the LLM. The LLM only receives aggregated JSON metric dictionaries for qualitative summary writing.
* **Auditability:** Change logs are hash-chained, and model versions are strictly mapped against the central YAML inventory.

---

## 🧪 Testing & CI/CD

Run the test suite using `pytest`:
```bash
pytest tests/
```
Tests cover graph topology and edge branching (`test_agent_graph.py`), safe subprocess execution (`test_pipeline_runner.py`), and API workflow state handling (`test_workflow_api.py`).

---

## 🗺️ Roadmap & Limitations

- [x] Core PD modelling, validation suite, and scoring API
- [x] Drift monitoring and inventory-gated loading
- [x] LangGraph agentic orchestration layer & human-in-the-loop checkpointing
- [ ] Role-Based Access Control (RBAC) for workflow approvals
- [ ] Postgres-backed state checkpointer for multi-node deployments
- [ ] Server-Sent Events (SSE) stream endpoint for live stage visualization

---

## ⚠️ Disclaimer

*This repository is an engineering demonstration. It does not constitute a fully validated commercial credit risk system, nor does it guarantee regulatory compliance or model approval. The agentic workflow layer serves to illustrate modern orchestration patterns and does not substitute for independent model risk management controls.*