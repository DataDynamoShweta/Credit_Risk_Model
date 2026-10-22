# Implementation brief for repository refactor

Act as a senior credit-risk model validation engineer and Python platform architect. Refactor the existing `DataDynamoShweta/Credit_Risk_Model` project into the modular structure in this bundle, while preserving all valid original analysis and business logic. Before editing, inspect every tracked file, notebook cell, dataset schema, dependency, and existing model assumption. Do not assume the README fully describes the implementation.

## Business objective

Align the project to a UK building-society Model Validation role: independent quantitative and qualitative validation, model-risk inventory and tracking, clear senior-stakeholder reporting, auditability, challenger development, lifecycle engineering, and evidence relevant to PRA SS1/23, IFRS 9, and Standardised/IRB capital frameworks where applicable.

## Required architecture

Use `src/credit_risk_model/` as the installable package with:
- `data/loader.py`, `data/schema.py`
- `features/woe_binning.py`, `features/scorecard.py`
- `models/champion.py`, `models/challenger.py`, `models/registry.py`
- `validation/discrimination.py`, `calibration.py`, `stability.py`, `oot.py`, `challenger_bench.py`
- `monitoring/drift.py`, `alerts.py`
- `governance/report_builder.py`, `inventory.py`
- `api/main.py`, `schemas.py`, `model_loader.py`, `logging_middleware.py`
- `pipelines/train.py`, `validate.py`, `monitor.py`
- `reports/`, `tests/`, `notebooks/Final.ipynb`, `.github/workflows/ci.yml`, Docker files, `requirements.txt`, `pyproject.toml`, and platform-level `README.md`.

## Critical rules

1. **Inspect first:** produce an inventory of files, current notebook sections, target/feature definitions, data leakage risks, preprocessing, split strategy, class imbalance, metric choices, limitations, and missing artefacts. Then implement.
2. **Preserve source work:** do not delete or overwrite original notebook analysis. Move/copy it to `notebooks/Final.ipynb`; retain useful plots and conclusions. Never fabricate unavailable data, findings, metrics, regulatory approval, or notebook cells.
3. **Independent validation:** distinguish model development from independent validation. Report data quality, conceptual soundness, implementation verification, discrimination, calibration, stability, OOT/vintage performance, sensitivity, limitations, and use restrictions. Where a check is not possible, mark it “Not assessed” and explain why.
4. **Metrics:** provide AUROC, Gini, KS and bootstrap confidence intervals; Brier/log loss, reliability data and grouped Hosmer–Lemeshow statistic; PSI and feature-level CSI; chronological OOT/vintage testing; a champion–challenger comparison using the same evaluation cohort. Explain that threshold conventions are policy-dependent.
5. **WoE scorecard:** include documented binning, smoothing, event definition, IV, monotonicity checks, missing/special-value handling, and leakage-safe fit/transform semantics. Never claim bins are monotonic unless verified; do not fit binning on holdout data.
6. **Scoring:** implement transparent PDO scaling and document good:bad odds convention. Add unit tests for score direction, numerical bounds, and reproducibility.
7. **Governance:** create model inventory metadata, risk tier rationale, ownership, intended use, approval state, version, validation date, limitations, findings, severity, action owner, due date, and closure evidence. Keep risk-tier rules configurable and explicitly illustrative until replaced by approved policy.
8. **Regulatory accuracy:** reference PRA SS1/23 as a model-risk management expectation, but do not claim automated SS1/23 compliance. Treat IFRS 9 expected-credit-loss modelling and IRB capital models as distinct use cases; do not conflate their targets, calibration requirements, validation evidence, or approval paths. Identify where qualified regulatory interpretation is required.
9. **API:** validate request and response contracts, use request/correlation IDs, privacy-conscious audit logs, safe error responses, health/model-card endpoints, and no raw PII logging. Add tests for invalid payloads and missing artifacts.
10. **MLOps:** support local reproducible training, separate validation and monitoring entrypoints, optional MLflow registry, optional Evidently integration, Docker, CI, dependency pinning/ranges, and a clear local quickstart.
11. **Tests:** test normal and edge cases for metrics, single-class targets, missing columns, empty datasets, constant features, categorical data, unseen categories, missing model artifacts, malformed API input, and chronological splits.
12. **Reporting:** generate a Markdown report (optional PDF) with executive summary, methodology, data/implementation review, metrics, stability/OOT results, findings, limitations, actions and sign-off placeholders. Do not automatically state “pass” or “compliant” from metric thresholds alone.
13. **Quality:** type hints, docstrings, structured logging, clear errors, no dead code, no hardcoded local paths, no credentials, no customer data, and no unsupported claims. Add `ruff` and `pytest` to CI.
14. **Deliverables:** working source tree, tests, documentation, environment/config examples, generated sample report, change summary, known limitations, and exact commands to reproduce results. Ensure `pip install -e ".[dev]"`, `pytest`, training, validation, API startup and monitoring all work.

## Acceptance criteria

- The new tree matches the requested architecture, using `src/credit_risk_model/` so imports are unambiguous.
- Original notebook and valid existing business logic are preserved.
- Unit tests and CI pass.
- Demo workflow runs without private data or external services.
- Model and validation conclusions are evidence-based and appropriately caveated.
- README explains architecture, setup, commands, validation methods, audit/governance artefacts, and production hardening.
- Final response summarizes actual changes, tests executed and passed/failed, and all items that remain unverified. Do not claim success for commands not actually run.
