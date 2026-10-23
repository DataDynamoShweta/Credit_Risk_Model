# Monitoring

```bash
python scripts/generate_synthetic.py --rows 5000 --seed 42 --out data/reference.csv
python scripts/generate_synthetic.py --rows 5000 --seed 43 --drift 0.15 --out data/current.csv
python pipelines/monitor.py --reference data/reference.csv --current data/current.csv
echo $?   # 0 ok, 1 investigate, 2 escalate, 3 input error
```

## What is monitored

- **Feature stability (CSI):** PSI of each champion input feature, reference vs current. The
  target and date columns are never monitored as features. Continuous features use
  reference-quantile bins; binary, low-cardinality (≤ 20 levels) and categorical features
  use category proportions, with unseen categories kept.
- **Population stability:** PSI of predicted PD, reference vs current.
- **Performance** (when both samples have labels): AUROC drop, mean predicted PD and
  observed default rate.

The run stops with exit code 3 if either sample lacks a model feature; it never falls back
to a different measure.

## Levels

One vocabulary everywhere (drift summary, alerts, reports, change events): `ok`,
`investigate`, `escalate`. Thresholds come from `Settings` / `.env` only:

| Setting | Default |
|---|---|
| `PSI_INVESTIGATE` / `PSI_ESCALATE` | 0.10 / 0.25 (population PSI) |
| `CSI_INVESTIGATE` / `CSI_ESCALATE` | 0.10 / 0.25 (per feature) |
| `AUROC_DROP_INVESTIGATE` / `AUROC_DROP_ESCALATE` | 0.03 / 0.05 |

The overall level is the worst of the population, feature and performance levels. These
thresholds are conventional starting points, not approved policy.

## Outputs

| File | Purpose |
|---|---|
| `<ARTIFACTS_DIR>/monitoring/monitoring_summary.json` | Levels, population PSI, performance, recommended action |
| `<ARTIFACTS_DIR>/monitoring/psi_csi_metrics.csv` | Per-feature CSI and level |
| `<REPORTS_DIR>/monitoring_report.md` | Human-readable report |
| `<REPORTS_DIR>/drift_summary.json` | Feature drift and alert records |
| `governance/audit_trail/monitoring/*.jsonl` | One audit record per run |
| `governance/audit_trail/changes/*.jsonl` | `monitoring_alert` event when the level is not `ok` |

## Nightly workflow

`.github/workflows/monitoring.yml` trains the champion on the canonical synthetic data,
generates a reference sample and a drifted current sample with the same generator, and runs
the monitor. Exit codes are not masked: an alert fails the step, and the artefacts are
still uploaded. A GitHub issue is opened only when the overall level is `escalate`.

This workflow is a demonstration on synthetic data. Real monitoring needs production
reference and current samples.

## Alert workflow

1. `escalate` → the workflow opens a GitHub issue.
2. The validator reviews the report and records any post-model adjustment in the register.
3. The validator updates the inventory (`next_review_date`, `findings`, or `status`).
