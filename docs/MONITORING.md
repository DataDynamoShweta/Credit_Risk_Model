# Monitoring

## Inputs

- `data/reference.csv` — baseline window
- `data/current.csv` — new window

## Thresholds

Set in `.env` (`PSI_INVESTIGATE`, `PSI_ALERT`, `CSI_INVESTIGATE`,
`CSI_ALERT`).

## Outputs

| File | Purpose |
|---|---|
| `artifacts/monitoring/monitoring_summary.json` | Machine-readable |
| `artifacts/monitoring/psi_csi_metrics.csv` | Per-feature metrics |
| `reports/monitoring_report.md` | Human-readable |
| `governance/audit_trail/monitoring/monitoring-*.jsonl` | Audit |

## Alert workflow

1. `overall_alert_level=escalate` → workflow opens a GitHub issue.
2. Validator records a PMA event.
3. Validator updates `next_review_date` in the inventory.
