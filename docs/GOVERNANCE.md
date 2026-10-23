# Governance

## Model inventory

The register lives at `governance/model_inventory.yaml` and is validated
by `src/credit_risk_model/governance/inventory.py::ModelEntry`.

### Statuses

| Status | Can be served? | Notes |
|---|---|---|
| draft | No (unless override) | Work in progress |
| active | Yes | Requires `approved_by` and `approval_date` |
| under_review | No | Validation in progress |
| suspended | No | Incident or concern |
| retired | No | Removed from service |

### Workflow

1. Developer adds a `draft` entry pointing to a staging artifact.
2. Independent validation writes `reports/validation_report.md`.
3. Validator updates the entry to `active` with approval metadata.
4. PR requires review from the validator, `@DataDynamoShweta` (see `.github/CODEOWNERS`).
5. On merge, the API restart picks up the new entry.

### Demo override

`ALLOW_UNREVIEWED_MODEL=true` unlocks non-active statuses for local
demo only. Never set this in production.

## PMA log

`governance/pma_log.csv` records every material model change. Schema
defined in `src/credit_risk_model/governance/pma.py::COLUMNS`.
