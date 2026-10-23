# Governance

## Model inventory

`governance/model_inventory.yaml` is the register and the single source of model identity
(name, version, status) for the API, model card, audit events and reports. It is validated
by `src/credit_risk_model/governance/inventory.py::ModelEntry`, which **rejects unknown
fields**.

### Schema

| Field | Required | Notes |
|---|---|---|
| `model_name`, `model_version`, `model_type`, `owner`, `purpose` | yes | `model_name` + `model_version` must be unique |
| `status` | yes | `draft`, `active`, `under_review`, `suspended`, `retired` |
| `artifact_path` | yes | Relative to `PROJECT_ROOT`; produced by `pipelines/train.py` |
| `approved_by`, `approval_date` | if `active` | An `active` entry without both is rejected |
| `next_review_date` | no | After this date an `active` model is not served |
| `last_validation_date` | no | |
| `risk_tier`, `risk_tier_rationale` | no | `low` / `medium` / `high`; a tier needs a rationale |
| `intended_use`, `limitations`, `use_restrictions` | no | Shown on `/model-card` and in the validation report |
| `findings` | no | List of `finding_id`, `title`, `severity` (`low`…`critical`), `action_owner`, `due_date`, `status` (`open`/`closed`), `closure_evidence` (required when closed) |
| `training_data_reference`, `feature_list_reference`, `metadata_reference` | no | |
| `artifact_sha256` | no | If set, the API refuses an artefact with a different hash |
| `mlflow_model_uri`, `validation_report` | no | |

Monitoring thresholds are **not** in the inventory; the single source is `Settings` / `.env`.

### Serving gate

`can_serve()` decides whether the API may load a model:

| Status | Served? |
|---|---|
| `active`, review not overdue | Yes |
| `active`, `next_review_date` passed | No (override: yes) |
| `draft`, `under_review` | No (override: yes) |
| `suspended`, `retired` | Never |

The API also refuses an artefact whose recorded `model_version` differs from the inventory
version, or whose SHA-256 differs from `MODEL_SHA256` / `artifact_sha256`.

`ALLOW_UNREVIEWED_MODEL=true` is the override. It is for local demos only. CI and CD do not
set it, so their smoke tests fail if the inventory blocks the model.

`tests/test_inventory.py::test_shipped_inventory_serves_champion_today` loads the shipped
inventory with today's date. It will start failing when the champion's `next_review_date`
passes. That failure is intended: it means the review is overdue.

### Current state of the shipped inventory

The champion entry is `active` with `approved_by` and `approval_date` set. The model is a
synthetic-data demonstration and its limitations and use restrictions say so. The
approval fields were left as they are on the repository owner's instruction; they are not
evidence of an independent validation (see the gap analysis, item G2).

### Workflow

1. A developer adds a `draft` entry pointing to a staging artefact.
2. Validation runs and writes `reports/validation_report.md`.
3. The validator reviews the report, completes the sign-off section and updates the entry
   (status, approval fields, findings).
4. The change is merged through a pull request that needs code-owner review (below).
5. The API picks up the new entry without a restart: the inventory is re-read when its file
   changes. The training and validation pipelines record an `inventory_changed` event in the
   change log whenever the file's hash differs from the last recorded one.

### Code owners and branch protection

`.github/CODEOWNERS`: governance files are owned by `@DwaipayanDutta`, validation code and
reports by `@DataDynamoShweta` (validator), and the inventory by both.

CODEOWNERS only **requests** reviews. To enforce them, enable branch protection on `main`
with "Require a pull request before merging" and "Require review from Code Owners". This
cannot be checked from the repository.

## Post-model adjustment (PMA) register

`governance/pma_log.csv` is a register of **post-model adjustments** (overlays on model
output). Schema and controls: `src/credit_risk_model/governance/pma.py`.

| Field | Notes |
|---|---|
| `adjustment_type` | `overlay`, `scalar`, `floor`, `cap`, `segment_override` |
| `direction`, `magnitude`, `unit` | `increase`/`decrease`; magnitude ≥ 0; `pd_absolute`, `pd_relative_pct`, `score_points` |
| `rationale`, `evidence_reference` | required |
| `effective_date`, `review_date`, `expiry_date` | must satisfy effective ≤ review ≤ expiry |
| `owner`, `approver`, `approval_state` | `proposed`, `approved`, `rejected`, `withdrawn`; `approved` needs an approver who is not the owner |

`PMARegister.active()`, `overdue_reviews()` and `expired()` report status as of a date. The
validation report summarises the register, and `GET /audit/adjustments` returns it. Adding
an adjustment writes a `post_model_adjustment_added` change event. The shipped register is
empty: no adjustments exist.

The register is a plain CSV. It must not be stored in Git LFS (`.gitattributes` tracks only
`loan_data_2015.csv`); the loader rejects an LFS pointer instead of appending to it.

## Model-change log

Separate from the PMA register. Every material change writes a hash-chained event to
`governance/audit_trail/changes/` (see [AUDIT_TRAIL.md](AUDIT_TRAIL.md)):

| `change_type` | Written by |
|---|---|
| `model_trained` | `pipelines/train.py` |
| `validation_run` | `pipelines/validate.py` |
| `inventory_changed` | train, validate (when the inventory hash changed) |
| `monitoring_alert` | `pipelines/monitor.py` (level `investigate` or `escalate`) |
| `post_model_adjustment_added` | `PMARegister.append` |

`GET /audit/changes` returns these events.
