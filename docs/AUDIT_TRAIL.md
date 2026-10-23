# Audit trail

For every `/predict` call, a JSONL line is appended to
`governance/audit_trail/decisions/decisions-YYYY-MM-DD.jsonl`.

## Fields

`request_id`, `timestamp_utc`, `model_name`, `model_version`,
`model_status`, `probability_of_default`, `score`, `decision_threshold`,
`decision`, `input_feature_names`, `latency_ms`, `environment`.

## What is not stored

- Raw feature values
- Customer identifiers
- Credentials or IPs

## Retention

`AUDIT_RETENTION_DAYS` (default 90). The demo does not auto-prune.

## Reading

- API: `GET /audit/decisions?date=2026-01-01&limit=100`
- File: `cat governance/audit_trail/decisions/decisions-2026-01-01.jsonl`
