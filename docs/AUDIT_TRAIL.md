# Audit trail

Daily-rotated, append-only JSONL files under `governance/audit_trail/`:

| Folder | Contents | Written by |
|---|---|---|
| `decisions/` | One record per successful `/predict` | API |
| `changes/` | Model-change events (see [GOVERNANCE.md](GOVERNANCE.md#model-change-log)) | Pipelines, PMA register |
| `monitoring/` | One record per monitoring run | `pipelines/monitor.py` |

## Prediction record fields

`request_id`, `timestamp_utc`, `model_name`, `model_version`, `model_status`,
`probability_of_default`, `score`, `input_feature_names`, `latency_ms`, `environment`,
`api_user_or_client_id`, `event_type`, `prev_hash`, `record_hash`.

- `request_id` is the same value as the `x-request-id` response header, the response body
  and the server log line.
- `api_user_or_client_id` comes from the `x-client-id` request header if it is a safe
  identifier (`[A-Za-z0-9._:-]{1,64}`); otherwise it is null. There is no authenticated
  identity yet.
- There is **no approve/decline decision**. A decision needs an approved credit policy and
  threshold, and none exists.

## What is not stored

- Raw feature values (only feature names)
- Customer identifiers
- Credentials or IP addresses

## Tamper evidence

Each record carries `prev_hash` (the previous record's `record_hash` in the same file; the
first record uses 64 zeros) and `record_hash` (SHA-256 of the record's canonical JSON
without `record_hash`). Editing, deleting or reordering a line breaks the chain.

```bash
python -m credit_risk_model.governance.audit verify     # exit 1 if any file fails
curl -H "Authorization: Bearer $AUDIT_API_TOKEN" http://localhost:8000/audit/verify
```

Limits: this is tamper *evidence*, not prevention. Someone with write access can rewrite a
whole file and its chain, and deleting an entire daily file shows only as a gap in dates.
Write-once storage or external anchoring of hashes would be needed for stronger guarantees.
The store assumes a single writer process.

## Write failures

`AUDIT_FAIL_CLOSED=true` (default): if the record cannot be written, `/predict` returns 503
and no prediction is returned. `AUDIT_FAIL_CLOSED=false` is a demo-only fail-open mode: the
prediction is returned and an error is printed to stderr, so the record is lost.

## Retention

```bash
python -m credit_risk_model.governance.audit prune   # deletes files older than AUDIT_RETENTION_DAYS
```

Nothing prunes automatically; schedule the command if needed.

## Reading

Audit endpoints are **disabled** unless `AUDIT_API_TOKEN` is set; then they need
`Authorization: Bearer <token>`. Page size is capped at `AUDIT_MAX_PAGE_SIZE`.

- `GET /audit/decisions?date=2026-01-01&model_version=1.3.0&request_id=...&limit=100&offset=0`
- `GET /audit/changes?model_version=1.3.0`
- `GET /audit/adjustments`
- `GET /audit/monitoring?limit=20`
- `GET /audit/verify`
