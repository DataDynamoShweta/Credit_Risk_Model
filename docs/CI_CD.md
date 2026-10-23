# CI/CD

## Workflows

| Workflow | File | Trigger | What it does |
|---|---|---|---|
| CI | `.github/workflows/ci.yml` | Every push and pull request | `ruff`, `pytest` (incl. the end-to-end pipeline test) on Python 3.10–3.12; CLI pipeline smoke test (train → validate → monitor → audit verify); Docker build and container smoke test |
| cd | `.github/workflows/cd.yml` | Push to `main`, `v*` tags, manual | Builds the image, smoke-tests it, pushes to GHCR |
| nightly-monitoring | `.github/workflows/monitoring.yml` | Daily 02:00 UTC, manual | See [MONITORING.md](MONITORING.md) |

## Container smoke test

`scripts/smoke_test.py` fails unless `/health` reports `model_loaded: true` and `/predict`
returns a probability. The image trains the demo model during the build, so the container
can serve without a mounted artefact. `ALLOW_UNREVIEWED_MODEL` is **not** set in CI or CD,
so a model blocked by the inventory (for example, an overdue review) fails the build.

## Git LFS

Only `loan_data_2015.csv` is stored in LFS. CI does not read it, so checkouts do not fetch
LFS objects. Governance files are plain text.

## Secrets

| Name | Purpose | Notes |
|---|---|---|
| `GITHUB_TOKEN` | GHCR push, issue creation | Provided automatically; needs `packages: write` / `issues: write` |

## Image tags

- `<git-sha-12>` on every build
- `vX.Y.Z` on semver tags
- `latest` on `main` only

## Branch protection

For CODEOWNERS to be enforced, `main` needs branch protection with required pull-request
reviews from code owners, and the CI workflow as a required status check. This is a
repository setting and cannot be verified from the code.
