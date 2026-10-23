# CI/CD

## Workflows

| Workflow | File | Trigger |
|---|---|---|
| ci | `.github/workflows/ci.yml` | push / PR |
| cd | `.github/workflows/cd.yml` | push to main, tags |
| nightly-monitoring | `.github/workflows/monitoring.yml` | daily 02:00 UTC |

## Secrets

| Name | Purpose | Notes |
|---|---|---|
| `GITHUB_TOKEN` | GHCR push | Auto; needs `packages: write` |

Optional: `GHCR_USERNAME`, `GHCR_TOKEN`.

## Image tags

- `<git-sha-12>` on every build
- `vX.Y.Z` on semver tags
- `latest` on `main` only
