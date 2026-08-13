# Automation schedule and security model

- `multihorizon-v3.yml`: 08:00 UTC / 15:00 Vietnam time.
- `daily-forecast.yml`: 09:00 UTC / 16:00 Vietnam time.
- `daily-settle.yml`: 13:30 UTC / 20:30 Vietnam time with a 14:30 UTC / 21:30 fallback.
- `challenger-paper-settle.yml`: 14:30 UTC / 21:30 Vietnam time during the paper-test window.
- `full-research.yml`: Sunday 15:15 UTC / 22:15 Vietnam time.
- `ci.yml`: test changes on PRs and relevant pushes to `main`.

## Security model

- Third-party GitHub Actions are pinned to immutable commit SHAs.
- Python dependencies are pinned to exact versions in `requirements.txt`.
- Workflows that execute Python/model code run that code in a `contents: read` job with checkout credentials removed.
- Repository writes happen only in a separate clean job with `contents: write`, after downloading and validating a narrowly scoped artifact.
- Upstream lottery history is accepted only as an append-only extension of the already trusted `data/upstream/xsmb.csv`; historical mutation, shrinkage, future dates, schema errors, or cross-file date mismatches fail closed.
- `actions-history-cleanup.yml` keeps at least 7 days of run history and defaults to 30 days for audit/forensics.
