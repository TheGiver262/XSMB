# Live-generated files

- `forecasts/YYYY-MM-DD.csv`: immutable official live probability snapshot.
- `forecasts/YYYY-MM-DD_metrics.csv`: model/backtest metadata for the snapshot.
- `forecasts/manifest.csv`: cutoff dates and hashes.
- `evaluation/daily_metrics.csv`: realized live scores.
- `evaluation/rolling_summary.csv`: 30/60/90 settled-draw aggregates.
- `evaluation/model_runs.csv`: model-selection history.
- `output/next_preview.csv`: mutable post-draw preview for the next date.
- `output/pipeline_state.json`: settlement idempotency state.
