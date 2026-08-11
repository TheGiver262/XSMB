# Live pipeline invariants

- Official live forecasts are written to `forecasts/YYYY-MM-DD.csv` before the draw and are immutable.
- Forecast training always ends at `target_date - 1 day`.
- If the target draw already exists upstream, live forecast creation is refused.
- Settlement is only recorded when an observed draw exists upstream.
- The canonical training dataset rolls over the latest 730 calendar days.
- Live performance is tracked separately from historical backtests with Brier score and log-loss over 30/60/90 settled draws.
