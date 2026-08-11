# Predictive research architecture

The project separates two horizons:

- **Full history (2005 → present):** research, statistical testing and walk-forward feature screening.
- **Rolling 3 calendar years (1,095 days):** live training/calibration and daily forecasts.

## Prediction targets

### 2 digits
`00..99` is a hit when it appears as the 2-digit suffix of at least one of all 27 published prizes. Fair baseline: `1 - (99/100)^27`.

### 3 digits: `suffix3_any`
`000..999` is a hit when it appears as the last three digits of a prize with at least three published digits. There are 23 eligible positions: DB, G1, G2, G3, G4, G5 and G6. G7 is excluded because it only publishes two digits. Fair baseline: `1 - (999/1000)^23`.

### 3 digits: `g6_exact`
`000..999` is a hit when it appears in any of the three exact G6 results. Fair baseline: `1 - (999/1000)^3`.

## Predictive feature gate

`src/research_predictive.py` performs full-history walk-forward screening using only data available before each evaluated draw. Optional signal groups must improve Brier score overall **and** in at least 60% of calendar-year folds before they become eligible in live models.

The live model performs another validation/calibration step inside the rolling 3-year window. Even an approved signal can be assigned `blend = 0`, returning the forecast to the theoretical fair-draw baseline.

This prevents descriptive anomalies, long gaps or short-term frequency spikes from automatically becoming predictive features.

## Upstream mirror

`src/sync_upstream.py` mirrors the three canonical CSV datasets from `khiemdoan/vietnam-lottery-xsmb-analysis`:

- `xsmb.csv`: raw prize results;
- `xsmb-2-digits.csv`: 2-digit suffixes;
- `xsmb-sparse.csv`: daily 00–99 counts.

These three files contain the complete underlying lottery records represented by the upstream project. JSON and Parquet are alternate encodings of the same records and are not committed daily here to avoid unnecessary Git history growth.
