# Immutable live forecasts

Each `YYYY-MM-DD.csv` file is an official pre-draw snapshot for that date. The pipeline never overwrites an existing snapshot. `manifest.csv` records cutoff dates and SHA-256 hashes once forecasts begin accumulating.
