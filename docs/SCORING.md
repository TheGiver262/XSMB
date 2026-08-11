# Scoring

Live forecasts are evaluated against the 100 binary events “number 00–99 appeared at least once among the 27 prize suffixes”. Primary metrics are mean Brier score and mean log-loss across all 100 numbers. The fair-draw baseline uses `1 - 0.99^27` for every number.
