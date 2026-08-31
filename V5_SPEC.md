# V5 Set-Coverage Spec

Primary target: choose exactly 10 distinct 2-digit suffixes from 00..99 for each target draw using only information available before that draw.

Strict success: at least 3 distinct selected suffixes appear among the 27 XSMB 2-digit result positions.

Primary metric: strict success rate = draws with distinct_hits >= 3 / valid prospective or OOS forecasts.

Secondary metrics: full distinct-hit distribution 0..10, total flashes across the selected set, and optional DB/G1 coverage reported separately.

Random-set analytical baseline for the primary metric is approximately 43.9%; validation must compare V5 against this baseline and against V4-derived ranking baselines.

Leakage rules:
- Forecast for D may only use outcomes before D.
- Train/validation/test folds are chronological.
- Candidate selection, feature selection, reverse/pair/triple weights and hyperparameters are selected inside training/validation only.
- Reverse relationships, double-number penalties/bonuses, DB/G1 terms, flash-intensity terms, pair interactions and triple interactions default to zero unless OOS validation supports them.
- No retuning from a single prospective result.
- Prospective forecasts are immutable once locked.

V4 remains a comparator/individual-signal source; V5 is a set-level optimizer rather than a replacement justified by recent misses.
