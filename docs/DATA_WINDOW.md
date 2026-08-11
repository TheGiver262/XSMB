# Rolling data window

After a settled draw on date `D`, the canonical training data is regenerated from upstream for the inclusive 730-calendar-day interval `[D - 729 days, D]`. Days without a draw remain absent from the observed-draw rows and are listed in `data/dataset_summary.json`.
