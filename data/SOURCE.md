# Data source

Primary source:
https://raw.githubusercontent.com/khiemdoan/vietnam-lottery-xsmb-analysis/refs/heads/main/data/xsmb.csv

Project:
https://github.com/khiemdoan/vietnam-lottery-xsmb-analysis

Window used: 2024-08-11 through 2026-08-10 (730 calendar days).
Observed draw records: 722.

Breakdown:
- 2024-08-11 through 2025-08-10: 361 draws, fetched from the upstream historical dataset.
- 2025-08-11 through 2026-08-10: 361 draws, preserving the previously checked repository segment.

No XSMB records in this two-year window:
- 2025-01-28
- 2025-01-29
- 2025-01-30
- 2025-01-31
- 2026-02-16
- 2026-02-17
- 2026-02-18
- 2026-02-19

Independent spot-check for the recent segment:
https://xskt.com.vn/xsmb/ngay-11-8-2025

Dataset layout in this repository: 8 chronological CSV parts under `data/parts/`.
Machine-readable window metadata: `data/dataset_summary.json`.
The rebuild procedure is implemented in `src/extend_history.py` and validated by the GitHub Actions workflow.
