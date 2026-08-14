# XSMB Top-3 Walk-forward Report

- Data: 2005-10-01 -> 2026-08-14 (7522 draws)
- Development: 2023-08-04 -> 2025-08-10
- Final holdout: 2025-08-11 -> 2026-08-14 (365 draws)
- Selected model: `long_l10_lr010` / feature_set=`long_only` / L2=10.0 / online_lr=0.1

## Final holdout

- Top-1: 94/365 = 25.75% (random baseline 23.77%)
- Top-2 any-hit: 160/365 = 43.84% (random baseline 42.04%)
- Top-3 any-hit: **212/365 = 58.08%** (random baseline 56.06%)
- Top-3 lift: +2.02 percentage points
- 95% Wilson CI: [52.96%, 63.03%]
- z vs random: 0.778; two-sided p=0.4369
- Longest hit streak: 11 days; longest miss streak: 10 days

Important: a 50% hit rate is below the fair top-3 baseline (~56.06%), so 50% is not a valid promotion threshold.

## Development model selection

| candidate | feature set | L2 | online lr | top3 hit | Brier |
|---|---:|---:|---:|---:|---:|
| long_l10_lr010 | long_only | 10.0 | 0.1 | 57.12% | 0.181410 |
| long_l100_lr030 | long_only | 100.0 | 0.3 | 55.89% | 0.181438 |
| full_l300_lr010 | full | 300.0 | 0.1 | 55.48% | 0.181490 |
| recency_l300_lr030 | recency | 300.0 | 0.3 | 55.34% | 0.181592 |
| recgap_l300_lr010 | recency_gap | 300.0 | 0.1 | 54.93% | 0.181474 |
| full_l1000_lr030 | full | 1000.0 | 0.3 | 54.11% | 0.181729 |
| recency_l100_lr010 | recency | 100.0 | 0.1 | 53.01% | 0.181445 |

## Monthly holdout

| month | draws | top3 hits | hit rate | lift vs random |
|---|---:|---:|---:|---:|
| 2025-08 | 21 | 17 | 80.95% | +24.89 pp |
| 2025-09 | 30 | 18 | 60.00% | +3.94 pp |
| 2025-10 | 31 | 19 | 61.29% | +5.23 pp |
| 2025-11 | 30 | 10 | 33.33% | -22.73 pp |
| 2025-12 | 31 | 22 | 70.97% | +14.91 pp |
| 2026-01 | 31 | 15 | 48.39% | -7.68 pp |
| 2026-02 | 24 | 10 | 41.67% | -14.40 pp |
| 2026-03 | 31 | 16 | 51.61% | -4.45 pp |
| 2026-04 | 30 | 18 | 60.00% | +3.94 pp |
| 2026-05 | 31 | 21 | 67.74% | +11.68 pp |
| 2026-06 | 30 | 17 | 56.67% | +0.60 pp |
| 2026-07 | 31 | 19 | 61.29% | +5.23 pp |
| 2026-08 | 14 | 10 | 71.43% | +15.37 pp |

## Next forecast

- Target date: 2026-08-15
- Top 3: **51 - 28 - 50**
- This ranking is only a research output; if the holdout does not beat random baseline credibly, it should not be treated as proven edge.
