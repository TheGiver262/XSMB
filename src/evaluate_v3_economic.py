#!/usr/bin/env python3
"""Strict economic-style evaluation for V3 XSMB top-3 predictions.

This evaluator deliberately separates three concepts that must not be conflated:
1) coverage: at least one predicted number appears;
2) result quality: 0/3, 1/3, 2/3 or 3/3 distinct picks hit and how many total occurrences (nhay);
3) monetary P&L: requires an explicit stake/payout configuration and is therefore NOT invented here.

Current user rule for a conservative proxy:
- 0/3 = loss
- 1/3 = loss
- 2/3 or 3/3 = strict success proxy

The script also reports total occurrences of each pick among the 27 published XSMB
prize suffixes, so a later payout configuration can compute exact P&L without
rerunning predictions or introducing look-ahead.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from xsmb_probability import PRIZE_COLS, load_draws

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "upstream" / "xsmb.csv"
DAILY = ROOT / "evaluation" / "walkforward_top3_v3_daily.csv"
OUT_DAILY = ROOT / "evaluation" / "walkforward_top3_v3_economic_daily.csv"
OUT_SUMMARY = ROOT / "evaluation" / "walkforward_top3_v3_economic_summary.json"


def suffix_counter(draw: dict) -> Counter[int]:
    return Counter(int(str(draw[col])[-2:]) for col in PRIZE_COLS)


def main() -> None:
    draws = load_draws(DATA)
    by_date = {d["date"].isoformat(): d for d in draws}

    with DAILY.open("r", encoding="utf-8", newline="") as f:
        pred_rows = list(csv.DictReader(f))

    out = []
    dist = {"0_of_3": 0, "1_of_3": 0, "2_of_3": 0, "3_of_3": 0}
    total_occ_dist: Counter[int] = Counter()
    coverage_days = 0
    strict_success_days = 0
    total_occurrences = 0

    for row in pred_rows:
        day = row["date"]
        draw = by_date.get(day)
        if draw is None:
            raise RuntimeError(f"Missing actual draw for {day}")
        counts = suffix_counter(draw)
        picks = [int(row[f"pick{i}"]) for i in (1, 2, 3)]
        nhay = [int(counts.get(p, 0)) for p in picks]
        hit_flags = [int(x > 0) for x in nhay]
        distinct_hits = sum(hit_flags)
        total_nhay = sum(nhay)

        coverage = int(distinct_hits >= 1)
        # User-defined conservative proxy: one of three hitting still counts as a losing day.
        strict_success = int(distinct_hits >= 2)
        coverage_days += coverage
        strict_success_days += strict_success
        total_occurrences += total_nhay
        dist[f"{distinct_hits}_of_3"] += 1
        total_occ_dist[total_nhay] += 1

        # Abstract break-even return multiple if one equal stake unit is placed on each
        # of the 3 picks and every occurrence pays the same gross multiple. This is NOT
        # a real-market payout assumption; it only allows later exact P&L parameterization.
        breakeven_return_multiple = (3.0 / total_nhay) if total_nhay > 0 else None

        out.append({
            "date": day,
            "station": row["station"],
            "pick1": f"{picks[0]:02d}",
            "pick1_nhay": nhay[0],
            "pick2": f"{picks[1]:02d}",
            "pick2_nhay": nhay[1],
            "pick3": f"{picks[2]:02d}",
            "pick3_nhay": nhay[2],
            "distinct_hits": distinct_hits,
            "total_nhay": total_nhay,
            "coverage_any_hit": coverage,
            "strict_success_2plus_of_3": strict_success,
            "user_rule_result": "success_proxy" if strict_success else "loss",
            "breakeven_gross_return_multiple_per_nhay_equal_stake": (
                "" if breakeven_return_multiple is None else f"{breakeven_return_multiple:.6f}"
            ),
        })

    OUT_DAILY.parent.mkdir(parents=True, exist_ok=True)
    with OUT_DAILY.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)

    n = len(out)
    summary = {
        "n": n,
        "metric_warning": "coverage_any_hit is not a monetary win rate",
        "user_rule": "0/3 and 1/3 are losses; 2/3 or 3/3 is only a strict success proxy until an explicit payout/stake is configured",
        "coverage_any_hit_days": coverage_days,
        "coverage_any_hit_rate": coverage_days / n,
        "strict_success_2plus_of_3_days": strict_success_days,
        "strict_success_2plus_of_3_rate": strict_success_days / n,
        "distinct_hit_distribution": dist,
        "distinct_hit_distribution_rate": {k: v / n for k, v in dist.items()},
        "total_nhay": total_occurrences,
        "mean_total_nhay_per_day": total_occurrences / n,
        "total_nhay_distribution": {str(k): v for k, v in sorted(total_occ_dist.items())},
        "pnl_status": "not_computed_without_explicit_payout_and_stake",
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

# maintenance rerun marker: 2026-08-21
