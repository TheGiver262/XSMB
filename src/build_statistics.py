#!/usr/bin/env python3
"""Generate detailed two-year descriptive statistics for XSMB 00..99."""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import numpy as np

from xsmb_probability import P0, PRIZE_COLS, load_draws, matrices

SPLIT = dt.date(2025, 8, 11)


def max_gap_and_streak(series: np.ndarray) -> tuple[int, int]:
    max_gap = 0
    current_gap = 0
    max_streak = 0
    current_streak = 0
    for hit in series.astype(int):
        if hit:
            max_gap = max(max_gap, current_gap)
            current_gap = 0
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_gap += 1
            current_streak = 0
    max_gap = max(max_gap, current_gap)
    return int(max_gap), int(max_streak)


def current_gap(series: np.ndarray) -> int:
    hits = np.flatnonzero(series)
    return int(len(series) - 1 - hits[-1]) if len(hits) else int(len(series))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    draws = load_draws(root / "data" / "parts")
    presence, counts = matrices(draws)
    dates = [d["date"] for d in draws]
    split_idx = next(i for i, day in enumerate(dates) if day >= SPLIT)

    special = np.zeros((len(draws), 100), dtype=np.int8)
    for i, draw in enumerate(draws):
        special[i, int(draw["special"][-2:])] = 1

    rows = []
    for number in range(100):
        max_gap, longest_streak = max_gap_and_streak(presence[:, number])
        prev_presence = int(presence[:split_idx, number].sum())
        recent_presence = int(presence[split_idx:, number].sum())
        prev_rate = prev_presence / split_idx
        recent_rate = recent_presence / (len(draws) - split_idx)
        rows.append({
            "number": f"{number:02d}",
            "draw_presence_2y": int(presence[:, number].sum()),
            "presence_rate_2y": float(presence[:, number].mean()),
            "occurrences_2y": int(counts[:, number].sum()),
            "occurrence_rate_per_prize": float(counts[:, number].sum() / (len(draws) * 27)),
            "presence_prev_year": prev_presence,
            "presence_rate_prev_year": prev_rate,
            "presence_recent_year": recent_presence,
            "presence_rate_recent_year": recent_rate,
            "presence_rate_delta_recent_minus_prev": recent_rate - prev_rate,
            "occurrences_prev_year": int(counts[:split_idx, number].sum()),
            "occurrences_recent_year": int(counts[split_idx:, number].sum()),
            "presence_last_30": int(presence[-30:, number].sum()),
            "presence_rate_last_30": float(presence[-30:, number].mean()),
            "presence_last_90": int(presence[-90:, number].sum()),
            "presence_rate_last_90": float(presence[-90:, number].mean()),
            "presence_last_180": int(presence[-180:, number].sum()),
            "presence_rate_last_180": float(presence[-180:, number].mean()),
            "current_gap_draws": current_gap(presence[:, number]),
            "max_gap_draws_2y": max_gap,
            "longest_presence_streak_2y": longest_streak,
            "special_hits_2y": int(special[:, number].sum()),
            "special_hits_prev_year": int(special[:split_idx, number].sum()),
            "special_hits_recent_year": int(special[split_idx:, number].sum()),
        })

    out = root / "output" / "statistics_2y_00_99.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_presence = sorted(rows, key=lambda r: r["presence_rate_2y"], reverse=True)
    by_occurrence = sorted(rows, key=lambda r: r["occurrences_2y"], reverse=True)
    by_gap = sorted(rows, key=lambda r: r["current_gap_draws"], reverse=True)
    biggest_up = sorted(rows, key=lambda r: r["presence_rate_delta_recent_minus_prev"], reverse=True)
    biggest_down = list(reversed(biggest_up))

    mean_presence = float(presence.mean(axis=0).mean())
    summary_rows = [
        ["window_start", dates[0].isoformat()],
        ["window_end", dates[-1].isoformat()],
        ["observed_draws", len(draws)],
        ["total_prize_observations", len(draws) * len(PRIZE_COLS)],
        ["theoretical_presence_probability", P0],
        ["mean_empirical_presence_probability", mean_presence],
        ["highest_presence_number", by_presence[0]["number"]],
        ["highest_presence_rate", by_presence[0]["presence_rate_2y"]],
        ["lowest_presence_number", by_presence[-1]["number"]],
        ["lowest_presence_rate", by_presence[-1]["presence_rate_2y"]],
        ["highest_occurrence_number", by_occurrence[0]["number"]],
        ["highest_occurrences", by_occurrence[0]["occurrences_2y"]],
        ["longest_current_gap_number", by_gap[0]["number"]],
        ["longest_current_gap_draws", by_gap[0]["current_gap_draws"]],
        ["largest_recent_increase_number", biggest_up[0]["number"]],
        ["largest_recent_increase", biggest_up[0]["presence_rate_delta_recent_minus_prev"]],
        ["largest_recent_decrease_number", biggest_down[0]["number"]],
        ["largest_recent_decrease", biggest_down[0]["presence_rate_delta_recent_minus_prev"]],
    ]
    with (root / "output" / "statistics_2y_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(summary_rows)

    print(f"Wrote {out}")
    print("Top 10 by 2-year draw presence:")
    for r in by_presence[:10]:
        print(r["number"], f"{r['presence_rate_2y']:.4%}", r["draw_presence_2y"])


if __name__ == "__main__":
    main()
