#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

from xsmb_probability import PRIZE_LENGTHS, load_draws

ROOT = Path(__file__).resolve().parents[1]
THREE_DIGIT_COLS = [col for col, width in PRIZE_LENGTHS.items() if width >= 3]
ANNOTATION_FIELDS = [
    "historical_best_prefix_digit",
    "historical_best_3digit",
    "historical_best_3digit_count",
    "historical_best_3digit_hit_draws",
    "historical_best_3digit_position_rate_pct",
]


def suffix3_history(draws: list[dict]) -> tuple[list[int], list[int]]:
    """Return occurrence counts and distinct-draw hit counts for suffixes 000..999."""
    counts = [0] * 1000
    hit_draws = [0] * 1000
    for draw in draws:
        seen: set[int] = set()
        for col in THREE_DIGIT_COLS:
            suffix = int(str(draw[col])[-3:])
            counts[suffix] += 1
            seen.add(suffix)
        for suffix in seen:
            hit_draws[suffix] += 1
    return counts, hit_draws


def build_prefix3_stats(draws: list[dict]) -> list[dict]:
    """For every 2D suffix AB, choose the historically most frequent xAB suffix.

    Primary ranking is total occurrences across the 23 prize positions with at
    least three digits (DB through G6). Ties are broken by the number of draws
    containing the 3D suffix, then by the smaller prefix digit only to keep the
    output deterministic. This is descriptive metadata and never changes the
    2D model probability/rank.
    """
    counts, hit_draws = suffix3_history(draws)
    exposure = len(draws) * len(THREE_DIGIT_COLS)
    rows = []
    for number in range(100):
        candidates = []
        for prefix in range(10):
            suffix = prefix * 100 + number
            candidates.append((counts[suffix], hit_draws[suffix], prefix, suffix))
        candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))
        best = candidates[0]
        rows.append({
            "number": f"{number:02d}",
            "historical_best_prefix_digit": str(best[2]),
            "historical_best_3digit": f"{best[3]:03d}",
            "historical_best_3digit_count": best[0],
            "historical_best_3digit_hit_draws": best[1],
            "historical_best_3digit_position_rate_pct": (100.0 * best[0] / exposure) if exposure else 0.0,
        })
    return rows


def annotate_csv(path: Path, stats: list[dict]) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    by_number = {row["number"]: row for row in stats}
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        original_fields = [x for x in (reader.fieldnames or []) if x not in ANNOTATION_FIELDS]
    for row in rows:
        number = f"{int(row['number']):02d}"
        stat = by_number[number]
        for field in ANNOTATION_FIELDS:
            row[field] = stat[field]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=original_fields + ANNOTATION_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_stats(path: Path, stats: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(stats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "upstream" / "xsmb.csv"))
    ap.add_argument("--target-date", required=True)
    ap.add_argument(
        "--probability-csv",
        default=str(ROOT / "research" / "multihorizon_v3" / "next_prediction.csv"),
    )
    ap.add_argument(
        "--ranking-csv",
        default=str(ROOT / "research" / "rank_challenger" / "current_ranking.csv"),
    )
    ap.add_argument(
        "--stats-out",
        default=str(ROOT / "research" / "prefix3_history" / "current.csv"),
    )
    ap.add_argument(
        "--summary-out",
        default=str(ROOT / "research" / "prefix3_history" / "summary.json"),
    )
    args = ap.parse_args()

    target = dt.date.fromisoformat(args.target_date)
    draws = [draw for draw in load_draws(args.data) if draw["date"] < target]
    if not draws:
        raise RuntimeError("No historical draws before target date")

    stats = build_prefix3_stats(draws)
    annotate_csv(Path(args.probability_csv), stats)
    annotate_csv(Path(args.ranking_csv), stats)
    write_stats(Path(args.stats_out), stats)

    summary = {
        "target_date": target.isoformat(),
        "data_cutoff": draws[-1]["date"].isoformat(),
        "historical_draws": len(draws),
        "eligible_positions_per_draw": len(THREE_DIGIT_COLS),
        "definition": "For each 2D suffix AB, choose the most frequent 3D suffix xAB among DB through G6 using only draws before target_date.",
        "selection_rule": "max total occurrences; tie-break by distinct draw hits, then lower prefix digit",
        "prediction_effect": "descriptive annotation only; does not alter 2D probability, score, or rank",
    }
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Top-prefix annotations added to probability and ranking CSVs")


if __name__ == "__main__":
    main()
