#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "upstream" / "xsmb-2-digits.csv"


def load_rows():
    rows = []
    with DATA.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        prize_cols = [c for c in (reader.fieldnames or []) if c != "date"]
        for row in reader:
            d = date.fromisoformat(row["date"])
            nums = {str(row[c]).strip().zfill(2) for c in prize_cols if row.get(c) not in (None, "")}
            rows.append((d, nums))
    rows.sort(key=lambda x: x[0])
    return rows


def maximal_runs(rows, suffix: str, strict_calendar: bool):
    runs = []
    start = None
    prev = None
    length = 0
    for d, nums in rows:
        present = suffix in nums
        adjacent = prev is not None and (not strict_calendar or d == prev + timedelta(days=1))
        if present:
            if start is not None and adjacent:
                length += 1
            else:
                if start is not None:
                    runs.append((start, prev, length))
                start = d
                length = 1
        else:
            if start is not None:
                runs.append((start, prev, length))
                start = None
                length = 0
        prev = d
    if start is not None:
        runs.append((start, prev, length))
    return runs


def summarize(rows, strict_calendar: bool):
    per_number = []
    all_ge7 = []
    for n in range(100):
        s = f"{n:02d}"
        runs = maximal_runs(rows, s, strict_calendar)
        longest = max(runs, key=lambda x: x[2]) if runs else (None, None, 0)
        ge7 = [r for r in runs if r[2] >= 7]
        seven_windows = sum(r[2] - 6 for r in ge7)
        row = {
            "number": s,
            "longest_run": longest[2],
            "longest_start": longest[0].isoformat() if longest[0] else None,
            "longest_end": longest[1].isoformat() if longest[1] else None,
            "events_ge7": len(ge7),
            "seven_day_windows": seven_windows,
        }
        per_number.append(row)
        for r in ge7:
            all_ge7.append({"number": s, "start": r[0].isoformat(), "end": r[1].isoformat(), "length": r[2]})

    max_len = max(x["longest_run"] for x in per_number)
    longest = [x for x in per_number if x["longest_run"] == max_len]
    max_events = max(x["events_ge7"] for x in per_number)
    max_windows = max(x["seven_day_windows"] for x in per_number)
    return {
        "distinct_numbers_with_ge7": sum(x["events_ge7"] > 0 for x in per_number),
        "total_maximal_events_ge7": len(all_ge7),
        "longest_run_length": max_len,
        "longest_run_holders": longest,
        "max_events_ge7_per_number": max_events,
        "numbers_with_max_events_ge7": [x for x in per_number if x["events_ge7"] == max_events],
        "max_seven_day_windows_per_number": max_windows,
        "numbers_with_max_seven_day_windows": [x for x in per_number if x["seven_day_windows"] == max_windows],
        "top_by_events_ge7": sorted(per_number, key=lambda x: (-x["events_ge7"], -x["seven_day_windows"], -x["longest_run"], x["number"]))[:20],
        "top_by_longest": sorted(per_number, key=lambda x: (-x["longest_run"], -x["events_ge7"], x["number"]))[:20],
        "all_events_ge7": sorted(all_ge7, key=lambda x: (-x["length"], x["start"], x["number"])),
    }


def main():
    rows = load_rows()
    gaps = []
    for (d0, _), (d1, _) in zip(rows, rows[1:]):
        if d1 != d0 + timedelta(days=1):
            gaps.append({"from": d0.isoformat(), "to": d1.isoformat(), "days": (d1-d0).days})
    out = {
        "draws": len(rows),
        "start": rows[0][0].isoformat(),
        "end": rows[-1][0].isoformat(),
        "calendar_gaps": len(gaps),
        "largest_calendar_gap_days": max((g["days"] for g in gaps), default=1),
        "strict_calendar_days": summarize(rows, True),
        "consecutive_recorded_draws": summarize(rows, False),
    }
    print("STREAK_STATS_JSON=" + json.dumps(out, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
