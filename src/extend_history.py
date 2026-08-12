#!/usr/bin/env python3
"""Extend the checked XSMB dataset backward by one year.

The repository already contains the verified recent segment 2025-08-11..2026-08-10.
This script downloads the upstream historical CSV, selects 2024-08-11..2025-08-10,
merges the two non-overlapping segments, validates the result, then rewrites data/parts
as chronological chunks and emits a machine-readable summary.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import urllib.request
from pathlib import Path

from xsmb_probability import PRIZE_LENGTHS, load_draws

UPSTREAM_URL = (
    "https://raw.githubusercontent.com/khiemdoan/"
    "vietnam-lottery-xsmb-analysis/refs/heads/main/data/xsmb.csv"
)
START = dt.date(2024, 8, 11)
RECENT_START = dt.date(2025, 8, 11)
END = dt.date(2026, 8, 10)
PREVIOUS_END = RECENT_START - dt.timedelta(days=1)
FIELDNAMES = ["date", *PRIZE_LENGTHS.keys()]


def normalize_row(row: dict[str, str]) -> dict[str, str]:
    out = {"date": row["date"]}
    for col, width in PRIZE_LENGTHS.items():
        out[col] = str(row[col]).zfill(width)
    return out


def row_from_draw(draw: dict) -> dict[str, str]:
    return {
        "date": draw["date"].isoformat(),
        **{col: str(draw[col]).zfill(width) for col, width in PRIZE_LENGTHS.items()},
    }


def download_previous_year() -> list[dict[str, str]]:
    req = urllib.request.Request(
        UPSTREAM_URL,
        headers={"User-Agent": "xsmb-probability-research/1.0"},
    )
    with urllib.request.urlopen(req, timeout=45) as response:
        text = response.read().decode("utf-8-sig")

    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        day = dt.date.fromisoformat(row["date"])
        if START <= day <= PREVIOUS_END:
            rows.append(normalize_row(row))
    rows.sort(key=lambda r: r["date"])
    return rows


def validate(rows: list[dict[str, str]]) -> dict:
    dates = [dt.date.fromisoformat(r["date"]) for r in rows]
    if dates != sorted(dates):
        raise ValueError("Dataset is not chronological")
    if len(dates) != len(set(dates)):
        raise ValueError("Duplicate draw dates detected")
    if not dates or dates[0] != START or dates[-1] != END:
        raise ValueError(
            f"Unexpected date bounds: {dates[0] if dates else None}..{dates[-1] if dates else None}"
        )

    expected_calendar = []
    d = START
    while d <= END:
        expected_calendar.append(d)
        d += dt.timedelta(days=1)
    missing = sorted(set(expected_calendar) - set(dates))

    previous_count = sum(START <= d <= PREVIOUS_END for d in dates)
    recent_count = sum(RECENT_START <= d <= END for d in dates)
    if previous_count < 350 or recent_count < 350:
        raise ValueError(
            f"Suspicious yearly counts: previous={previous_count}, recent={recent_count}"
        )

    return {
        "window_start": START.isoformat(),
        "window_end": END.isoformat(),
        "calendar_days": len(expected_calendar),
        "observed_draws": len(rows),
        "previous_year_draws": previous_count,
        "recent_year_draws": recent_count,
        "missing_calendar_dates": [d.isoformat() for d in missing],
        "upstream_url": UPSTREAM_URL,
    }


def write_parts(rows: list[dict[str, str]], data_dir: Path, chunk_size: int = 100) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for old in data_dir.glob("xsmb_part_*.csv"):
        old.unlink()

    for part_no, start in enumerate(range(0, len(rows), chunk_size), 1):
        path = data_dir / f"xsmb_part_{part_no:02d}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows[start : start + chunk_size])


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "parts"

    # Preserve only the already checked recent segment from the repository.
    recent_draws = [
        d for d in load_draws(data_dir) if RECENT_START <= d["date"] <= END
    ]
    if not recent_draws:
        raise ValueError("No existing recent checked segment found")
    if recent_draws[0]["date"] != RECENT_START or recent_draws[-1]["date"] != END:
        raise ValueError(
            f"Recent segment bounds are {recent_draws[0]['date']}..{recent_draws[-1]['date']}, expected {RECENT_START}..{END}"
        )

    previous_rows = download_previous_year()
    recent_rows = [row_from_draw(d) for d in recent_draws]
    rows = sorted(previous_rows + recent_rows, key=lambda r: r["date"])

    summary = validate(rows)
    write_parts(rows, data_dir)

    summary_path = root / "data" / "dataset_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
