#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import walkforward_top3_v4 as v4
from walkforward_top3_v2 import DATA

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "data" / "manual" / "xsmb_2026-08-28.csv"
EFFECTIVE = ROOT / "evaluation" / "_v4_effective_xsmb.csv"


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def main() -> None:
    fields, base = read_rows(Path(DATA))
    o_fields, overlay = read_rows(OVERLAY)
    if fields != o_fields:
        raise RuntimeError("Manual overlay schema does not match upstream xsmb.csv")
    rows = {row["date"]: row for row in base}
    for row in overlay:
        if row["date"] in rows:
            if rows[row["date"]] != row:
                raise RuntimeError(f"Overlay conflicts with canonical data for {row['date']}")
        else:
            rows[row["date"]] = row
    EFFECTIVE.parent.mkdir(parents=True, exist_ok=True)
    with EFFECTIVE.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for day in sorted(rows):
            w.writerow(rows[day])
    v4.DATA = EFFECTIVE
    v4.main()


if __name__ == "__main__":
    main()
