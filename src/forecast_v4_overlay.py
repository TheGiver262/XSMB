#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from xsmb_probability import PRIZE_LENGTHS, load_draws, matrices
from walkforward_top3_v2 import DATA, score_day, station_for_date, target_features
from walkforward_top3_v4 import BASE_MODEL, SCHEMES, reverse_number, is_double, select_combo

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "data" / "manual_overlay_v4.json"
OUT = ROOT / "forecasts" / "top3_v4_next.json"


def load_overlay() -> list[dict]:
    if not OVERLAY.exists():
        return []
    raw = json.loads(OVERLAY.read_text(encoding="utf-8"))
    rows = raw if isinstance(raw, list) else raw.get("draws", [])
    out = []
    for row in rows:
        item = {"date": dt.date.fromisoformat(row["date"])}
        for col, width in PRIZE_LENGTHS.items():
            item[col] = str(row[col]).zfill(width)
        item["source"] = row.get("source")
        out.append(item)
    return out


def main() -> None:
    draws = load_draws(DATA)
    canonical_dates = {d["date"] for d in draws}
    for row in load_overlay():
        if row["date"] not in canonical_dates:
            draws.append({k: v for k, v in row.items() if k != "source"})
    draws.sort(key=lambda d: d["date"])

    presence, _counts = matrices(draws)
    target = draws[-1]["date"] + dt.timedelta(days=1)
    x_next = target_features(draws, presence, target)
    base_scores = score_day(x_next, BASE_MODEL)

    # V4 selection folds are frozen before 2025-08-11; current selected scheme
    # is marginal_pool12. Do not reselect using prospective/burned dates.
    selected = next(s for s in SCHEMES if s["name"] == "marginal_pool12")
    picks = select_combo(base_scores, presence, len(draws), selected)

    detail = []
    for n in picks:
        rev = reverse_number(n)
        detail.append({
            "number": f"{n:02d}",
            "base_score": float(base_scores[n]),
            "is_double": is_double(n),
            "reverse": None if is_double(n) else f"{rev:02d}",
            "reverse_base_score": None if is_double(n) else float(base_scores[rev]),
        })

    payload = {
        "target_date": target.isoformat(),
        "station": station_for_date(target),
        "model": "V4_combo_reverse",
        "selected_scheme": selected,
        "top3": [f"{n:02d}" for n in picks],
        "detail": detail,
        "data_cutoff": draws[-1]["date"].isoformat(),
        "canonical_cutoff": load_draws(DATA)[-1]["date"].isoformat(),
        "overlay_used": draws[-1]["date"] not in canonical_dates,
        "prospective_status": "valid_if_generated_before_draw_time",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
