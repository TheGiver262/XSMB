#!/usr/bin/env python3
"""Leakage-safe XSMB top-3 walk-forward V3.

V3 keeps short-term evidence as the core signal and treats the scheduled
issuing station only as an optional auxiliary feature. Station weight is
selected on a development validation slice and is automatically shrunk to zero
unless it improves that slice by a minimum number of hits.

Protocol
--------
- latest 365 draws: untouched final holdout;
- previous 730 draws: development only, split into:
  * first 365: choose short-term core weights;
  * second 365: validate station weight;
- every target date uses only history strictly before that date;
- holdout is opened only after core + station policy are locked.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np

from xsmb_probability import load_draws, matrices
from walkforward_top3_v2 import (
    DATA,
    FEATURE_NAMES,
    TOP3_BASELINE,
    build_feature_tensor,
    run_window,
    score_day,
    station_for_date,
    target_features,
    top3,
)

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "evaluation"
FORECAST = ROOT / "forecasts"
HOLDOUT_DRAWS = 365
DEV_DRAWS = 730
CORE_DEV_DRAWS = 365
STATION_MIN_GAIN_HITS = 4

# All core candidates are deliberately short-term dominated. Station columns
# are zero here and are calibrated only after a core has been selected.
CORE_CANDIDATES = [
    {"name": "month50", "weights": [0.05, 0.20, 0.15, 0.10, 0.50, 0.0, 0.0]},
    {"name": "short_month33", "weights": [0.08, 0.27, 0.18, 0.14, 0.33, 0.0, 0.0]},
    {"name": "week_month40", "weights": [0.05, 0.30, 0.15, 0.10, 0.40, 0.0, 0.0]},
    {"name": "week_heavy", "weights": [0.05, 0.40, 0.20, 0.15, 0.20, 0.0, 0.0]},
    {"name": "mid_horizon", "weights": [0.05, 0.15, 0.30, 0.25, 0.25, 0.0, 0.0]},
    {"name": "month40", "weights": [0.05, 0.25, 0.15, 0.15, 0.40, 0.0, 0.0]},
]

STATION_ALPHAS = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15]
STATION_RECENT_MIXES = [0.0, 0.5]


def stationized(core: dict, alpha: float, recent_mix: float) -> dict:
    base = np.asarray(core["weights"][:5], dtype=float)
    base = base / base.sum()
    scaled = (1.0 - alpha) * base
    station_rate = alpha * (1.0 - recent_mix)
    station_recent = alpha * recent_mix
    return {
        "name": f"{core['name']}_st{alpha:.3f}_r{recent_mix:.1f}",
        "weights": [*scaled.tolist(), station_rate, station_recent],
        "core": core["name"],
        "station_alpha": alpha,
        "station_recent_mix": recent_mix,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def wilson(hits: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = hits / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return center - margin, center + margin


def p_vs_random(hits: int, n: int) -> tuple[float, float]:
    expected = n * TOP3_BASELINE
    var = n * TOP3_BASELINE * (1 - TOP3_BASELINE)
    z = (hits - expected) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return z, p


def monthly(daily: list[dict]) -> list[dict]:
    buckets: dict[str, list[dict]] = {}
    for row in daily:
        buckets.setdefault(row["date"][:7], []).append(row)
    out = []
    for month in sorted(buckets):
        rows = buckets[month]
        hits = sum(int(r["top3_any_hit"]) for r in rows)
        out.append({
            "month": month,
            "draws": len(rows),
            "hits": hits,
            "hit_rate": hits / len(rows),
            "lift_pp_vs_random": (hits / len(rows) - TOP3_BASELINE) * 100.0,
        })
    return out


def main() -> None:
    draws = load_draws(DATA)
    presence, _counts = matrices(draws)
    features = build_feature_tensor(draws, presence)
    n = len(draws)
    holdout_start = n - HOLDOUT_DRAWS
    dev_start = holdout_start - DEV_DRAWS
    dev_mid = dev_start + CORE_DEV_DRAWS
    if dev_start < 365:
        raise RuntimeError("Not enough history for V3 protocol")

    # Stage 1: choose the short-term core using only the first half of dev.
    core_rows = []
    for core in CORE_CANDIDATES:
        stats, _ = run_window(draws, presence, features, dev_start, dev_mid, core, collect=False)
        core_rows.append({
            "candidate": core["name"],
            "hits": stats["hits"],
            "hit_rate": stats["hit_rate"],
            "lift_pp_vs_random": stats["lift_pp_vs_random"],
            "pick_set_change_rate": stats["pick_set_change_rate"],
            "weights": json.dumps(dict(zip(FEATURE_NAMES, core["weights"])), sort_keys=True),
        })
    core_rows.sort(key=lambda r: (-r["hits"], -r["pick_set_change_rate"], r["candidate"]))
    chosen_core = next(c for c in CORE_CANDIDATES if c["name"] == core_rows[0]["candidate"])

    # Stage 2: station is allowed only if it adds enough validation hits on the
    # second half of dev. This is the automatic shrink-to-zero gate.
    station_rows = []
    station_candidates = []
    for alpha in STATION_ALPHAS:
        mixes = [0.0] if alpha == 0 else STATION_RECENT_MIXES
        for mix in mixes:
            cand = stationized(chosen_core, alpha, mix)
            station_candidates.append(cand)
            stats, _ = run_window(draws, presence, features, dev_mid, holdout_start, cand, collect=False)
            station_rows.append({
                "candidate": cand["name"],
                "station_alpha": alpha,
                "station_recent_mix": mix,
                "hits": stats["hits"],
                "hit_rate": stats["hit_rate"],
                "lift_pp_vs_random": stats["lift_pp_vs_random"],
                "pick_set_change_rate": stats["pick_set_change_rate"],
                "weights": json.dumps(dict(zip(FEATURE_NAMES, cand["weights"])), sort_keys=True),
            })
    station_rows.sort(key=lambda r: (-r["hits"], r["station_alpha"], r["station_recent_mix"]))
    zero_row = next(r for r in station_rows if float(r["station_alpha"]) == 0.0)
    best_row = station_rows[0]
    station_gain_hits = int(best_row["hits"]) - int(zero_row["hits"])
    if float(best_row["station_alpha"]) > 0 and station_gain_hits >= STATION_MIN_GAIN_HITS:
        selected_row = best_row
        station_shrunk = False
    else:
        selected_row = zero_row
        station_shrunk = True
    selected = next(c for c in station_candidates if c["name"] == selected_row["candidate"])

    # Final holdout is opened only now.
    holdout_stats, daily = run_window(draws, presence, features, holdout_start, n, selected, collect=True)
    no_station = stationized(chosen_core, 0.0, 0.0)
    no_station_stats, _ = run_window(draws, presence, features, holdout_start, n, no_station, collect=False)

    low, high = wilson(holdout_stats["hits"], HOLDOUT_DRAWS)
    z, p = p_vs_random(holdout_stats["hits"], HOLDOUT_DRAWS)

    target = draws[-1]["date"] + dt.timedelta(days=1)
    x_next = target_features(draws, presence, target)
    next_scores = score_day(x_next, selected)
    picks = top3(next_scores)
    details = []
    for number in picks:
        item = {"number": f"{number:02d}", "score": float(next_scores[number])}
        for j, name in enumerate(FEATURE_NAMES):
            item[name] = float(x_next[number, j])
        details.append(item)

    summary = {
        "protocol": {
            "development_draws": DEV_DRAWS,
            "development_core_selection_draws": CORE_DEV_DRAWS,
            "development_station_validation_draws": DEV_DRAWS - CORE_DEV_DRAWS,
            "final_holdout_draws": HOLDOUT_DRAWS,
            "station_min_gain_hits": STATION_MIN_GAIN_HITS,
            "ordering": "predict from strict prior history -> reveal result -> next date may use it",
            "random_top3_baseline": TOP3_BASELINE,
        },
        "data": {
            "first_date": draws[0]["date"].isoformat(),
            "latest_date": draws[-1]["date"].isoformat(),
            "draws": n,
            "core_dev_start": draws[dev_start]["date"].isoformat(),
            "core_dev_end": draws[dev_mid - 1]["date"].isoformat(),
            "station_validation_start": draws[dev_mid]["date"].isoformat(),
            "station_validation_end": draws[holdout_start - 1]["date"].isoformat(),
            "holdout_start": draws[holdout_start]["date"].isoformat(),
            "holdout_end": draws[-1]["date"].isoformat(),
        },
        "selected_core": chosen_core,
        "station_gate": {
            "best_validation_candidate": best_row,
            "no_station_validation": zero_row,
            "gain_hits": station_gain_hits,
            "shrunk_to_zero": station_shrunk,
            "selected_candidate": selected,
        },
        "holdout": {
            **holdout_stats,
            "wilson_95_low": low,
            "wilson_95_high": high,
            "z_vs_random": z,
            "two_sided_p_vs_random": p,
        },
        "holdout_no_station_control": no_station_stats,
        "next_forecast": {
            "target_date": target.isoformat(),
            "station": station_for_date(target),
            "top3": [f"{x:02d}" for x in picks],
            "detail": details,
        },
    }

    write_csv(EVAL / "walkforward_top3_v3_core_selection.csv", core_rows)
    write_csv(EVAL / "walkforward_top3_v3_station_validation.csv", station_rows)
    write_csv(EVAL / "walkforward_top3_v3_daily.csv", daily)
    write_csv(EVAL / "walkforward_top3_v3_monthly.csv", monthly(daily))
    (EVAL / "walkforward_top3_v3_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (FORECAST / "top3_v3_next.json").write_text(
        json.dumps(summary["next_forecast"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(json.dumps({
        "selected_core": chosen_core["name"],
        "station_alpha": selected.get("station_alpha", 0.0),
        "station_shrunk_to_zero": station_shrunk,
        "holdout_hit_rate": holdout_stats["hit_rate"],
        "holdout_lift_pp": holdout_stats["lift_pp_vs_random"],
        "p_vs_random": p,
        "next_top3": [f"{x:02d}" for x in picks],
    }, indent=2))


if __name__ == "__main__":
    main()
