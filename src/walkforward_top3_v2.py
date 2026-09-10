#!/usr/bin/env python3
"""Leakage-safe XSMB top-3 walk-forward V2.

V2 deliberately gives more ranking weight to short-horizon evidence than to
long-run frequency. It also tests the scheduled issuing station as a feature.
The physical Northern draw is common/centralized; `station` here means the
scheduled issuing company/day label (Ha Noi, Quang Ninh, Bac Ninh, Hai Phong,
Nam Dinh, Thai Binh), not a claim that the physical draw machine moves there.

Protocol:
- latest 365 draws = untouched final holdout;
- previous 730 draws = development/model selection only;
- every evaluated date uses only draws strictly before that date;
- candidate weights are fixed before holdout is opened;
- each day's result becomes available only for features of the next day.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from xsmb_probability import P0, PRIZE_COLS, load_draws, matrices

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "upstream" / "xsmb.csv"
EVAL = ROOT / "evaluation"
FORECAST = ROOT / "forecasts"
HOLDOUT_DRAWS = 365
DEV_DRAWS = 730
LONG_LOOKBACK = 5 * 365
TOP3_BASELINE = 1.0 - 0.97**27

FEATURE_NAMES = [
    "long_rate",
    "rate_7",
    "rate_14",
    "rate_30",
    "month_rate",
    "station_rate",
    "station_recent_rate",
]

# Python weekday: Monday=0 ... Sunday=6.
# Ha Noi is scheduled on both Monday and Thursday.
STATION_BY_WEEKDAY = {
    0: "Ha Noi",
    1: "Quang Ninh",
    2: "Bac Ninh",
    3: "Ha Noi",
    4: "Hai Phong",
    5: "Nam Dinh",
    6: "Thai Binh",
}

# Short-term weight is intentionally larger than long-term weight in every
# adaptive candidate. Candidate selection is development-only.
CANDIDATES = [
    {
        "name": "short_month_no_station",
        "weights": [0.10, 0.30, 0.20, 0.15, 0.35, 0.00, 0.00],
    },
    {
        "name": "short_month_station_1",
        "weights": [0.10, 0.30, 0.20, 0.10, 0.35, 0.15, 0.00],
    },
    {
        "name": "short_month_station_2",
        "weights": [0.05, 0.25, 0.20, 0.15, 0.30, 0.25, 0.00],
    },
    {
        "name": "short_station_recent",
        "weights": [0.10, 0.25, 0.15, 0.10, 0.25, 0.15, 0.20],
    },
    {
        "name": "month_heavy_station",
        "weights": [0.05, 0.20, 0.15, 0.10, 0.50, 0.15, 0.00],
    },
    {
        "name": "station_heavy_control",
        "weights": [0.10, 0.15, 0.10, 0.10, 0.15, 0.45, 0.20],
    },
]


def station_for_date(day: dt.date) -> str:
    return STATION_BY_WEEKDAY[day.weekday()]


def zscore_columns(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (x - mu) / sd


def bayes_rate(hits: np.ndarray, n: int, strength: float) -> np.ndarray:
    return (hits + strength * P0) / (n + strength)


def build_feature_tensor(draws: list[dict], presence: np.ndarray) -> np.ndarray:
    n = len(draws)
    out = np.zeros((n, 100, len(FEATURE_NAMES)), dtype=np.float64)
    prefix = np.zeros((n + 1, 100), dtype=np.float64)
    prefix[1:] = np.cumsum(presence, axis=0)

    station_hits: dict[str, np.ndarray] = {
        name: np.zeros(100, dtype=np.float64) for name in set(STATION_BY_WEEKDAY.values())
    }
    station_days: dict[str, int] = {name: 0 for name in station_hits}
    station_recent: dict[str, list[np.ndarray]] = {name: [] for name in station_hits}
    month_start = 0

    for i, draw in enumerate(draws):
        day = draw["date"]
        if i == 0 or (draws[i - 1]["date"].year, draws[i - 1]["date"].month) != (day.year, day.month):
            month_start = i

        def rolling(window: int, prior: float) -> np.ndarray:
            start = max(0, i - window)
            m = i - start
            hits = prefix[i] - prefix[start]
            return bayes_rate(hits, m, prior)

        long_start = max(0, i - LONG_LOOKBACK)
        long_n = i - long_start
        long_rate = bayes_rate(prefix[i] - prefix[long_start], long_n, 100.0)
        r7 = rolling(7, 3.0)
        r14 = rolling(14, 5.0)
        r30 = rolling(30, 10.0)

        month_n = i - month_start
        month_rate = bayes_rate(prefix[i] - prefix[month_start], month_n, 5.0)

        station = station_for_date(day)
        station_rate = bayes_rate(station_hits[station], station_days[station], 30.0)
        rec_rows = station_recent[station][-52:]
        if rec_rows:
            rec_hits = np.sum(np.stack(rec_rows, axis=0), axis=0)
            station_recent_rate = bayes_rate(rec_hits, len(rec_rows), 10.0)
        else:
            station_recent_rate = np.full(100, P0, dtype=np.float64)

        out[i, :, 0] = long_rate
        out[i, :, 1] = r7
        out[i, :, 2] = r14
        out[i, :, 3] = r30
        out[i, :, 4] = month_rate
        out[i, :, 5] = station_rate
        out[i, :, 6] = station_recent_rate

        # Reveal today's result only after today's feature vector has been locked.
        station_hits[station] += presence[i]
        station_days[station] += 1
        station_recent[station].append(presence[i].astype(np.float64))

    return out


def score_day(feature_day: np.ndarray, candidate: dict) -> np.ndarray:
    z = zscore_columns(feature_day)
    return z @ np.asarray(candidate["weights"], dtype=float)


def top3(score: np.ndarray) -> list[int]:
    return sorted(range(100), key=lambda n: (-float(score[n]), n))[:3]


def suffixes(draw: dict) -> set[int]:
    return {int(str(draw[col])[-2:]) for col in PRIZE_COLS}


def run_window(draws, presence, features, start: int, end: int, candidate: dict, collect=False):
    hits = 0
    daily = []
    changes = 0
    prev = None
    station_hits = defaultdict(lambda: [0, 0])
    for i in range(start, end):
        scores = score_day(features[i], candidate)
        picks = top3(scores)
        actual = suffixes(draws[i])
        hit_flags = [int(x in actual) for x in picks]
        any_hit = int(any(hit_flags))
        hits += any_hit
        st = station_for_date(draws[i]["date"])
        station_hits[st][0] += any_hit
        station_hits[st][1] += 1
        if prev is not None and picks != prev:
            changes += 1
        prev = picks
        if collect:
            row = {
                "date": draws[i]["date"].isoformat(),
                "station": st,
                "pick1": f"{picks[0]:02d}",
                "pick2": f"{picks[1]:02d}",
                "pick3": f"{picks[2]:02d}",
                "pick1_hit": hit_flags[0],
                "pick2_hit": hit_flags[1],
                "pick3_hit": hit_flags[2],
                "top3_any_hit": any_hit,
            }
            for j, name in enumerate(FEATURE_NAMES):
                row[f"{name}_p1"] = float(features[i, picks[0], j])
                row[f"{name}_p2"] = float(features[i, picks[1], j])
                row[f"{name}_p3"] = float(features[i, picks[2], j])
            daily.append(row)
    n = end - start
    by_station = {
        st: {"hits": h, "draws": d, "hit_rate": h / d if d else 0.0}
        for st, (h, d) in sorted(station_hits.items())
    }
    return {
        "candidate": candidate["name"],
        "n": n,
        "hits": hits,
        "hit_rate": hits / n,
        "lift_pp_vs_random": (hits / n - TOP3_BASELINE) * 100.0,
        "pick_set_changes": changes,
        "pick_set_change_rate": changes / max(n - 1, 1),
        "by_station": by_station,
    }, daily


def target_features(draws: list[dict], presence: np.ndarray, target: dt.date) -> np.ndarray:
    n = len(draws)
    prefix = np.zeros((n + 1, 100), dtype=float)
    prefix[1:] = np.cumsum(presence, axis=0)

    def range_rate(indices: list[int], strength: float) -> np.ndarray:
        if not indices:
            return np.full(100, P0, dtype=float)
        return bayes_rate(presence[indices].sum(axis=0), len(indices), strength)

    long_idx = list(range(max(0, n - LONG_LOOKBACK), n))
    r7_idx = list(range(max(0, n - 7), n))
    r14_idx = list(range(max(0, n - 14), n))
    r30_idx = list(range(max(0, n - 30), n))
    month_idx = [i for i, d in enumerate(draws) if (d["date"].year, d["date"].month) == (target.year, target.month)]
    st = station_for_date(target)
    station_idx = [i for i, d in enumerate(draws) if station_for_date(d["date"]) == st]
    station_recent_idx = station_idx[-52:]

    return np.column_stack([
        range_rate(long_idx, 100.0),
        range_rate(r7_idx, 3.0),
        range_rate(r14_idx, 5.0),
        range_rate(r30_idx, 10.0),
        range_rate(month_idx, 5.0),
        range_rate(station_idx, 30.0),
        range_rate(station_recent_idx, 10.0),
    ])


def station_diagnostics(draws: list[dict], presence: np.ndarray, end_idx: int) -> dict:
    overall = bayes_rate(presence[:end_idx].sum(axis=0), end_idx, 100.0)
    out = {}
    for st in sorted(set(STATION_BY_WEEKDAY.values())):
        idx = [i for i in range(end_idx) if station_for_date(draws[i]["date"]) == st]
        rate = bayes_rate(presence[idx].sum(axis=0), len(idx), 30.0)
        lift = rate - overall
        top = sorted(range(100), key=lambda x: (-float(lift[x]), x))[:10]
        bottom = sorted(range(100), key=lambda x: (float(lift[x]), x))[:10]
        out[st] = {
            "draws": len(idx),
            "mean_abs_number_lift_pp": float(np.mean(np.abs(lift)) * 100.0),
            "max_abs_number_lift_pp": float(np.max(np.abs(lift)) * 100.0),
            "top_positive": [
                {"number": f"{x:02d}", "station_rate": float(rate[x]), "overall_rate": float(overall[x]), "lift_pp": float(lift[x] * 100.0)}
                for x in top
            ],
            "top_negative": [
                {"number": f"{x:02d}", "station_rate": float(rate[x]), "overall_rate": float(overall[x]), "lift_pp": float(lift[x] * 100.0)}
                for x in bottom
            ],
        }
    return out


def wilson(hits: int, n: int):
    z = 1.959963984540054
    p = hits / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n))/den
    margin = z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/den
    return center-margin, center+margin


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def main():
    draws = load_draws(DATA)
    presence, _counts = matrices(draws)
    features = build_feature_tensor(draws, presence)
    n = len(draws)
    holdout_start = n - HOLDOUT_DRAWS
    dev_start = holdout_start - DEV_DRAWS
    if dev_start < 365:
        raise RuntimeError("Not enough history")

    dev_rows = []
    for cand in CANDIDATES:
        stats, _ = run_window(draws, presence, features, dev_start, holdout_start, cand, collect=False)
        dev_rows.append({
            "candidate": cand["name"],
            "hit_rate": stats["hit_rate"],
            "hits": stats["hits"],
            "lift_pp_vs_random": stats["lift_pp_vs_random"],
            "pick_set_change_rate": stats["pick_set_change_rate"],
            "weights": json.dumps(dict(zip(FEATURE_NAMES, cand["weights"])), sort_keys=True),
        })
    dev_rows.sort(key=lambda r: (-r["hit_rate"], -r["pick_set_change_rate"], r["candidate"]))
    selected_name = dev_rows[0]["candidate"]
    selected = next(c for c in CANDIDATES if c["name"] == selected_name)

    holdout_stats, daily = run_window(draws, presence, features, holdout_start, n, selected, collect=True)
    # Controls measured on holdout only after model selection is already locked.
    controls = {}
    for name in ["short_month_no_station", "short_month_station_1", "station_heavy_control"]:
        cand = next(c for c in CANDIDATES if c["name"] == name)
        controls[name], _ = run_window(draws, presence, features, holdout_start, n, cand, collect=False)

    low, high = wilson(holdout_stats["hits"], HOLDOUT_DRAWS)
    station_diag = station_diagnostics(draws, presence, holdout_start)

    target = draws[-1]["date"] + dt.timedelta(days=1)
    x_next = target_features(draws, presence, target)
    scores = score_day(x_next, selected)
    picks = top3(scores)
    next_detail = []
    for p in picks:
        next_detail.append({
            "number": f"{p:02d}",
            "score": float(scores[p]),
            **{FEATURE_NAMES[j]: float(x_next[p, j]) for j in range(len(FEATURE_NAMES))},
        })

    summary = {
        "protocol": {
            "development_draws": DEV_DRAWS,
            "final_holdout_draws": HOLDOUT_DRAWS,
            "ordering": "predict using history strictly before target -> reveal target -> next day may use result",
            "random_top3_baseline": TOP3_BASELINE,
        },
        "data": {
            "first_date": draws[0]["date"].isoformat(),
            "latest_date": draws[-1]["date"].isoformat(),
            "draws": n,
            "development_start": draws[dev_start]["date"].isoformat(),
            "development_end": draws[holdout_start-1]["date"].isoformat(),
            "holdout_start": draws[holdout_start]["date"].isoformat(),
            "holdout_end": draws[-1]["date"].isoformat(),
        },
        "selected_model": selected,
        "holdout": {
            **holdout_stats,
            "wilson_95_low": low,
            "wilson_95_high": high,
        },
        "holdout_controls": controls,
        "next_forecast": {
            "target_date": target.isoformat(),
            "station": station_for_date(target),
            "top3": [f"{p:02d}" for p in picks],
            "detail": next_detail,
        },
        "station_diagnostics_pre_holdout": station_diag,
    }

    EVAL.mkdir(parents=True, exist_ok=True)
    FORECAST.mkdir(parents=True, exist_ok=True)
    (EVAL / "walkforward_top3_v2_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    write_csv(EVAL / "walkforward_top3_v2_model_selection.csv", dev_rows)
    write_csv(EVAL / "walkforward_top3_v2_daily.csv", daily)
    (FORECAST / "top3_v2_next.json").write_text(json.dumps(summary["next_forecast"], indent=2, ensure_ascii=False)+"\n", encoding="utf-8")

    print(json.dumps({
        "selected": selected_name,
        "development_hit_rate": dev_rows[0]["hit_rate"],
        "holdout_hit_rate": holdout_stats["hit_rate"],
        "random_baseline": TOP3_BASELINE,
        "pick_change_rate": holdout_stats["pick_set_change_rate"],
        "next": summary["next_forecast"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
