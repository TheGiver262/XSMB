#!/usr/bin/env python3
"""Leakage-safe XSMB top-3 V4: combo + reverse aware.

V4 changes the target from "at least one of the top 3 hits" to selecting a
3-number set whose joint historical behaviour is better for the user's strict
success proxy: at least 2 distinct numbers hit in the 27 XSMB suffixes.

Important methodology:
- V4 hyperparameters are selected ONLY on six 365-draw folds ending before
  2025-08-11.
- 2025-08-11 onward is already-burned retrospective benchmark only.
- every target date uses only history strictly before that date.
- reverse (AB <-> BA), pair co-occurrence and double-number penalties are
  candidates, never assumptions; selection can fall back to the marginal V3
  control if those additions do not help robustly.
"""
from __future__ import annotations

import csv
import datetime as dt
import itertools
import json
import statistics
from pathlib import Path

import numpy as np

from xsmb_probability import load_draws, matrices
from walkforward_top3_v2 import (
    DATA,
    FEATURE_NAMES,
    TOP3_BASELINE,
    build_feature_tensor,
    score_day,
    station_for_date,
    target_features,
)

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "evaluation"
FORECAST = ROOT / "forecasts"

BASE_WEIGHTS = [0.045, 0.225, 0.135, 0.135, 0.360, 0.050, 0.050]
BASE_MODEL = {"name": "v3_frozen_base", "weights": BASE_WEIGHTS}

DEV_CUTOFF = dt.date(2025, 8, 11)
FOLD_DRAWS = 365
N_FOLDS = 6

STRICT_RANDOM_BASELINE = 1.0 - 3.0 * (0.98 ** 27) + 2.0 * (0.97 ** 27)

SCHEMES = [
    {"name": "marginal_pool12", "pool_size": 12, "reverse_lambda": 0.0, "pair_lambda": 0.0, "double_penalty": 0.0, "reverse_pair_penalty": 0.0},
    {"name": "marginal_pool18", "pool_size": 18, "reverse_lambda": 0.0, "pair_lambda": 0.0, "double_penalty": 0.0, "reverse_pair_penalty": 0.0},
    {"name": "reverse10", "pool_size": 15, "reverse_lambda": 0.10, "pair_lambda": 0.0, "double_penalty": 0.0, "reverse_pair_penalty": 0.0},
    {"name": "reverse20", "pool_size": 15, "reverse_lambda": 0.20, "pair_lambda": 0.0, "double_penalty": 0.0, "reverse_pair_penalty": 0.0},
    {"name": "reverse30", "pool_size": 15, "reverse_lambda": 0.30, "pair_lambda": 0.0, "double_penalty": 0.0, "reverse_pair_penalty": 0.0},
    {"name": "pair20", "pool_size": 15, "reverse_lambda": 0.0, "pair_lambda": 0.20, "double_penalty": 0.0, "reverse_pair_penalty": 0.0},
    {"name": "reverse20_pair10", "pool_size": 15, "reverse_lambda": 0.20, "pair_lambda": 0.10, "double_penalty": 0.0, "reverse_pair_penalty": 0.0},
    {"name": "reverse20_pair10_double10", "pool_size": 15, "reverse_lambda": 0.20, "pair_lambda": 0.10, "double_penalty": 0.10, "reverse_pair_penalty": 0.0},
]


def reverse_number(n: int) -> int:
    return (n % 10) * 10 + (n // 10)


def is_double(n: int) -> bool:
    return n // 10 == n % 10


def candidate_pool(base_scores: np.ndarray, pool_size: int) -> list[int]:
    top = sorted(range(100), key=lambda n: (-float(base_scores[n]), n))[:pool_size]
    pool = set(top)
    for n in top:
        if not is_double(n):
            pool.add(reverse_number(n))
    return sorted(pool, key=lambda n: (-float(base_scores[n]), n))


def pair_lift_matrix(presence: np.ndarray, end_idx: int, pool: list[int]) -> np.ndarray:
    k = len(pool)
    if k < 2:
        return np.zeros((k, k), dtype=float)
    signals = np.zeros((k, k), dtype=float)
    for window, weight in ((30, 0.60), (90, 0.40)):
        start = max(0, end_idx - window)
        x = presence[start:end_idx][:, pool].astype(float)
        m = len(x)
        if m == 0:
            continue
        freq = x.mean(axis=0)
        co = (x.T @ x) / m
        lift = co - np.outer(freq, freq)
        signals += weight * lift
    vals = [signals[a, b] for a in range(k) for b in range(a + 1, k)]
    if vals:
        mu = float(np.mean(vals))
        sd = float(np.std(vals))
        if sd > 1e-12:
            signals = (signals - mu) / sd
        else:
            signals[:] = 0.0
    np.fill_diagonal(signals, 0.0)
    return signals


def combo_score(combo, base_scores, pool_index, pair_z, scheme) -> float:
    score = sum(float(base_scores[n]) for n in combo)
    rev_lam = float(scheme["reverse_lambda"])
    if rev_lam:
        score += rev_lam * sum(float(base_scores[reverse_number(n)]) for n in combo if not is_double(n))
    pair_lam = float(scheme["pair_lambda"])
    if pair_lam:
        a, b, c = combo
        score += pair_lam * (float(pair_z[pool_index[a], pool_index[b]]) + float(pair_z[pool_index[a], pool_index[c]]) + float(pair_z[pool_index[b], pool_index[c]]))
    dpen = float(scheme["double_penalty"])
    if dpen:
        score -= dpen * sum(1 for n in combo if is_double(n))
    rpen = float(scheme["reverse_pair_penalty"])
    if rpen:
        pairs = ((combo[0], combo[1]), (combo[0], combo[2]), (combo[1], combo[2]))
        score -= rpen * sum(1 for a, b in pairs if reverse_number(a) == b and a != b)
    return score


def select_combo(base_scores: np.ndarray, presence: np.ndarray, end_idx: int, scheme: dict) -> list[int]:
    pool = candidate_pool(base_scores, int(scheme["pool_size"]))
    pair_z = pair_lift_matrix(presence, end_idx, pool)
    pool_index = {n: i for i, n in enumerate(pool)}
    best_combo = None
    best_score = None
    for combo in itertools.combinations(pool, 3):
        s = combo_score(combo, base_scores, pool_index, pair_z, scheme)
        key = (s, tuple(-n for n in combo))
        if best_score is None or key > best_score:
            best_score = key
            best_combo = combo
    assert best_combo is not None
    return sorted(best_combo, key=lambda n: (-float(base_scores[n]), n))


def evaluate_window(draws, presence, counts, features, start: int, end: int, scheme: dict, collect: bool = False):
    dist = [0, 0, 0, 0]
    total_nhay = 0
    any_days = 0
    strict_days = 0
    rows = []
    for i in range(start, end):
        base_scores = score_day(features[i], BASE_MODEL)
        picks = select_combo(base_scores, presence, i, scheme)
        hit_flags = [int(presence[i, n] > 0) for n in picks]
        nhay = [int(counts[i, n]) for n in picks]
        distinct = sum(hit_flags)
        total = sum(nhay)
        dist[distinct] += 1
        any_days += int(distinct >= 1)
        strict_days += int(distinct >= 2)
        total_nhay += total
        if collect:
            rows.append({"date": draws[i]["date"].isoformat(), "station": station_for_date(draws[i]["date"]), "pick1": f"{picks[0]:02d}", "pick2": f"{picks[1]:02d}", "pick3": f"{picks[2]:02d}", "pick1_nhay": nhay[0], "pick2_nhay": nhay[1], "pick3_nhay": nhay[2], "distinct_hits": distinct, "total_nhay": total, "strict_2plus": int(distinct >= 2)})
    n = end - start
    return {"scheme": scheme["name"], "n": n, "zero_of_3": dist[0], "one_of_3": dist[1], "two_of_3": dist[2], "three_of_3": dist[3], "any_hit_days": any_days, "any_hit_rate": any_days / n, "strict_2plus_days": strict_days, "strict_2plus_rate": strict_days / n, "strict_lift_pp_vs_random": (strict_days / n - STRICT_RANDOM_BASELINE) * 100.0, "total_nhay": total_nhay, "mean_total_nhay": total_nhay / n}, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def main() -> None:
    draws = load_draws(DATA)
    presence, counts = matrices(draws)
    features = build_feature_tensor(draws, presence)
    n = len(draws)
    cutoff_idx = next((i for i, d in enumerate(draws) if d["date"] >= DEV_CUTOFF), n)
    first_fold_start = cutoff_idx - N_FOLDS * FOLD_DRAWS
    if first_fold_start < 5 * 365:
        raise RuntimeError("Not enough prior history for six V4 selection folds")
    fold_ranges = [(first_fold_start + f * FOLD_DRAWS, first_fold_start + (f + 1) * FOLD_DRAWS) for f in range(N_FOLDS)]

    fold_rows = []
    scheme_summaries = []
    for scheme in SCHEMES:
        strict_rates = []
        nhay_rates = []
        for fold_no, (start, end) in enumerate(fold_ranges, start=1):
            stats, _ = evaluate_window(draws, presence, counts, features, start, end, scheme)
            strict_rates.append(stats["strict_2plus_rate"]); nhay_rates.append(stats["mean_total_nhay"])
            fold_rows.append({"scheme": scheme["name"], "fold": fold_no, "start": draws[start]["date"].isoformat(), "end": draws[end - 1]["date"].isoformat(), **{k: v for k, v in stats.items() if k not in {"scheme", "n"}}, "n": stats["n"]})
        median_rate = statistics.median(strict_rates)
        mean_rate = statistics.mean(strict_rates)
        worst_rate = min(strict_rates)
        mean_nhay = statistics.mean(nhay_rates)
        robust_score = median_rate + 0.50 * worst_rate
        scheme_summaries.append({"scheme": scheme["name"], "median_strict_2plus_rate": median_rate, "mean_strict_2plus_rate": mean_rate, "worst_fold_strict_2plus_rate": worst_rate, "mean_total_nhay": mean_nhay, "robust_score": robust_score, "strict_random_baseline": STRICT_RANDOM_BASELINE, "params": json.dumps(scheme, sort_keys=True)})

    scheme_summaries.sort(key=lambda r: (-float(r["robust_score"]), -float(r["median_strict_2plus_rate"]), -float(r["worst_fold_strict_2plus_rate"]), -float(r["mean_strict_2plus_rate"]), -float(r["mean_total_nhay"]), r["scheme"]))
    selected = next(s for s in SCHEMES if s["name"] == scheme_summaries[0]["scheme"])
    marginal = next(s for s in SCHEMES if s["name"] == "marginal_pool12")

    burned_selected, burned_daily = evaluate_window(draws, presence, counts, features, cutoff_idx, n, selected, collect=True)
    burned_marginal, _ = evaluate_window(draws, presence, counts, features, cutoff_idx, n, marginal, collect=False)

    target = draws[-1]["date"] + dt.timedelta(days=1)
    x_next = target_features(draws, presence, target)
    next_base_scores = score_day(x_next, BASE_MODEL)
    next_picks = select_combo(next_base_scores, presence, n, selected)
    reverse_details = []
    for num in next_picks:
        rev = reverse_number(num)
        reverse_details.append({"number": f"{num:02d}", "base_score": float(next_base_scores[num]), "is_double": is_double(num), "reverse": None if is_double(num) else f"{rev:02d}", "reverse_base_score": None if is_double(num) else float(next_base_scores[rev])})

    forecast = {"target_date": target.isoformat(), "station": station_for_date(target), "model": "V4_combo_reverse", "selected_scheme": selected, "top3": [f"{x:02d}" for x in next_picks], "detail": reverse_details, "data_cutoff": draws[-1]["date"].isoformat(), "prospective_status": "valid_if_generated_before_draw_time"}
    summary = {"protocol": {"selection_cutoff_exclusive": DEV_CUTOFF.isoformat(), "selection_folds": N_FOLDS, "fold_draws": FOLD_DRAWS, "selection_metric": "robust_score = median(strict_2plus_rate) + 0.5 * worst_fold_rate", "strict_success": "at least 2 of 3 distinct selected numbers appear in the 27 suffixes", "one_of_3_is_loss": True, "strict_random_baseline": STRICT_RANDOM_BASELINE, "any_hit_random_baseline": TOP3_BASELINE, "burned_benchmark_warning": "2025-08-11 onward has already been inspected in V1/V2/V3; report only, never select V4 on it", "leading_zero_policy": "all numbers serialized as two digits; 0 means 00"}, "data": {"first_date": draws[0]["date"].isoformat(), "latest_date": draws[-1]["date"].isoformat(), "draws": n, "selection_first_date": draws[first_fold_start]["date"].isoformat(), "selection_last_date": draws[cutoff_idx - 1]["date"].isoformat(), "burned_benchmark_start": draws[cutoff_idx]["date"].isoformat(), "burned_benchmark_end": draws[-1]["date"].isoformat()}, "selected_scheme": selected, "candidate_ranking": scheme_summaries, "burned_benchmark_selected": burned_selected, "burned_benchmark_marginal_control": burned_marginal, "next_forecast": forecast}

    write_csv(EVAL / "walkforward_top3_v4_candidate_summary.csv", scheme_summaries)
    write_csv(EVAL / "walkforward_top3_v4_fold_results.csv", fold_rows)
    write_csv(EVAL / "walkforward_top3_v4_burned_daily.csv", burned_daily)
    (EVAL / "walkforward_top3_v4_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (FORECAST / "top3_v4_next.json").write_text(json.dumps(forecast, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"selected_scheme": selected["name"], "selection_median_2plus": scheme_summaries[0]["median_strict_2plus_rate"], "selection_worst_2plus": scheme_summaries[0]["worst_fold_strict_2plus_rate"], "burned_2plus": burned_selected["strict_2plus_rate"], "marginal_burned_2plus": burned_marginal["strict_2plus_rate"], "next": forecast["top3"], "target": forecast["target_date"]}, indent=2))


if __name__ == "__main__":
    main()
