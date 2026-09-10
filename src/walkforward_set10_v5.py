#!/usr/bin/env python3
"""Leakage-safe XSMB V5 set-coverage research.

V5 changes the operational target from top-3 / >=2 distinct hits to:

    choose exactly 10 distinct suffixes from 00..99
    strict success = at least 3 distinct selected suffixes appear in the
    27 published XSMB two-digit suffix positions.

This file is research/development code. Historical walk-forward results are not
independent prospective confirmation. A V5 forecast becomes prospective only
when it is generated and locked before the target draw.

Design principles:
- reuse the frozen V3/V4 marginal score as an individual-signal comparator;
- optimize the 10-number set rather than ten independent probabilities;
- pair and regime-conditioned reverse terms are challengers only;
- interaction terms are enabled only when pre-cutoff folds support their
  ablation versus the simpler parent;
- doubles are never penalized; reverse signal for a double is exactly zero;
- every feature for target index i uses outcomes strictly before i.
"""
from __future__ import annotations

import csv
import datetime as dt
import itertools
import json
import math
import statistics
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from xsmb_probability import load_draws, matrices
from walkforward_top3_v2 import (
    DATA,
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

SET_SIZE = 10
STRICT_MIN_DISTINCT = 3
DEV_CUTOFF = dt.date(2025, 8, 11)
FOLD_DRAWS = 365
N_FOLDS = 6
POOL_TOP_N = 20

SCHEMES = [
    {
        "name": "marginal_top10",
        "pair_lambda": 0.0,
        "reverse_lambda": 0.0,
        "requires": [],
    },
    {
        "name": "pair_005",
        "pair_lambda": 0.05,
        "reverse_lambda": 0.0,
        "requires": ["pair"],
    },
    {
        "name": "regime_reverse_005",
        "pair_lambda": 0.0,
        "reverse_lambda": 0.05,
        "requires": ["reverse"],
    },
    {
        "name": "pair_005_regime_reverse_005",
        "pair_lambda": 0.05,
        "reverse_lambda": 0.05,
        "requires": ["pair", "reverse"],
    },
]


def reverse_number(n: int) -> int:
    return (n % 10) * 10 + (n // 10)


def is_double(n: int) -> bool:
    return n // 10 == n % 10


def random_distinct_distribution(
    set_size: int = SET_SIZE, positions: int = 27, universe: int = 100
) -> list[float]:
    """Exact IID-null distribution for # distinct selected labels observed.

    Exactly k of the selected labels appear at least once; all non-selected
    labels are unrestricted. Inclusion-exclusion handles repeated suffixes.
    """
    outside = universe - set_size
    out = []
    for k in range(set_size + 1):
        p = math.comb(set_size, k) * sum(
            ((-1) ** j)
            * math.comb(k, j)
            * ((outside + k - j) / universe) ** positions
            for j in range(k + 1)
        )
        out.append(float(p))
    return out


RANDOM_DIST = random_distinct_distribution()
STRICT_RANDOM_BASELINE = float(sum(RANDOM_DIST[STRICT_MIN_DISTINCT:]))
ANY_RANDOM_BASELINE = 1.0 - ((100 - SET_SIZE) / 100.0) ** 27
DB_G1_RANDOM_BASELINE = 1.0 - ((100 - SET_SIZE) / 100.0) ** 2
EXPECTED_RANDOM_NHAY = 27.0 * SET_SIZE / 100.0

# Protect the analytical baseline from accidental changes.
assert abs(sum(RANDOM_DIST) - 1.0) < 1e-9
assert abs(STRICT_RANDOM_BASELINE - 0.43895690001767923) < 1e-10


def candidate_pool(base_scores: np.ndarray) -> list[int]:
    top = sorted(range(100), key=lambda n: (-float(base_scores[n]), n))[:POOL_TOP_N]
    pool = set(top)
    for n in top:
        if not is_double(n):
            pool.add(reverse_number(n))
    return sorted(pool, key=lambda n: (-float(base_scores[n]), n))


def _standardize_offdiag(mat: np.ndarray) -> np.ndarray:
    if len(mat) < 2:
        return np.zeros_like(mat, dtype=float)
    mask = ~np.eye(len(mat), dtype=bool)
    vals = mat[mask]
    mu = float(np.mean(vals))
    sd = float(np.std(vals))
    if sd <= 1e-12:
        return np.zeros_like(mat, dtype=float)
    z = (mat - mu) / sd
    np.fill_diagonal(z, 0.0)
    return z


def pair_lift_matrix(presence: np.ndarray, end_idx: int, pool: list[int]) -> np.ndarray:
    """Past-only same-day pair residual, shrunk and multi-horizon."""
    k = len(pool)
    signal = np.zeros((k, k), dtype=float)
    for window, weight, shrink in ((90, 0.60, 90.0), (365, 0.40, 180.0)):
        start = max(0, end_idx - window)
        x = presence[start:end_idx][:, pool].astype(float)
        m = len(x)
        if m == 0:
            continue
        freq = x.mean(axis=0)
        co = (x.T @ x) / m
        residual = co - np.outer(freq, freq)
        residual *= m / (m + shrink)
        signal += weight * residual
    return _standardize_offdiag(signal)


def build_regime_series(presence: np.ndarray) -> np.ndarray:
    """Binary recurrence regime for each target day, based only on prior draws.

    recurrence[j] describes transition (j-1 -> j). regime[i] compares the mean
    recurrence of the last 7 known transitions with the last 90 known
    transitions, both ending at i-1. Therefore regime[i] never uses outcome i.
    """
    n = len(presence)
    recurrence = np.zeros(n, dtype=float)
    for j in range(1, n):
        prev = presence[j - 1] > 0
        cur = presence[j] > 0
        denom = max(int(prev.sum()), 1)
        recurrence[j] = float(np.logical_and(prev, cur).sum()) / denom

    regime = np.zeros(n + 1, dtype=np.int8)
    for i in range(2, n + 1):
        s7 = max(1, i - 7)
        s90 = max(1, i - 90)
        short = recurrence[s7:i].mean() if i > s7 else 0.0
        long = recurrence[s90:i].mean() if i > s90 else short
        regime[i] = int(short >= long)
    return regime


def regime_reverse_signal(
    presence: np.ndarray, regime: np.ndarray, end_idx: int, pool: list[int]
) -> np.ndarray:
    """Directional AB(t-1) -> BA(t) residual inside the current past-only regime."""
    out = np.zeros(len(pool), dtype=float)
    if end_idx < 2:
        return out
    target_regime = int(regime[end_idx])
    start = max(1, end_idx - 365)
    js = [j for j in range(start, end_idx) if int(regime[j]) == target_regime]
    if not js:
        return out

    idx = np.asarray(js, dtype=int)
    for p, n in enumerate(pool):
        if is_double(n):
            out[p] = 0.0
            continue
        rev = reverse_number(n)
        antecedent = presence[idx - 1, n].astype(float)
        target = presence[idx, rev].astype(float)
        exposure = float(antecedent.sum())
        marginal = float(target.mean())
        successes = float((antecedent * target).sum())
        # Shrink conditional successor rate to same-regime marginal.
        cond = (successes + 20.0 * marginal) / (exposure + 20.0)
        out[p] = cond - marginal

    non_double = np.asarray([not is_double(n) for n in pool], dtype=bool)
    vals = out[non_double]
    if len(vals):
        mu = float(vals.mean())
        sd = float(vals.std())
        if sd > 1e-12:
            out[non_double] = (vals - mu) / sd
        else:
            out[:] = 0.0
    out[~non_double] = 0.0
    return out


def set_objective(
    selected: tuple[int, ...] | list[int],
    base_scores: np.ndarray,
    pool_index: dict[int, int],
    pair_z: np.ndarray,
    reverse_z: np.ndarray,
    scheme: dict,
) -> float:
    selected = tuple(selected)
    score = sum(float(base_scores[n]) for n in selected)

    pair_lambda = float(scheme["pair_lambda"])
    if pair_lambda and len(selected) >= 2:
        score += pair_lambda * sum(
            float(pair_z[pool_index[a], pool_index[b]])
            for a, b in itertools.combinations(selected, 2)
        )

    reverse_lambda = float(scheme["reverse_lambda"])
    if reverse_lambda:
        score += reverse_lambda * sum(
            float(reverse_z[pool_index[n]]) for n in selected
        )
    return float(score)


def build_set_context(
    base_scores: np.ndarray,
    presence: np.ndarray,
    regime: np.ndarray,
    end_idx: int,
) -> tuple[list[int], dict[int, int], np.ndarray, np.ndarray]:
    pool = candidate_pool(base_scores)
    pool_index = {n: i for i, n in enumerate(pool)}
    pair_z = pair_lift_matrix(presence, end_idx, pool)
    reverse_z = regime_reverse_signal(presence, regime, end_idx, pool)
    return pool, pool_index, pair_z, reverse_z


def select_set_from_context(
    base_scores: np.ndarray,
    pool: list[int],
    pool_index: dict[int, int],
    pair_z: np.ndarray,
    reverse_z: np.ndarray,
    scheme: dict,
) -> list[int]:
    # Additive control is exactly the marginal top-10.
    if not float(scheme["pair_lambda"]) and not float(scheme["reverse_lambda"]):
        return sorted(pool, key=lambda n: (-float(base_scores[n]), n))[:SET_SIZE]

    selected: list[int] = []
    remaining = set(pool)
    while len(selected) < SET_SIZE:
        best = None
        best_key = None
        for n in sorted(remaining):
            trial = selected + [n]
            obj = set_objective(trial, base_scores, pool_index, pair_z, reverse_z, scheme)
            key = (obj, -n)
            if best_key is None or key > best_key:
                best_key = key
                best = n
        assert best is not None
        selected.append(best)
        remaining.remove(best)

    # Deterministic 1-swap local improvement.
    improved = True
    while improved:
        improved = False
        current_obj = set_objective(
            selected, base_scores, pool_index, pair_z, reverse_z, scheme
        )
        best_trial = None
        best_key = (current_obj, tuple(-n for n in sorted(selected)))
        selected_set = set(selected)
        for old in sorted(selected):
            for new in pool:
                if new in selected_set:
                    continue
                trial = [n for n in selected if n != old] + [new]
                obj = set_objective(
                    trial, base_scores, pool_index, pair_z, reverse_z, scheme
                )
                key = (obj, tuple(-n for n in sorted(trial)))
                if key > best_key:
                    best_key = key
                    best_trial = trial
        if best_trial is not None and best_key[0] > current_obj + 1e-12:
            selected = best_trial
            improved = True

    return sorted(selected, key=lambda n: (-float(base_scores[n]), n))


def select_set(
    base_scores: np.ndarray,
    presence: np.ndarray,
    regime: np.ndarray,
    end_idx: int,
    scheme: dict,
) -> list[int]:
    context = build_set_context(base_scores, presence, regime, end_idx)
    return select_set_from_context(base_scores, *context, scheme)


def init_stats(scheme: dict) -> dict:
    return {
        "scheme": scheme["name"],
        "n": 0,
        "dist": [0] * (SET_SIZE + 1),
        "strict_days": 0,
        "any_days": 0,
        "total_nhay": 0,
        "db_g1_days": 0,
        "rows": [],
    }


def finalize_stats(acc: dict) -> dict:
    n = int(acc["n"])
    strict_rate = acc["strict_days"] / n
    any_rate = acc["any_days"] / n
    db_g1_rate = acc["db_g1_days"] / n
    out = {
        "scheme": acc["scheme"],
        "n": n,
        **{f"{k}_of_10": int(acc["dist"][k]) for k in range(SET_SIZE + 1)},
        "strict_3plus_days": int(acc["strict_days"]),
        "strict_3plus_rate": strict_rate,
        "strict_lift_pp_vs_random": (strict_rate - STRICT_RANDOM_BASELINE) * 100.0,
        "any_hit_days": int(acc["any_days"]),
        "any_hit_rate": any_rate,
        "total_nhay": int(acc["total_nhay"]),
        "mean_total_nhay": acc["total_nhay"] / n,
        "db_g1_covered_days": int(acc["db_g1_days"]),
        "db_g1_coverage_rate": db_g1_rate,
        "db_g1_lift_pp_vs_random": (db_g1_rate - DB_G1_RANDOM_BASELINE) * 100.0,
    }
    return out


def evaluate_schemes_window(
    draws,
    presence: np.ndarray,
    counts: np.ndarray,
    features: np.ndarray,
    regime: np.ndarray,
    start: int,
    end: int,
    schemes: list[dict],
    collect_scheme: str | None = None,
) -> tuple[dict[str, dict], list[dict]]:
    accs = {s["name"]: init_stats(s) for s in schemes}
    collected: list[dict] = []

    for i in range(start, end):
        base_scores = score_day(features[i], BASE_MODEL)
        context = build_set_context(base_scores, presence, regime, i)
        for scheme in schemes:
            picks = select_set_from_context(base_scores, *context, scheme)
            nhay = [int(counts[i, n]) for n in picks]
            distinct = sum(int(x > 0) for x in nhay)
            total = sum(nhay)
            actual_db = int(draws[i]["special"][-2:])
            actual_g1 = int(draws[i]["prize1"][-2:])
            db_g1 = int(actual_db in picks or actual_g1 in picks)

            acc = accs[scheme["name"]]
            acc["n"] += 1
            acc["dist"][distinct] += 1
            acc["strict_days"] += int(distinct >= STRICT_MIN_DISTINCT)
            acc["any_days"] += int(distinct >= 1)
            acc["total_nhay"] += total
            acc["db_g1_days"] += db_g1

            if collect_scheme == scheme["name"]:
                row = {
                    "date": draws[i]["date"].isoformat(),
                    "station": station_for_date(draws[i]["date"]),
                    **{f"pick{j+1}": f"{n:02d}" for j, n in enumerate(picks)},
                    **{f"pick{j+1}_nhay": nhay[j] for j in range(SET_SIZE)},
                    "distinct_hits": distinct,
                    "total_nhay": total,
                    "strict_3plus": int(distinct >= STRICT_MIN_DISTINCT),
                    "db": f"{actual_db:02d}",
                    "g1": f"{actual_g1:02d}",
                    "db_g1_covered": db_g1,
                }
                collected.append(row)

    return {name: finalize_stats(acc) for name, acc in accs.items()}, collected


def component_support(fold_rates: dict[str, list[float]]) -> dict[str, dict]:
    tests = {
        "pair": ("pair_005", "marginal_top10"),
        "reverse": ("regime_reverse_005", "marginal_top10"),
    }
    out = {}
    for component, (challenger, parent) in tests.items():
        deltas = [
            a - b for a, b in zip(fold_rates[challenger], fold_rates[parent])
        ]
        out[component] = {
            "challenger": challenger,
            "parent": parent,
            "positive_folds": sum(int(x > 0) for x in deltas),
            "median_delta_pp": statistics.median(deltas) * 100.0,
            "mean_delta_pp": statistics.mean(deltas) * 100.0,
            "supported": bool(
                sum(int(x > 0) for x in deltas) >= 4
                and statistics.median(deltas) > 0.0
            ),
        }
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    draws = load_draws(DATA)
    presence, counts = matrices(draws)
    features = build_feature_tensor(draws, presence)
    regime = build_regime_series(presence)
    n = len(draws)

    cutoff_idx = next(
        (i for i, draw in enumerate(draws) if draw["date"] >= DEV_CUTOFF), n
    )
    first_fold_start = cutoff_idx - N_FOLDS * FOLD_DRAWS
    if first_fold_start < 5 * 365:
        raise RuntimeError("Not enough prior history for six V5 development folds")
    fold_ranges = [
        (
            first_fold_start + fold * FOLD_DRAWS,
            first_fold_start + (fold + 1) * FOLD_DRAWS,
        )
        for fold in range(N_FOLDS)
    ]

    fold_rows: list[dict] = []
    fold_rates = {scheme["name"]: [] for scheme in SCHEMES}
    fold_nhay = {scheme["name"]: [] for scheme in SCHEMES}

    for fold_no, (start, end) in enumerate(fold_ranges, start=1):
        stats_by_scheme, _ = evaluate_schemes_window(
            draws, presence, counts, features, regime, start, end, SCHEMES
        )
        for scheme in SCHEMES:
            stats = stats_by_scheme[scheme["name"]]
            fold_rates[scheme["name"]].append(stats["strict_3plus_rate"])
            fold_nhay[scheme["name"]].append(stats["mean_total_nhay"])
            fold_rows.append(
                {
                    "scheme": scheme["name"],
                    "fold": fold_no,
                    "start": draws[start]["date"].isoformat(),
                    "end": draws[end - 1]["date"].isoformat(),
                    **{k: v for k, v in stats.items() if k not in {"scheme", "n"}},
                    "n": stats["n"],
                }
            )

    support = component_support(fold_rates)

    scheme_summaries = []
    for scheme in SCHEMES:
        rates = fold_rates[scheme["name"]]
        eligible = all(support[c]["supported"] for c in scheme["requires"])
        median_rate = statistics.median(rates)
        mean_rate = statistics.mean(rates)
        worst_rate = min(rates)
        robust_score = median_rate + 0.50 * worst_rate
        scheme_summaries.append(
            {
                "scheme": scheme["name"],
                "eligible_by_ablation": eligible,
                "median_strict_3plus_rate": median_rate,
                "mean_strict_3plus_rate": mean_rate,
                "worst_fold_strict_3plus_rate": worst_rate,
                "mean_total_nhay": statistics.mean(fold_nhay[scheme["name"]]),
                "robust_score": robust_score,
                "strict_random_baseline": STRICT_RANDOM_BASELINE,
                "params": json.dumps(scheme, sort_keys=True),
            }
        )

    eligible_rows = [x for x in scheme_summaries if x["eligible_by_ablation"]]
    eligible_rows.sort(
        key=lambda r: (
            -float(r["robust_score"]),
            -float(r["median_strict_3plus_rate"]),
            -float(r["worst_fold_strict_3plus_rate"]),
            -float(r["mean_strict_3plus_rate"]),
            r["scheme"],
        )
    )
    selected_name = eligible_rows[0]["scheme"]
    selected = next(s for s in SCHEMES if s["name"] == selected_name)
    marginal = next(s for s in SCHEMES if s["name"] == "marginal_top10")

    # Burned period is report-only. V5 was designed after these outcomes existed.
    burned_stats_all, burned_daily = evaluate_schemes_window(
        draws,
        presence,
        counts,
        features,
        regime,
        cutoff_idx,
        n,
        [selected, marginal] if selected_name != marginal["name"] else [selected],
        collect_scheme=selected_name,
    )
    burned_selected = burned_stats_all[selected_name]
    burned_marginal = burned_stats_all[marginal["name"]]

    target = draws[-1]["date"] + dt.timedelta(days=1)
    now_ict = dt.datetime.now(ZoneInfo("Asia/Bangkok"))
    lock_deadline = dt.datetime.combine(
        target, dt.time(18, 0), tzinfo=ZoneInfo("Asia/Bangkok")
    )
    if now_ict >= lock_deadline:
        raise RuntimeError(
            "Refusing to create V5 forecast after target draw lock deadline; "
            "refresh upstream data first"
        )
    x_next = target_features(draws, presence, target)
    next_base_scores = score_day(x_next, BASE_MODEL)
    next_picks = select_set(next_base_scores, presence, regime, n, selected)

    detail = []
    pool = candidate_pool(next_base_scores)
    reverse_z = regime_reverse_signal(presence, regime, n, pool)
    pool_index = {num: i for i, num in enumerate(pool)}
    for num in next_picks:
        rev = reverse_number(num)
        detail.append(
            {
                "number": f"{num:02d}",
                "base_score": float(next_base_scores[num]),
                "is_double": is_double(num),
                "reverse": None if is_double(num) else f"{rev:02d}",
                "reverse_base_score": None
                if is_double(num)
                else float(next_base_scores[rev]),
                "regime_reverse_z": 0.0
                if is_double(num)
                else float(reverse_z[pool_index[num]]),
            }
        )

    forecast = {
        "target_date": target.isoformat(),
        "station": station_for_date(target),
        "model": "V5_set10_3plus",
        "selected_scheme": selected,
        "top10": [f"{x:02d}" for x in next_picks],
        "detail": detail,
        "data_cutoff": draws[-1]["date"].isoformat(),
        "generated_at_ict": now_ict.isoformat(),
        "lock_deadline_ict": lock_deadline.isoformat(),
        "strict_success_rule": "at least 3 distinct selected suffixes appear among 27 positions",
        "prospective_status": "valid_only_if_locked_before_target_draw",
    }

    summary = {
        "protocol": {
            "set_size": SET_SIZE,
            "strict_min_distinct": STRICT_MIN_DISTINCT,
            "strict_random_baseline": STRICT_RANDOM_BASELINE,
            "random_distinct_distribution_0_to_10": RANDOM_DIST,
            "any_hit_random_baseline": ANY_RANDOM_BASELINE,
            "db_g1_random_baseline": DB_G1_RANDOM_BASELINE,
            "expected_random_total_nhay": EXPECTED_RANDOM_NHAY,
            "selection_cutoff_exclusive": DEV_CUTOFF.isoformat(),
            "selection_folds": N_FOLDS,
            "fold_draws": FOLD_DRAWS,
            "selection_metric": "robust_score = median(strict_3plus_rate) + 0.5 * worst_fold_rate",
            "component_gate": "interaction component requires >=4/6 positive ablation folds and positive median delta",
            "leading_zero_policy": "serialize every suffix with two digits; integer 0 is 00",
            "double_policy": "no penalty/bonus; reverse term is zero because reverse(double)=self",
            "triple_interaction_status": "not activated in V5.0; pair/reverse must first survive ablation before expanding search complexity",
            "burned_benchmark_warning": "2025-08-11 onward existed before V5 design; report only, never use as independent confirmation",
        },
        "data": {
            "first_date": draws[0]["date"].isoformat(),
            "latest_date": draws[-1]["date"].isoformat(),
            "draws": n,
            "selection_first_date": draws[first_fold_start]["date"].isoformat(),
            "selection_last_date": draws[cutoff_idx - 1]["date"].isoformat(),
            "burned_benchmark_start": draws[cutoff_idx]["date"].isoformat(),
            "burned_benchmark_end": draws[-1]["date"].isoformat(),
        },
        "component_support": support,
        "candidate_ranking": sorted(
            scheme_summaries,
            key=lambda r: (
                not bool(r["eligible_by_ablation"]),
                -float(r["robust_score"]),
                r["scheme"],
            ),
        ),
        "selected_scheme": selected,
        "burned_benchmark_selected": burned_selected,
        "burned_benchmark_marginal_control": burned_marginal,
        "next_forecast": forecast,
    }

    write_csv(EVAL / "walkforward_set10_v5_candidate_summary.csv", summary["candidate_ranking"])
    write_csv(EVAL / "walkforward_set10_v5_fold_results.csv", fold_rows)
    write_csv(EVAL / "walkforward_set10_v5_burned_daily.csv", burned_daily)
    (EVAL / "walkforward_set10_v5_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (FORECAST / "set10_v5_next.json").write_text(
        json.dumps(forecast, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "selected_scheme": selected_name,
                "component_support": support,
                "selection_median_3plus": next(
                    x["median_strict_3plus_rate"]
                    for x in scheme_summaries
                    if x["scheme"] == selected_name
                ),
                "strict_random_baseline": STRICT_RANDOM_BASELINE,
                "burned_3plus": burned_selected["strict_3plus_rate"],
                "marginal_burned_3plus": burned_marginal["strict_3plus_rate"],
                "next": forecast["top10"],
                "target": forecast["target_date"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
