#!/usr/bin/env python3
"""Leakage-safe walk-forward backtest for XSMB top-3 two-digit picks.

Protocol
--------
- Final holdout: latest 365 observed draws.
- Model/hyperparameter selection: previous 730 observed draws only.
- Each candidate is initialized using only draws before its evaluation window.
- For every evaluated day:
  1) build features from history strictly before the target day;
  2) rank 00..99 and lock exactly 3 picks;
  3) score against that day's 27 published suffixes;
  4) only then apply one online gradient update using that day's labels.
- The final holdout is never used to select or tune the candidate.

The script intentionally reports the correct random top-3 baseline:
1 - (97/100)^27 ~= 56.06%. Therefore >50% alone is not evidence of edge.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from xsmb_probability import (
    FEATURE_NAMES,
    P0,
    PRIZE_COLS,
    features_for_day,
    fit_logistic_ridge,
    load_draws,
    matrices,
    predict,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "upstream" / "xsmb.csv"
EVAL_DIR = ROOT / "evaluation"
FORECAST_DIR = ROOT / "forecasts"

HOLDOUT_DRAWS = 365
DEV_DRAWS = 730
TRAIN_DRAWS = 5 * 365
WARMUP = 90

# Pre-registered candidates. Selection is based ONLY on the development window.
FEATURE_SETS = {
    "long_only": [4],
    "recency": [0, 1, 2, 3, 4],
    "recency_gap": [0, 1, 2, 3, 4, 6, 7],
    "full": list(range(len(FEATURE_NAMES))),
}
CANDIDATES = [
    {"name": "long_l10_lr010", "feature_set": "long_only", "l2": 10.0, "lr": 0.10},
    {"name": "long_l100_lr030", "feature_set": "long_only", "l2": 100.0, "lr": 0.30},
    {"name": "recency_l100_lr010", "feature_set": "recency", "l2": 100.0, "lr": 0.10},
    {"name": "recency_l300_lr030", "feature_set": "recency", "l2": 300.0, "lr": 0.30},
    {"name": "recgap_l300_lr010", "feature_set": "recency_gap", "l2": 300.0, "lr": 0.10},
    {"name": "full_l300_lr010", "feature_set": "full", "l2": 300.0, "lr": 0.10},
    {"name": "full_l1000_lr030", "feature_set": "full", "l2": 1000.0, "lr": 0.30},
]

BASELINES = {
    1: 1.0 - 0.99**27,
    2: 1.0 - 0.98**27,
    3: 1.0 - 0.97**27,
}


def top_k(prob: np.ndarray, k: int) -> list[int]:
    return sorted(range(100), key=lambda n: (-float(prob[n]), n))[:k]


def suffix_set(draw: dict) -> set[int]:
    return {int(str(draw[col])[-2:]) for col in PRIZE_COLS}


def suffix_counter(draw: dict) -> Counter[int]:
    return Counter(int(str(draw[col])[-2:]) for col in PRIZE_COLS)


def precompute_features(
    draws: list[dict], presence: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    """Vectorized-equivalent feature tensor with strict history [0, i) only."""
    n = len(draws)
    out = np.zeros((n, 100, len(FEATURE_NAMES)), dtype=np.float64)

    p_prefix = np.zeros((n + 1, 100), dtype=np.float64)
    c_prefix = np.zeros((n + 1, 100), dtype=np.float64)
    p_prefix[1:] = np.cumsum(presence, axis=0)
    c_prefix[1:] = np.cumsum(counts, axis=0)

    weekday_hits = np.zeros((7, 100), dtype=np.float64)
    weekday_days = np.zeros(7, dtype=np.int64)
    last_hit = np.full(100, -1, dtype=np.int64)

    for i, draw in enumerate(draws):
        def rolling(prefix: np.ndarray, window: int) -> np.ndarray:
            start = max(0, i - window)
            length = i - start
            if length <= 0:
                return np.full(100, P0, dtype=np.float64)
            return (prefix[i] - prefix[start]) / length

        f7 = rolling(p_prefix, 7)
        f30 = rolling(p_prefix, 30)
        f90 = rolling(p_prefix, 90)

        start30 = max(0, i - 30)
        len30 = i - start30
        if len30:
            c30 = ((c_prefix[i] - c_prefix[start30]) / len30) / 27.0
        else:
            c30 = np.full(100, 0.01, dtype=np.float64)

        long_rate = (p_prefix[i] + 100.0 * P0) / (i + 100.0)

        wd = draw["date"].weekday()
        if weekday_days[wd] > 0:
            weekday_rate = (weekday_hits[wd] + 30.0 * P0) / (weekday_days[wd] + 30.0)
        else:
            weekday_rate = np.full(100, P0, dtype=np.float64)

        gap = np.where(last_hit >= 0, i - 1 - last_hit, i)
        gap_norm = np.minimum(gap, 30) / 30.0
        prev_hit = presence[i - 1].astype(np.float64) if i else np.zeros(100, dtype=np.float64)

        out[i, :, 0] = f7
        out[i, :, 1] = f30
        out[i, :, 2] = f90
        out[i, :, 3] = c30
        out[i, :, 4] = long_rate
        out[i, :, 5] = weekday_rate
        out[i, :, 6] = gap_norm
        out[i, :, 7] = prev_hit

        # Reveal current result only after today's features have been locked.
        weekday_hits[wd] += presence[i]
        weekday_days[wd] += 1
        hit_mask = presence[i].astype(bool)
        last_hit[hit_mask] = i

    return out


def wilson_interval(hits: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    phat = hits / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def normal_two_sided_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def longest_streak(values: Iterable[int], target: int) -> int:
    best = cur = 0
    for value in values:
        if int(value) == target:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def fit_at_cutoff(
    features: np.ndarray,
    presence: np.ndarray,
    cutoff_idx: int,
    candidate: dict,
) -> dict:
    train_start = max(WARMUP, cutoff_idx - TRAIN_DRAWS)
    if cutoff_idx - train_start < 240:
        train_start = WARMUP
    idx = FEATURE_SETS[candidate["feature_set"]]
    x_train = features[train_start:cutoff_idx, :, :][:, :, idx].reshape(-1, len(idx))
    y_train = presence[train_start:cutoff_idx].reshape(-1)
    model = fit_logistic_ridge(x_train, y_train, l2=float(candidate["l2"]))
    model["train_rows"] = int(len(y_train))
    model["feature_idx"] = idx
    model["candidate"] = dict(candidate)
    return model


def online_update(model: dict, x_day_full: np.ndarray, y_day: np.ndarray) -> None:
    idx = model["feature_idx"]
    x = x_day_full[:, idx]
    z = (x - model["mu"]) / model["sd"]
    a = np.column_stack([np.ones(len(z)), z])
    beta = model["beta"]
    eta = np.clip(a @ beta, -30, 30)
    p = 1.0 / (1.0 + np.exp(-eta))
    grad = a.T @ (p - y_day) / len(y_day)

    l2 = float(model["candidate"]["l2"])
    reg_scale = l2 / max(int(model["train_rows"]), 1)
    grad[1:] += reg_scale * beta[1:]

    beta -= float(model["candidate"]["lr"]) * grad
    model["beta"] = beta
    model["train_rows"] = int(model["train_rows"]) + len(y_day)


def predict_day(model: dict, x_day_full: np.ndarray) -> np.ndarray:
    idx = model["feature_idx"]
    return predict(model, x_day_full[:, idx])


def run_walkforward(
    draws: list[dict],
    presence: np.ndarray,
    counts: np.ndarray,
    features: np.ndarray,
    start_idx: int,
    end_idx: int,
    candidate: dict,
    collect_daily: bool,
) -> tuple[dict, list[dict], dict]:
    model = fit_at_cutoff(features, presence, start_idx, candidate)
    daily: list[dict] = []
    hit1 = hit2 = hit3 = 0
    brier_sum = 0.0

    for i in range(start_idx, end_idx):
        target_date = draws[i]["date"]
        x_day = features[i]
        prob = predict_day(model, x_day)
        picks = top_k(prob, 3)
        actual = suffix_set(draws[i])
        actual_counts = suffix_counter(draws[i])
        y_day = presence[i].astype(float)

        flags = [int(number in actual) for number in picks]
        h1 = flags[0]
        h2 = int(any(flags[:2]))
        h3 = int(any(flags[:3]))
        hit1 += h1
        hit2 += h2
        hit3 += h3
        brier_sum += float(np.mean((y_day - prob) ** 2))

        if collect_daily:
            daily.append(
                {
                    "date": target_date.isoformat(),
                    "pick1": f"{picks[0]:02d}",
                    "prob1": float(prob[picks[0]]),
                    "pick1_count": int(actual_counts.get(picks[0], 0)),
                    "pick1_hit": flags[0],
                    "pick2": f"{picks[1]:02d}",
                    "prob2": float(prob[picks[1]]),
                    "pick2_count": int(actual_counts.get(picks[1], 0)),
                    "pick2_hit": flags[1],
                    "pick3": f"{picks[2]:02d}",
                    "prob3": float(prob[picks[2]]),
                    "pick3_count": int(actual_counts.get(picks[2], 0)),
                    "pick3_hit": flags[2],
                    "top1_hit": h1,
                    "top2_any_hit": h2,
                    "top3_any_hit": h3,
                    "picked_hit_count": int(sum(flags)),
                    "actual_unique_suffixes": " ".join(f"{n:02d}" for n in sorted(actual)),
                    "candidate": candidate["name"],
                }
            )

        # Strict ordering: update only AFTER the target result has been scored.
        online_update(model, x_day, y_day)

    n = end_idx - start_idx
    stats = {
        "candidate": candidate["name"],
        "feature_set": candidate["feature_set"],
        "l2": candidate["l2"],
        "online_lr": candidate["lr"],
        "n": n,
        "top1_hits": hit1,
        "top1_hit_rate": hit1 / n,
        "top2_hits": hit2,
        "top2_hit_rate": hit2 / n,
        "top3_hits": hit3,
        "top3_hit_rate": hit3 / n,
        "mean_brier": brier_sum / n,
    }
    return stats, daily, model


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def monthly_stats(daily: list[dict]) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in daily:
        buckets[row["date"][:7]].append(row)
    out = []
    for month in sorted(buckets):
        rows = buckets[month]
        n = len(rows)
        top3_hits = sum(int(r["top3_any_hit"]) for r in rows)
        out.append(
            {
                "month": month,
                "draws": n,
                "top1_hits": sum(int(r["top1_hit"]) for r in rows),
                "top1_hit_rate": sum(int(r["top1_hit"]) for r in rows) / n,
                "top2_hits": sum(int(r["top2_any_hit"]) for r in rows),
                "top2_hit_rate": sum(int(r["top2_any_hit"]) for r in rows) / n,
                "top3_hits": top3_hits,
                "top3_hit_rate": top3_hits / n,
                "top3_lift_pp_vs_random": (top3_hits / n - BASELINES[3]) * 100.0,
            }
        )
    return out


def number_stats(daily: list[dict]) -> list[dict]:
    stats = {
        n: {"number": f"{n:02d}", "times_picked": 0, "hit_days_when_picked": 0, "total_occurrences_when_picked": 0}
        for n in range(100)
    }
    for row in daily:
        for pos in (1, 2, 3):
            n = int(row[f"pick{pos}"])
            stats[n]["times_picked"] += 1
            stats[n]["hit_days_when_picked"] += int(row[f"pick{pos}_hit"])
            stats[n]["total_occurrences_when_picked"] += int(row[f"pick{pos}_count"])
    out = []
    for n in range(100):
        item = stats[n]
        times = item["times_picked"]
        item["hit_rate_when_picked"] = item["hit_days_when_picked"] / times if times else 0.0
        out.append(item)
    out.sort(key=lambda r: (-r["times_picked"], -r["hit_rate_when_picked"], r["number"]))
    return out


def make_summary(
    draws: list[dict],
    dev_start: int,
    holdout_start: int,
    final_stats: dict,
    daily: list[dict],
    chosen: dict,
    selection_rows: list[dict],
) -> dict:
    n = final_stats["n"]
    hits = final_stats["top3_hits"]
    rate = final_stats["top3_hit_rate"]
    baseline = BASELINES[3]
    se0 = math.sqrt(baseline * (1 - baseline) / n)
    z = (rate - baseline) / se0 if se0 > 0 else 0.0
    ci_low, ci_high = wilson_interval(hits, n)
    values = [int(r["top3_any_hit"]) for r in daily]

    dev_sorted = sorted(selection_rows, key=lambda r: (-r["top3_hit_rate"], r["mean_brier"], r["candidate"]))

    return {
        "protocol": {
            "target": "top3 distinct two-digit suffix picks; hit if >=1 appears among 27 XSMB prizes",
            "final_holdout_draws": HOLDOUT_DRAWS,
            "development_draws": DEV_DRAWS,
            "training_lookback_draws": TRAIN_DRAWS,
            "ordering": "predict -> lock 3 picks -> reveal result -> score -> online update",
            "leakage_guard": "candidate/hyperparameter selection uses only the development window ending before the final holdout",
            "important_baseline_note": "50% is below the fair random top-3 baseline, so success is judged against ~56.06%, not 50%.",
        },
        "data": {
            "first_date": draws[0]["date"].isoformat(),
            "latest_date": draws[-1]["date"].isoformat(),
            "observed_draws": len(draws),
            "development_start": draws[dev_start]["date"].isoformat(),
            "development_end": draws[holdout_start - 1]["date"].isoformat(),
            "holdout_start": draws[holdout_start]["date"].isoformat(),
            "holdout_end": draws[-1]["date"].isoformat(),
        },
        "selected_model": {
            **chosen,
            "selected_by": "highest development top3 hit rate; tie-break lower development Brier; never by holdout",
            "development_rank": [row["candidate"] for row in dev_sorted],
        },
        "holdout": {
            **final_stats,
            "random_baselines": {f"top{k}": BASELINES[k] for k in sorted(BASELINES)},
            "top3_expected_random_hits": n * baseline,
            "top3_lift_pp_vs_random": (rate - baseline) * 100.0,
            "top3_relative_lift_pct": (rate / baseline - 1.0) * 100.0,
            "top3_wilson_95_low": ci_low,
            "top3_wilson_95_high": ci_high,
            "top3_z_vs_random": z,
            "top3_two_sided_p_vs_random": normal_two_sided_p(z),
            "longest_hit_streak": longest_streak(values, 1),
            "longest_miss_streak": longest_streak(values, 0),
            "beats_50pct": rate > 0.50,
            "beats_random_baseline": rate > baseline,
            "promotion_rule": "Do not claim predictive edge unless holdout beats the random baseline with practically meaningful lift and uncertainty supports it.",
        },
    }


def write_markdown_report(summary: dict, selection_rows: list[dict], monthly: list[dict], forecast: dict) -> None:
    h = summary["holdout"]
    d = summary["data"]
    m = summary["selected_model"]
    lines = [
        "# XSMB Top-3 Walk-forward Report",
        "",
        f"- Data: {d['first_date']} -> {d['latest_date']} ({d['observed_draws']} draws)",
        f"- Development: {d['development_start']} -> {d['development_end']}",
        f"- Final holdout: {d['holdout_start']} -> {d['holdout_end']} ({h['n']} draws)",
        f"- Selected model: `{m['name']}` / feature_set=`{m['feature_set']}` / L2={m['l2']} / online_lr={m['lr']}",
        "",
        "## Final holdout",
        "",
        f"- Top-1: {h['top1_hits']}/{h['n']} = {h['top1_hit_rate']:.2%} (random baseline {BASELINES[1]:.2%})",
        f"- Top-2 any-hit: {h['top2_hits']}/{h['n']} = {h['top2_hit_rate']:.2%} (random baseline {BASELINES[2]:.2%})",
        f"- Top-3 any-hit: **{h['top3_hits']}/{h['n']} = {h['top3_hit_rate']:.2%}** (random baseline {BASELINES[3]:.2%})",
        f"- Top-3 lift: {h['top3_lift_pp_vs_random']:+.2f} percentage points",
        f"- 95% Wilson CI: [{h['top3_wilson_95_low']:.2%}, {h['top3_wilson_95_high']:.2%}]",
        f"- z vs random: {h['top3_z_vs_random']:.3f}; two-sided p={h['top3_two_sided_p_vs_random']:.4f}",
        f"- Longest hit streak: {h['longest_hit_streak']} days; longest miss streak: {h['longest_miss_streak']} days",
        "",
        "Important: a 50% hit rate is below the fair top-3 baseline (~56.06%), so 50% is not a valid promotion threshold.",
        "",
        "## Development model selection",
        "",
        "| candidate | feature set | L2 | online lr | top3 hit | Brier |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(selection_rows, key=lambda r: (-r["top3_hit_rate"], r["mean_brier"], r["candidate"])):
        lines.append(
            f"| {row['candidate']} | {row['feature_set']} | {row['l2']} | {row['online_lr']} | {row['top3_hit_rate']:.2%} | {row['mean_brier']:.6f} |"
        )
    lines += [
        "",
        "## Monthly holdout",
        "",
        "| month | draws | top3 hits | hit rate | lift vs random |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in monthly:
        lines.append(
            f"| {row['month']} | {row['draws']} | {row['top3_hits']} | {row['top3_hit_rate']:.2%} | {row['top3_lift_pp_vs_random']:+.2f} pp |"
        )
    lines += [
        "",
        "## Next forecast",
        "",
        f"- Target date: {forecast['target_date']}",
        f"- Top 3: **{' - '.join(forecast['top3'])}**",
        "- This ranking is only a research output; if the holdout does not beat random baseline credibly, it should not be treated as proven edge.",
        "",
    ]
    (EVAL_DIR / "walkforward_top3_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(f"Missing {DATA}; run src/sync_upstream.py first")

    draws = load_draws(DATA)
    if len(draws) < HOLDOUT_DRAWS + DEV_DRAWS + WARMUP + 240:
        raise RuntimeError("Not enough history for dev + holdout protocol.")

    presence, counts = matrices(draws)
    print("Precomputing leakage-safe feature tensor...")
    features = precompute_features(draws, presence, counts)
    n = len(draws)
    holdout_start = n - HOLDOUT_DRAWS
    dev_start = holdout_start - DEV_DRAWS

    selection_rows: list[dict] = []
    for candidate in CANDIDATES:
        stats, _, _ = run_walkforward(
            draws, presence, counts, features, dev_start, holdout_start, candidate, collect_daily=False
        )
        selection_rows.append(stats)
        print(
            f"DEV {candidate['name']}: top3={stats['top3_hit_rate']:.2%} "
            f"brier={stats['mean_brier']:.6f}"
        )

    selection_rows.sort(key=lambda r: (-r["top3_hit_rate"], r["mean_brier"], r["candidate"]))
    winner_name = selection_rows[0]["candidate"]
    chosen = next(c for c in CANDIDATES if c["name"] == winner_name)
    print(f"Selected on DEVELOPMENT ONLY: {winner_name}")

    final_stats, daily, final_model = run_walkforward(
        draws, presence, counts, features, holdout_start, n, chosen, collect_daily=True
    )

    summary = make_summary(
        draws, dev_start, holdout_start, final_stats, daily, chosen, selection_rows
    )
    monthly = monthly_stats(daily)
    per_number = number_stats(daily)

    target_date = draws[-1]["date"] + dt.timedelta(days=1)
    x_next = features_for_day(n, target_date.weekday(), draws, presence, counts)
    p_next = predict_day(final_model, x_next)
    next_picks = top_k(p_next, 3)
    forecast = {
        "target_date": target_date.isoformat(),
        "data_cutoff": draws[-1]["date"].isoformat(),
        "candidate": chosen,
        "top3": [f"{x:02d}" for x in next_picks],
        "probabilities": {f"{x:02d}": float(p_next[x]) for x in next_picks},
        "validated_holdout_top3_hit_rate": final_stats["top3_hit_rate"],
        "random_top3_baseline": BASELINES[3],
        "validated_edge": bool(
            final_stats["top3_hit_rate"] > BASELINES[3]
            and summary["holdout"]["top3_two_sided_p_vs_random"] < 0.05
        ),
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(EVAL_DIR / "walkforward_top3_model_selection.csv", selection_rows)
    write_csv(EVAL_DIR / "walkforward_top3_daily.csv", daily)
    write_csv(EVAL_DIR / "walkforward_top3_monthly.csv", monthly)
    write_csv(EVAL_DIR / "walkforward_top3_number_stats.csv", per_number)
    (EVAL_DIR / "walkforward_top3_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (FORECAST_DIR / "top3_next.json").write_text(
        json.dumps(forecast, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown_report(summary, selection_rows, monthly, forecast)

    print(json.dumps(summary["holdout"], ensure_ascii=False, indent=2))
    print(json.dumps(forecast, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
