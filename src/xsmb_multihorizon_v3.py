#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np

from xsmb_probability import P0, brier, load_draws, log_loss, matrices

ROOT = Path(__file__).resolve().parents[1]

HORIZONS = {
    "w90": 90,
    "w365": 365,
    "w1095": 1095,
    "w1825": 1825,
    "w3650": 3650,
}
HALF_LIVES = {
    "d365": 365.0,
    "d1095": 1095.0,
    "d1825": 1825.0,
}

RECIPES = {
    "w1y_s100": (100.0, [("w365", 1.0)]),
    "w1y_s300": (300.0, [("w365", 1.0)]),
    "w3y_s100": (100.0, [("w1095", 1.0)]),
    "w3y_s300": (300.0, [("w1095", 1.0)]),
    "w3y_s1000": (1000.0, [("w1095", 1.0)]),
    "w5y_s100": (100.0, [("w1825", 1.0)]),
    "w5y_s300": (300.0, [("w1825", 1.0)]),
    "w5y_s1000": (1000.0, [("w1825", 1.0)]),
    "w10y_s100": (100.0, [("w3650", 1.0)]),
    "w10y_s300": (300.0, [("w3650", 1.0)]),
    "full_s100": (100.0, [("full", 1.0)]),
    "full_s300": (300.0, [("full", 1.0)]),
    "decay1y_s300": (300.0, [("d365", 1.0)]),
    "decay3y_s300": (300.0, [("d1095", 1.0)]),
    "decay5y_s300": (300.0, [("d1825", 1.0)]),
    "multi_3_5_full_s300": (300.0, [("w1095", 0.45), ("w1825", 0.35), ("full", 0.20)]),
    "multi_1_3_5_10_full_s300": (300.0, [("w365", 0.10), ("w1095", 0.30), ("w1825", 0.25), ("w3650", 0.20), ("full", 0.15)]),
    "decay_mix_s300": (300.0, [("d365", 0.20), ("d1095", 0.50), ("d1825", 0.30)]),
    "hybrid_s300": (300.0, [("w1095", 0.25), ("w1825", 0.20), ("w3650", 0.15), ("full", 0.10), ("d365", 0.10), ("d1095", 0.20)]),
}


def prefix_hits(presence: np.ndarray) -> np.ndarray:
    return np.vstack([
        np.zeros((1, presence.shape[1]), dtype=np.int32),
        np.cumsum(presence, axis=0, dtype=np.int32),
    ])


def build_decay_states(presence: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    n, k = presence.shape
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, half_life in HALF_LIVES.items():
        lam = math.exp(math.log(0.5) / half_life)
        num = np.zeros(k, dtype=float)
        den = 0.0
        nums = np.zeros((n + 1, k), dtype=np.float32)
        dens = np.zeros(n + 1, dtype=np.float32)
        for i in range(n):
            nums[i] = num
            dens[i] = den
            num = lam * num + presence[i]
            den = lam * den + 1.0
        nums[n] = num
        dens[n] = den
        out[name] = (nums, dens)
    return out


def component_probability(component, i, strength, prefix, decay_states):
    if component == "full":
        hits = prefix[i]
        exposure = i
    elif component in HORIZONS:
        h = HORIZONS[component]
        start = max(0, i - h)
        hits = prefix[i] - prefix[start]
        exposure = i - start
    elif component in HALF_LIVES:
        nums, dens = decay_states[component]
        hits = nums[i]
        exposure = float(dens[i])
    else:
        raise KeyError(component)
    if exposure <= 0:
        return np.full(100, P0, dtype=float)
    return (hits.astype(float) + strength * P0) / (float(exposure) + strength)


def recipe_probability(recipe_name, i, prefix, decay_states):
    strength, components = RECIPES[recipe_name]
    p = np.zeros(100, dtype=float)
    for component, weight in components:
        p += weight * component_probability(component, i, strength, prefix, decay_states)
    return np.clip(p, 1e-6, 1 - 1e-6)


def build_forecast_cube(draws, presence, warmup=365):
    n = len(draws)
    names = list(RECIPES)
    cube = np.full((n + 1, len(names), 100), np.nan, dtype=np.float32)
    prefix = prefix_hits(presence)
    decay_states = build_decay_states(presence)
    for i in range(warmup, n + 1):
        for j, name in enumerate(names):
            cube[i, j] = recipe_probability(name, i, prefix, decay_states)
    return names, cube


def flatten_period(cube, presence, idx, recipe_idx):
    return (
        presence[idx].astype(float).reshape(-1),
        cube[idx, recipe_idx].astype(float).reshape(-1),
    )


def choose_recipe_blend(names, cube, presence, selection_idx):
    best = None
    for j, name in enumerate(names):
        y, raw = flatten_period(cube, presence, selection_idx, j)
        for blend in np.linspace(0.0, 1.0, 21):
            pred = P0 + float(blend) * (raw - P0)
            cand = (brier(y, pred), log_loss(y, pred), name, float(blend), j)
            if best is None or cand[:2] < best[:2]:
                best = cand
    return best


def topk_hit_rate(raw_daily, actual_daily, k):
    hits = []
    for p, y in zip(raw_daily, actual_daily):
        top = np.argsort(-p, kind="stable")[:k]
        hits.extend(y[top].tolist())
    return float(np.mean(hits)) if hits else float("nan")


def evaluate_outer_folds(draws, presence, names, cube):
    years = np.array([d["date"].year for d in draws], dtype=int)
    valid = ~np.isnan(cube[:len(draws), 0, 0])
    min_year = max(2018, int(years.min()) + 5)
    max_year = int(years.max())
    rows, all_y, all_p = [], [], []
    positive_years = 0
    for year in range(min_year, max_year + 1):
        selection_start = year - 4
        selection_idx = np.flatnonzero((years >= selection_start) & (years < year) & valid)
        test_idx = np.flatnonzero((years == year) & valid)
        if len(selection_idx) < 600 or len(test_idx) < 30:
            continue
        best = choose_recipe_blend(names, cube, presence, selection_idx)
        _, _, recipe, blend, j = best
        y, raw = flatten_period(cube, presence, test_idx, j)
        pred = P0 + blend * (raw - P0)
        base = np.full_like(y, P0, dtype=float)
        bb, bl = brier(y, base), log_loss(y, base)
        mb, ml = brier(y, pred), log_loss(y, pred)
        imp = bb - mb
        positive_years += int(imp > 0)
        raw_daily = cube[test_idx, j].astype(float)
        actual_daily = presence[test_idx].astype(float)
        rows.append({
            "year": year,
            "selection_years": f"{selection_start}-{year-1}",
            "recipe": recipe,
            "blend": blend,
            "draws": len(test_idx),
            "baseline_brier": bb,
            "model_brier": mb,
            "brier_improvement": imp,
            "baseline_log_loss": bl,
            "model_log_loss": ml,
            "log_loss_improvement": bl - ml,
            "top1_hit_rate": topk_hit_rate(raw_daily, actual_daily, 1),
            "top3_hit_rate": topk_hit_rate(raw_daily, actual_daily, 3),
            "top5_hit_rate": topk_hit_rate(raw_daily, actual_daily, 5),
        })
        all_y.append(y)
        all_p.append(pred)
    if not rows:
        raise RuntimeError("No outer folds available")
    yy, pp = np.concatenate(all_y), np.concatenate(all_p)
    base = np.full_like(yy, P0, dtype=float)
    summary = {
        "outer_folds": len(rows),
        "positive_brier_years": positive_years,
        "positive_brier_year_fraction": positive_years / len(rows),
        "nested_brier": brier(yy, pp),
        "nested_baseline_brier": brier(yy, base),
        "nested_brier_improvement": brier(yy, base) - brier(yy, pp),
        "nested_log_loss": log_loss(yy, pp),
        "nested_baseline_log_loss": log_loss(yy, base),
        "nested_log_loss_improvement": log_loss(yy, base) - log_loss(yy, pp),
        "mean_top1_hit_rate": float(np.mean([r["top1_hit_rate"] for r in rows])),
        "mean_top3_hit_rate": float(np.mean([r["top3_hit_rate"] for r in rows])),
        "mean_top5_hit_rate": float(np.mean([r["top5_hit_rate"] for r in rows])),
    }
    summary["promotion_eligible"] = bool(
        summary["nested_brier_improvement"] > 0
        and summary["nested_log_loss_improvement"] > 0
        and summary["positive_brier_year_fraction"] >= 0.60
        and summary["mean_top5_hit_rate"] >= P0
    )
    return rows, summary


def recipe_diagnostics(draws, presence, names, cube):
    years = np.array([d["date"].year for d in draws], dtype=int)
    idx = np.flatnonzero((years >= 2018) & ~np.isnan(cube[:len(draws), 0, 0]))
    out = []
    for j, name in enumerate(names):
        y, raw = flatten_period(cube, presence, idx, j)
        best = None
        for blend in np.linspace(0.0, 1.0, 21):
            p = P0 + blend * (raw - P0)
            cand = (brier(y, p), log_loss(y, p), float(blend))
            if best is None or cand[:2] < best[:2]:
                best = cand
        base = np.full_like(y, P0, dtype=float)
        out.append({
            "recipe": name,
            "best_blend": best[2],
            "brier": best[0],
            "baseline_brier": brier(y, base),
            "brier_improvement": brier(y, base) - best[0],
            "log_loss": best[1],
            "baseline_log_loss": log_loss(y, base),
            "log_loss_improvement": log_loss(y, base) - best[1],
        })
    out.sort(key=lambda r: (-r["brier_improvement"], -r["log_loss_improvement"], r["recipe"]))
    return out


def current_selection(draws, presence, names, cube, target_date):
    years = np.array([d["date"].year for d in draws], dtype=int)
    current_year = target_date.year
    selection_idx = np.flatnonzero(
        (years >= current_year - 4)
        & (years < current_year)
        & ~np.isnan(cube[:len(draws), 0, 0])
    )
    best = choose_recipe_blend(names, cube, presence, selection_idx)
    _, _, recipe, blend, j = best
    raw = cube[len(draws), j].astype(float)
    pred = P0 + blend * (raw - P0)
    rows = []
    for number in range(100):
        rows.append({
            "number": f"{number:02d}",
            "probability": float(pred[number]),
            "raw_recipe_probability": float(raw[number]),
            "uplift_vs_baseline_pp": float((pred[number] - P0) * 100.0),
            "selected_recipe": recipe,
            "selected_blend": blend,
        })
    rows.sort(key=lambda r: (-r["probability"], r["number"]))
    return rows, {
        "recipe": recipe,
        "blend": blend,
        "selection_years": f"{current_year-4}-{current_year-1}",
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "upstream" / "xsmb.csv"))
    ap.add_argument("--target-date", required=True)
    ap.add_argument("--out-dir", default=str(ROOT / "research" / "multihorizon_v3"))
    args = ap.parse_args()

    target = dt.date.fromisoformat(args.target_date)
    draws = [d for d in load_draws(args.data) if d["date"] < target]
    presence, _ = matrices(draws)
    names, cube = build_forecast_cube(draws, presence)
    folds, summary = evaluate_outer_folds(draws, presence, names, cube)
    diagnostics = recipe_diagnostics(draws, presence, names, cube)
    next_rows, selected = current_selection(draws, presence, names, cube, target)
    summary.update({
        "target_date": target.isoformat(),
        "data_cutoff": draws[-1]["date"].isoformat(),
        "observed_draws": len(draws),
        "theoretical_baseline": P0,
        "current_selected_recipe": selected["recipe"],
        "current_selected_blend": selected["blend"],
        "current_selection_years": selected["selection_years"],
        "candidate_count": len(names),
        "method": "nested yearly walk-forward selection among empirical-Bayes multi-horizon and exponential-decay recipes",
    })

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "outer_folds.csv", folds)
    write_csv(out / "recipe_diagnostics.csv", diagnostics)
    write_csv(out / "next_prediction.csv", next_rows)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print("Top 10 next draw:")
    for row in next_rows[:10]:
        print(
            row["number"],
            f"{row['probability']:.6%}",
            f"uplift={row['uplift_vs_baseline_pp']:+.4f}pp",
        )


if __name__ == "__main__":
    main()
