#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np

from xsmb_probability import P0, load_draws, matrices
from xsmb_multihorizon_v3 import build_forecast_cube

ROOT = Path(__file__).resolve().parents[1]
BASE_NAMES = [
    "w1y_s300", "w3y_s300", "w5y_s300", "w10y_s300", "full_s300",
    "decay1y_s300", "decay3y_s300", "decay5y_s300",
    "multi_3_5_full_s300", "multi_1_3_5_10_full_s300", "decay_mix_s300", "hybrid_s300",
]


def percentile_rank_row(x: np.ndarray) -> np.ndarray:
    """Average percentile ranks for ties; no arbitrary number-order advantage."""
    order = np.argsort(x, kind="stable")
    sx = x[order]
    ranks = np.empty(len(x), dtype=float)
    n = len(x)
    pos = 0
    while pos < n:
        end = pos + 1
        while end < n and sx[end] == sx[pos]:
            end += 1
        avg_rank = ((pos + end - 1) / 2.0) / max(1, n - 1)
        ranks[order[pos:end]] = avg_rank
        pos = end
    return ranks


def rank_cube(raw_cube: np.ndarray) -> np.ndarray:
    out = np.full_like(raw_cube, np.nan, dtype=np.float32)
    for i in range(raw_cube.shape[0]):
        if np.isnan(raw_cube[i, 0, 0]):
            continue
        for j in range(raw_cube.shape[1]):
            out[i, j] = percentile_rank_row(raw_cube[i, j])
    return out


def build_scores(names, cube):
    name_to_idx = {n: i for i, n in enumerate(names)}
    selected_idx = [name_to_idx[n] for n in BASE_NAMES]
    base = cube[:, selected_idx, :].astype(np.float32)
    base_rank = rank_cube(base)
    score_names = list(BASE_NAMES)
    score_arrays = [base_rank[:, j, :] for j in range(len(BASE_NAMES))]

    def add(name, members, reducer="mean"):
        idx = [BASE_NAMES.index(m) for m in members]
        stack = base_rank[:, idx, :]
        with np.errstate(invalid="ignore"):
            arr = np.nanmean(stack, axis=1) if reducer == "mean" else np.nanmin(stack, axis=1)
        score_names.append(name)
        score_arrays.append(arr.astype(np.float32))

    add("consensus_1_3_5", ["w1y_s300", "w3y_s300", "w5y_s300"])
    add("consensus_3_5_10", ["w3y_s300", "w5y_s300", "w10y_s300"])
    add("consensus_all", ["w1y_s300", "w3y_s300", "w5y_s300", "w10y_s300", "full_s300"])
    add("consensus_decay", ["decay1y_s300", "decay3y_s300", "decay5y_s300"])
    add("stable_hot_1_3_5", ["w1y_s300", "w3y_s300", "w5y_s300"], reducer="min")
    return score_names, np.stack(score_arrays, axis=1)


def topk_stats(score_matrix, actual, idx, k):
    hits = 0
    total = 0
    for i in idx:
        top = np.argsort(-score_matrix[i], kind="stable")[:k]
        hits += int(actual[i, top].sum())
        total += k
    rate = hits / total if total else float("nan")
    return hits, total, rate


def posterior_rate(hits, total, strength=1000.0):
    return (hits + strength * P0) / (total + strength)


def choose_ranker(score_names, scores, actual, years, selection_idx):
    best = None
    detail = []
    selection_years = sorted(set(years[selection_idx].tolist()))
    for j, name in enumerate(score_names):
        h1, n1, r1 = topk_stats(scores[:, j], actual, selection_idx, 1)
        h3, n3, r3 = topk_stats(scores[:, j], actual, selection_idx, 3)
        h5, n5, r5 = topk_stats(scores[:, j], actual, selection_idx, 5)
        p1, p3, p5 = posterior_rate(h1, n1), posterior_rate(h3, n3), posterior_rate(h5, n5)
        yr3 = []
        for yr in selection_years:
            yi = selection_idx[years[selection_idx] == yr]
            _, _, rate = topk_stats(scores[:, j], actual, yi, 3)
            yr3.append(rate)
        std3 = float(np.std(yr3)) if yr3 else 0.0
        positive_fraction = float(np.mean(np.asarray(yr3) > P0)) if yr3 else 0.0
        core = 0.20 * p1 + 0.55 * p3 + 0.25 * p5
        objective = core - 0.50 * std3 + 0.002 * (positive_fraction - 0.5)
        if positive_fraction < 0.50:
            objective -= 0.005
        cand = (objective, positive_fraction, -std3, p3, p5, p1, name, j)
        detail.append({
            "ranker": name,
            "selection_top1_rate": r1,
            "selection_top3_rate": r3,
            "selection_top5_rate": r5,
            "posterior_top1": p1,
            "posterior_top3": p3,
            "posterior_top5": p5,
            "top3_year_std": std3,
            "top3_positive_year_fraction": positive_fraction,
            "objective": objective,
        })
        if best is None or cand[:6] > best[:6]:
            best = cand
    detail.sort(key=lambda r: (-r["objective"], r["ranker"]))
    return best, detail


def binomial_z(rate, n):
    se = math.sqrt(P0 * (1 - P0) / n)
    return (rate - P0) / se if se > 0 else 0.0


def fixed_diagnostics(draws, actual, score_names, scores):
    years = np.array([d["date"].year for d in draws], dtype=int)
    valid = ~np.isnan(scores[:len(draws), 0, 0])
    eval_idx = np.flatnonzero((years >= 2018) & valid)
    eval_years = sorted(set(years[eval_idx].tolist()))
    rows = []
    for j, name in enumerate(score_names):
        row = {"ranker": name}
        for k in [1, 3, 5]:
            h, n, r = topk_stats(scores[:, j], actual, eval_idx, k)
            row[f"top{k}_hit_rate"] = r
            row[f"top{k}_lift_pp"] = (r - P0) * 100.0
            row[f"top{k}_z"] = binomial_z(r, n)
        yr_rates = []
        for yr in eval_years:
            yi = np.flatnonzero((years == yr) & valid)
            _, _, r = topk_stats(scores[:, j], actual, yi, 3)
            yr_rates.append(r)
        row["top3_positive_year_fraction"] = float(np.mean(np.asarray(yr_rates) > P0))
        row["top3_year_std"] = float(np.std(yr_rates))
        rows.append(row)
    rows.sort(key=lambda r: (-r["top3_hit_rate"], -r["top3_positive_year_fraction"], r["ranker"]))
    return rows


def evaluate(draws, actual, score_names, scores):
    years = np.array([d["date"].year for d in draws], dtype=int)
    valid = ~np.isnan(scores[:len(draws), 0, 0])
    rows = []
    pooled = {1: [0, 0], 3: [0, 0], 5: [0, 0]}
    pos3 = 0
    for year in range(max(2018, int(years.min()) + 5), int(years.max()) + 1):
        selection_idx = np.flatnonzero((years >= year - 4) & (years < year) & valid)
        test_idx = np.flatnonzero((years == year) & valid)
        if len(selection_idx) < 600 or len(test_idx) < 30:
            continue
        best, _ = choose_ranker(score_names, scores, actual, years, selection_idx)
        objective, sel_pos, neg_std, sel_p3, sel_p5, sel_p1, name, j = best
        row = {
            "year": year,
            "selection_years": f"{year-4}-{year-1}",
            "ranker": name,
            "selection_objective": objective,
            "selection_top3_positive_year_fraction": sel_pos,
            "selection_top3_year_std": -neg_std,
            "selection_posterior_top1": sel_p1,
            "selection_posterior_top3": sel_p3,
            "selection_posterior_top5": sel_p5,
            "draws": len(test_idx),
        }
        for k in [1, 3, 5]:
            h, n, r = topk_stats(scores[:, j], actual, test_idx, k)
            row[f"top{k}_hits"] = h
            row[f"top{k}_trials"] = n
            row[f"top{k}_hit_rate"] = r
            row[f"top{k}_lift_pp"] = (r - P0) * 100.0
            pooled[k][0] += h
            pooled[k][1] += n
        pos3 += int(row["top3_hit_rate"] > P0)
        rows.append(row)

    summary = {
        "outer_folds": len(rows),
        "positive_top3_years": pos3,
        "positive_top3_year_fraction": pos3 / len(rows),
    }
    for k in [1, 3, 5]:
        h, n = pooled[k]
        r = h / n
        summary[f"pooled_top{k}_hits"] = h
        summary[f"pooled_top{k}_trials"] = n
        summary[f"pooled_top{k}_hit_rate"] = r
        summary[f"pooled_top{k}_lift_pp"] = (r - P0) * 100.0
        summary[f"pooled_top{k}_z_vs_baseline"] = binomial_z(r, n)
    summary["ranking_promotion_eligible"] = bool(
        summary["pooled_top3_hit_rate"] > P0
        and summary["positive_top3_year_fraction"] >= 0.60
        and summary["pooled_top3_z_vs_baseline"] >= 1.96
        and summary["pooled_top5_hit_rate"] >= P0
    )
    return rows, summary


def current_ranking(draws, actual, score_names, scores, target_date):
    years = np.array([d["date"].year for d in draws], dtype=int)
    valid = ~np.isnan(scores[:len(draws), 0, 0])
    selection_idx = np.flatnonzero((years >= target_date.year - 4) & (years < target_date.year) & valid)
    best, detail = choose_ranker(score_names, scores, actual, years, selection_idx)
    objective, sel_pos, neg_std, p3, p5, p1, name, j = best
    current = scores[len(draws), j].astype(float)
    order = np.argsort(-current, kind="stable")
    rows = []
    for rank, num in enumerate(order, start=1):
        rows.append({
            "rank": rank,
            "number": f"{num:02d}",
            "score_percentile": float(current[num]),
            "selected_ranker": name,
            "selection_posterior_top1": p1,
            "selection_posterior_top3": p3,
            "selection_posterior_top5": p5,
            "selection_top3_positive_year_fraction": sel_pos,
            "selection_top3_year_std": -neg_std,
        })
    return rows, detail, {
        "ranker": name,
        "objective": objective,
        "posterior_top3": p3,
        "positive_fraction": sel_pos,
        "year_std": -neg_std,
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
    ap.add_argument("--out-dir", default=str(ROOT / "research" / "rank_challenger"))
    args = ap.parse_args()
    target = dt.date.fromisoformat(args.target_date)
    draws = [d for d in load_draws(args.data) if d["date"] < target]
    actual, _ = matrices(draws)
    names, cube = build_forecast_cube(draws, actual)
    score_names, scores = build_scores(names, cube)
    folds, summary = evaluate(draws, actual, score_names, scores)
    fixed = fixed_diagnostics(draws, actual, score_names, scores)
    ranking, diagnostics, current = current_ranking(draws, actual, score_names, scores, target)
    summary.update({
        "target_date": target.isoformat(),
        "data_cutoff": draws[-1]["date"].isoformat(),
        "theoretical_baseline": P0,
        "candidate_rankers": len(score_names),
        "current_selected_ranker": current["ranker"],
        "current_selection_posterior_top3": current["posterior_top3"],
        "current_selection_top3_positive_year_fraction": current["positive_fraction"],
        "current_selection_top3_year_std": current["year_std"],
        "selection_method": "4-year nested selection, tie-aware ranks, beta shrinkage and year-stability penalty",
    })
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "outer_folds.csv", folds)
    write_csv(out / "fixed_ranker_diagnostics.csv", fixed)
    write_csv(out / "current_ranking.csv", ranking)
    write_csv(out / "current_ranker_diagnostics.csv", diagnostics)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Top 10 ranking:")
    for r in ranking[:10]:
        print(r["rank"], r["number"], f"score={r['score_percentile']:.4f}")


if __name__ == "__main__":
    main()
