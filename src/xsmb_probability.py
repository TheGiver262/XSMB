#!/usr/bin/env python3
"""Estimate next-draw appearance probabilities for XSMB two-digit suffixes.

Outcome definition:
For each number 00..99, y=1 when that suffix appears at least once among the
27 published XSMB prize results of a draw; otherwise y=0.

The estimator is deliberately conservative:
1. Starts from the fair-draw theoretical baseline p0 = 1 - 0.99^27.
2. Fits a pooled ridge-logistic model on only lagged historical features.
3. Uses expanding-window time-series validation.
4. Shrinks the model back toward p0 by a validation-selected blend factor.
   If historical features do not improve out-of-sample Brier score, the blend
   is driven toward zero and the output reverts to the theoretical baseline.

This is a calibrated historical estimator, not evidence that lottery draws are
predictable or non-random.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
from collections import Counter
from pathlib import Path

import numpy as np

PRIZE_LENGTHS = {
    'special': 5, 'prize1': 5, 'prize2_1': 5, 'prize2_2': 5,
    **{f'prize3_{i}': 5 for i in range(1, 7)},
    **{f'prize4_{i}': 4 for i in range(1, 5)},
    **{f'prize5_{i}': 4 for i in range(1, 7)},
    **{f'prize6_{i}': 3 for i in range(1, 4)},
    **{f'prize7_{i}': 2 for i in range(1, 5)},
}
PRIZE_COLS = list(PRIZE_LENGTHS)
P0 = 1.0 - (99.0 / 100.0) ** 27
FEATURE_NAMES = [
    'hit_rate_7', 'hit_rate_30', 'hit_rate_90', 'position_rate_30',
    'bayes_long_rate', 'bayes_weekday_rate', 'gap_norm', 'prev_hit',
]


def load_draws(path: str | Path) -> list[dict]:
    draws = []
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            item = {'date': dt.date.fromisoformat(row['date'])}
            for col, width in PRIZE_LENGTHS.items():
                item[col] = str(row[col]).zfill(width)
            draws.append(item)
    draws.sort(key=lambda x: x['date'])
    return draws


def matrices(draws: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    n = len(draws)
    presence = np.zeros((n, 100), dtype=np.int8)
    counts = np.zeros((n, 100), dtype=np.int8)
    for i, row in enumerate(draws):
        suffixes = [int(row[col][-2:]) for col in PRIZE_COLS]
        cnt = Counter(suffixes)
        for number, amount in cnt.items():
            presence[i, number] = 1
            counts[i, number] = amount
    return presence, counts


def features_for_day(
    i: int,
    target_weekday: int,
    draws: list[dict],
    presence: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """Features for predicting draw index i from history [0, i)."""
    out = []
    for number in range(100):
        hist = presence[:i, number]
        hist_counts = counts[:i, number]

        def mean_window(window: int) -> float:
            values = hist[max(0, i - window):]
            return float(values.mean()) if len(values) else P0

        f7 = mean_window(7)
        f30 = mean_window(30)
        f90 = mean_window(90)
        c30 = float(hist_counts[max(0, i - 30):].mean() / 27.0) if i else 0.01

        prior_strength = 100.0
        long_rate = float((hist.sum() + prior_strength * P0) / (len(hist) + prior_strength))

        same_weekday_idx = [
            k for k in range(i) if draws[k]['date'].weekday() == target_weekday
        ]
        weekday_strength = 30.0
        if same_weekday_idx:
            weekday_hits = sum(int(presence[k, number]) for k in same_weekday_idx)
            weekday_rate = float(
                (weekday_hits + weekday_strength * P0)
                / (len(same_weekday_idx) + weekday_strength)
            )
        else:
            weekday_rate = P0

        hits = np.flatnonzero(hist)
        gap = (i - 1 - int(hits[-1])) if len(hits) else i
        gap_norm = min(gap, 30) / 30.0
        prev_hit = float(hist[-1]) if i else 0.0

        out.append([f7, f30, f90, c30, long_rate, weekday_rate, gap_norm, prev_hit])
    return np.asarray(out, dtype=float)


def build_dataset(
    start_i: int,
    end_i: int,
    draws: list[dict],
    presence: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(start_i, end_i):
        xs.append(features_for_day(i, draws[i]['date'].weekday(), draws, presence, counts))
        ys.append(presence[i])
    return np.vstack(xs), np.concatenate(ys)


def fit_logistic_ridge(
    x: np.ndarray,
    y: np.ndarray,
    l2: float = 10.0,
    max_iter: int = 50,
) -> dict:
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    z = (x - mu) / sd
    a = np.column_stack([np.ones(len(z)), z])

    beta = np.zeros(a.shape[1])
    empirical = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    beta[0] = math.log(empirical / (1 - empirical))
    reg = np.diag([0.0] + [l2] * (a.shape[1] - 1))

    for _ in range(max_iter):
        eta = np.clip(a @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = p * (1 - p)
        grad = a.T @ (p - y) + reg @ beta
        hessian = a.T @ (a * w[:, None]) + reg
        step = np.linalg.solve(hessian, grad)
        beta -= step
        if np.max(np.abs(step)) < 1e-8:
            break

    return {'mu': mu, 'sd': sd, 'beta': beta, 'l2': l2}


def predict(model: dict, x: np.ndarray) -> np.ndarray:
    z = (x - model['mu']) / model['sd']
    a = np.column_stack([np.ones(len(z)), z])
    eta = np.clip(a @ model['beta'], -30, 30)
    return 1.0 / (1.0 + np.exp(-eta))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((y - p) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def select_hyperparams(
    draws: list[dict],
    presence: np.ndarray,
    counts: np.ndarray,
    warmup: int = 90,
) -> tuple[float, float, dict]:
    n = len(draws)
    if n < 240:
        raise ValueError('Need at least 240 observed draws for the configured validation.')

    x_all, y_all = build_dataset(warmup, n, draws, presence, counts)

    def sl(a: int, b: int) -> slice:
        return slice((a - warmup) * 100, (b - warmup) * 100)

    cut1 = max(warmup + 60, n - 181)
    cut2 = max(cut1 + 40, n - 121)
    cut3 = max(cut2 + 40, n - 61)
    folds = [(cut1, cut2), (cut2, cut3), (cut3, n)]

    best = None
    for l2 in [10.0, 100.0, 300.0, 1000.0, 3000.0]:
        all_pred, all_y = [], []
        for val_start, val_end in folds:
            train_x = x_all[sl(warmup, val_start)]
            train_y = y_all[sl(warmup, val_start)]
            val_x = x_all[sl(val_start, val_end)]
            val_y = y_all[sl(val_start, val_end)]
            model = fit_logistic_ridge(train_x, train_y, l2=l2)
            all_pred.append(predict(model, val_x))
            all_y.append(val_y)
        pred = np.concatenate(all_pred)
        actual = np.concatenate(all_y)

        for blend in np.linspace(0.0, 1.0, 101):
            blended = P0 + blend * (pred - P0)
            score = brier(actual, blended)
            candidate = (score, l2, float(blend), log_loss(actual, blended))
            if best is None or candidate[0] < best[0]:
                best = candidate

    baseline_brier = brier(actual, np.full_like(actual, P0, dtype=float))
    baseline_logloss = log_loss(actual, np.full_like(actual, P0, dtype=float))
    metrics = {
        'cv_brier': best[0],
        'cv_log_loss': best[3],
        'baseline_brier': baseline_brier,
        'baseline_log_loss': baseline_logloss,
        'brier_improvement': baseline_brier - best[0],
    }
    return best[1], best[2], metrics


def next_probabilities(draws: list[dict], target_date: dt.date) -> tuple[list[dict], dict]:
    presence, counts = matrices(draws)
    n = len(draws)
    warmup = 90
    l2, blend, metrics = select_hyperparams(draws, presence, counts, warmup=warmup)
    x_all, y_all = build_dataset(warmup, n, draws, presence, counts)
    model = fit_logistic_ridge(x_all, y_all, l2=l2)

    x_next = features_for_day(n, target_date.weekday(), draws, presence, counts)
    raw_p = predict(model, x_next)
    final_p = P0 + blend * (raw_p - P0)

    rows = []
    for number in range(100):
        f = x_next[number]
        rows.append({
            'number': f'{number:02d}',
            'probability': float(final_p[number]),
            'raw_model_probability': float(raw_p[number]),
            **{name: float(f[idx]) for idx, name in enumerate(FEATURE_NAMES)},
        })
    rows.sort(key=lambda r: r['probability'], reverse=True)
    metrics.update({
        'theoretical_baseline': P0,
        'selected_l2': l2,
        'selected_blend': blend,
        'observed_draws': n,
        'target_date': target_date.isoformat(),
    })
    return rows, metrics


def write_prediction(rows: list[dict], metrics: dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    metrics_path = out_path.with_name(out_path.stem + '_metrics.csv')
    with metrics_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for key, value in metrics.items():
            writer.writerow([key, value])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Raw XSMB CSV')
    parser.add_argument('--target-date', required=True, help='YYYY-MM-DD')
    parser.add_argument('--out', required=True, help='Prediction CSV output')
    args = parser.parse_args()

    draws = load_draws(args.data)
    target = dt.date.fromisoformat(args.target_date)
    rows, metrics = next_probabilities(draws, target)
    write_prediction(rows, metrics, args.out)

    print(f"Observed draws: {metrics['observed_draws']}")
    print(f"Theoretical baseline: {metrics['theoretical_baseline']:.6%}")
    print(f"Selected ridge L2: {metrics['selected_l2']}")
    print(f"Selected blend: {metrics['selected_blend']:.2f}")
    print(f"CV Brier: {metrics['cv_brier']:.9f}")
    print(f"Baseline Brier: {metrics['baseline_brier']:.9f}")
    print('Top 10:')
    for row in rows[:10]:
        print(f"{row['number']}: {row['probability']:.4%}")


if __name__ == '__main__':
    main()
