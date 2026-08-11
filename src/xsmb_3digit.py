#!/usr/bin/env python3
"""Conservative 3-digit XSMB probability models.

Targets:
- suffix3_any: 000..999 appears as the last three digits of any prize that has
  at least three published digits (23 positions: DB, G1..G6; G7 excluded).
- g6_exact: 000..999 appears in any of the three exact G6 results.

The model is an empirical-Bayes ensemble with walk-forward selection of both
recipe and shrinkage-to-baseline. It is designed for sparse events.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import numpy as np

PRIZE_LENGTHS = {
    'special': 5, 'prize1': 5, 'prize2_1': 5, 'prize2_2': 5,
    **{f'prize3_{i}': 5 for i in range(1, 7)},
    **{f'prize4_{i}': 4 for i in range(1, 5)},
    **{f'prize5_{i}': 4 for i in range(1, 7)},
    **{f'prize6_{i}': 3 for i in range(1, 4)},
    **{f'prize7_{i}': 2 for i in range(1, 5)},
}
ALL_3PLUS_COLS = [c for c, width in PRIZE_LENGTHS.items() if width >= 3]
G6_COLS = [f'prize6_{i}' for i in range(1, 4)]


@dataclass(frozen=True)
class TargetSpec:
    name: str
    cols: tuple[str, ...]
    recent_short: int
    recent_long: int
    weekday_prior: float
    long_prior: float
    gap_cap: int
    validation_draws: int

    @property
    def positions(self) -> int:
        return len(self.cols)

    @property
    def baseline(self) -> float:
        return 1.0 - (999.0 / 1000.0) ** self.positions


SPECS = {
    'suffix3_any': TargetSpec('suffix3_any', tuple(ALL_3PLUS_COLS), 60, 180, 140.0, 500.0, 180, 365),
    'g6_exact': TargetSpec('g6_exact', tuple(G6_COLS), 120, 365, 210.0, 900.0, 365, 540),
}

RECIPES = {
    'long_only': {'long': 1.0},
    'recent': {'long': 0.50, 'short': 0.30, 'recent_long': 0.20},
    'weekday': {'long': 0.45, 'short': 0.25, 'recent_long': 0.15, 'weekday': 0.15},
    'trend_gap': {'long': 0.40, 'short': 0.30, 'recent_long': 0.15, 'weekday': 0.10, 'gap': 0.05},
}


def load_draws(path: str | Path) -> list[dict]:
    path = Path(path)
    paths = sorted(path.glob('xsmb_part_*.csv')) if path.is_dir() else [path]
    if not paths:
        raise FileNotFoundError(path)
    out = []
    for csv_path in paths:
        with csv_path.open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                item = {'date': dt.date.fromisoformat(row['date'])}
                for col, width in PRIZE_LENGTHS.items():
                    item[col] = str(row[col]).zfill(width)
                out.append(item)
    out.sort(key=lambda x: x['date'])
    return out


def matrices(draws: list[dict], spec: TargetSpec) -> tuple[np.ndarray, np.ndarray]:
    n = len(draws)
    presence = np.zeros((n, 1000), dtype=np.int8)
    counts = np.zeros((n, 1000), dtype=np.int8)
    for i, row in enumerate(draws):
        values = [int(row[col][-3:]) for col in spec.cols]
        for value, count in Counter(values).items():
            presence[i, value] = 1
            counts[i, value] = count
    return presence, counts


def features_at(i: int, target_weekday: int, draws: list[dict], presence: np.ndarray, spec: TargetSpec) -> dict[str, np.ndarray]:
    p0 = spec.baseline
    hist = presence[:i]
    if i == 0:
        base = np.full(1000, p0, dtype=float)
        return {'long': base, 'short': base, 'recent_long': base, 'weekday': base, 'gap': base}

    long = (hist.sum(axis=0) + spec.long_prior * p0) / (i + spec.long_prior)
    s = hist[max(0, i - spec.recent_short):]
    short = (s.sum(axis=0) + spec.recent_short * p0) / (len(s) + spec.recent_short)
    l = hist[max(0, i - spec.recent_long):]
    recent_long = (l.sum(axis=0) + spec.recent_long * p0) / (len(l) + spec.recent_long)

    weekday_idx = [k for k in range(i) if draws[k]['date'].weekday() == target_weekday]
    if weekday_idx:
        whits = hist[weekday_idx].sum(axis=0)
        weekday = (whits + spec.weekday_prior * p0) / (len(weekday_idx) + spec.weekday_prior)
    else:
        weekday = np.full(1000, p0, dtype=float)

    last = np.full(1000, -1, dtype=int)
    for k in range(i):
        hit = np.flatnonzero(hist[k])
        last[hit] = k
    gaps = np.where(last >= 0, i - 1 - last, i)
    centered = np.clip(gaps / max(spec.gap_cap, 1), 0.0, 1.0) - 0.5
    gap = np.clip(p0 * (1.0 + 0.12 * centered), p0 * 0.5, min(0.999, p0 * 1.5))

    return {
        'long': np.asarray(long, dtype=float),
        'short': np.asarray(short, dtype=float),
        'recent_long': np.asarray(recent_long, dtype=float),
        'weekday': np.asarray(weekday, dtype=float),
        'gap': np.asarray(gap, dtype=float),
    }


def recipe_probability(features: dict[str, np.ndarray], recipe: str) -> np.ndarray:
    weights = RECIPES[recipe]
    out = np.zeros(1000, dtype=float)
    for name, weight in weights.items():
        out += weight * features[name]
    return np.clip(out, 1e-9, 1 - 1e-9)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((y - p) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def allowed_recipes(spec: TargetSpec) -> list[str]:
    gate = ROOT / 'research' / 'feature_gate.json'
    if gate.exists():
        try:
            data = json.loads(gate.read_text(encoding='utf-8'))
            names = data.get(spec.name, {}).get('allowed_recipes', [])
            names = [n for n in names if n in RECIPES]
            if names:
                return names
        except Exception:
            pass
    return list(RECIPES)


def select_recipe(draws: list[dict], presence: np.ndarray, spec: TargetSpec) -> tuple[str, float, dict]:
    n = len(draws)
    if n < 540:
        raise ValueError('Need at least 540 observed draws for 3-digit validation')
    start = max(365, n - spec.validation_draws)
    ys = []
    preds = {name: [] for name in RECIPES}
    for i in range(start, n):
        f = features_at(i, draws[i]['date'].weekday(), draws, presence, spec)
        ys.append(presence[i].astype(float))
        for name in RECIPES:
            preds[name].append(recipe_probability(f, name))
    actual = np.concatenate(ys)
    baseline = np.full_like(actual, spec.baseline, dtype=float)
    baseline_b = brier(actual, baseline)
    baseline_ll = log_loss(actual, baseline)

    best = None
    for name, chunks in preds.items():
        if name not in allowed_recipes(spec):
            continue
        raw = np.concatenate(chunks)
        for blend in np.linspace(0.0, 1.0, 51):
            calibrated = spec.baseline + blend * (raw - spec.baseline)
            score = brier(actual, calibrated)
            candidate = (score, log_loss(actual, calibrated), name, float(blend))
            if best is None or candidate[:2] < best[:2]:
                best = candidate
    metrics = {
        'cv_brier': best[0],
        'cv_log_loss': best[1],
        'baseline_brier': baseline_b,
        'baseline_log_loss': baseline_ll,
        'brier_improvement': baseline_b - best[0],
        'validation_draws': n - start,
    }
    return best[2], best[3], metrics


def next_probabilities(draws: list[dict], target_date: dt.date, target: str) -> tuple[list[dict], dict]:
    spec = SPECS[target]
    presence, _counts = matrices(draws, spec)
    recipe, blend, metrics = select_recipe(draws, presence, spec)
    f = features_at(len(draws), target_date.weekday(), draws, presence, spec)
    raw = recipe_probability(f, recipe)
    final = spec.baseline + blend * (raw - spec.baseline)
    rows = []
    for number in range(1000):
        rows.append({
            'number': f'{number:03d}',
            'probability': float(final[number]),
            'raw_model_probability': float(raw[number]),
            'long_rate': float(f['long'][number]),
            'short_rate': float(f['short'][number]),
            'recent_long_rate': float(f['recent_long'][number]),
            'weekday_rate': float(f['weekday'][number]),
            'gap_signal': float(f['gap'][number]),
        })
    rows.sort(key=lambda r: (-r['probability'], r['number']))
    metrics.update({
        'target': target,
        'positions_per_draw': spec.positions,
        'theoretical_baseline': spec.baseline,
        'selected_recipe': recipe,
        'selected_blend': blend,
        'observed_draws': len(draws),
        'target_date': target_date.isoformat(),
    })
    return rows, metrics


def write_prediction(rows: list[dict], metrics: dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    mp = out_path.with_name(out_path.stem + '_metrics.csv')
    with mp.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['metric','value'])
        for k, v in metrics.items(): w.writerow([k, v])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--target-date', required=True)
    p.add_argument('--target', choices=sorted(SPECS), default='suffix3_any')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    draws = load_draws(args.data)
    rows, metrics = next_probabilities(draws, dt.date.fromisoformat(args.target_date), args.target)
    write_prediction(rows, metrics, args.out)
    print(f"Target: {args.target}; draws={len(draws)}; baseline={metrics['theoretical_baseline']:.6%}")
    print(f"Recipe={metrics['selected_recipe']}; blend={metrics['selected_blend']:.2f}")
    for row in rows[:20]: print(f"{row['number']}: {row['probability']:.5%}")

if __name__ == '__main__':
    main()
