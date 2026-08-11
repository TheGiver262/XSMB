#!/usr/bin/env python3
"""Rebuild the canonical rolling 1,095-day snapshot from current upstream data."""
from __future__ import annotations
import datetime as dt
import subprocess
import sys
from pathlib import Path

from daily_pipeline import ROOT, fetch_upstream, rows_between, write_dataset_summary, write_parts, write_raw_csv
from xsmb_probability_v2 import load_draws as load2, next_probabilities as next2, write_prediction as write2
from xsmb_3digit import SPECS, load_draws as load3, next_probabilities as next3, write_prediction as write3

WINDOW_DAYS = 1095


def main() -> None:
    upstream = fetch_upstream()
    if not upstream:
        raise RuntimeError('Upstream XSMB dataset is empty')
    end = dt.date.fromisoformat(upstream[-1]['date'])
    start = end - dt.timedelta(days=WINDOW_DAYS - 1)
    rolling = rows_between(upstream, start, end)
    if len(rolling) < 900:
        raise RuntimeError(f'Unexpectedly few draws in three-year window: {len(rolling)}')

    write_parts(ROOT / 'data' / 'parts', rolling)
    write_dataset_summary(ROOT / 'data' / 'dataset_summary.json', rolling, start, end)
    subprocess.run([sys.executable, str(ROOT / 'src' / 'build_statistics_3y.py')], cwd=ROOT, check=True)

    runtime = ROOT / '.runtime'; runtime.mkdir(exist_ok=True)
    temp = runtime / 'bootstrap_3y.csv'; write_raw_csv(temp, rolling)
    target_date = end + dt.timedelta(days=1)
    try:
        draws2 = load2(temp)
        rows2, metrics2 = next2(draws2, target_date)
        metrics2.update({'forecast_kind': 'bootstrap_post_data_preview', 'data_cutoff_date': end.isoformat(), 'history_calendar_start': start.isoformat(), 'history_calendar_end': end.isoformat()})
        write2(rows2, metrics2, ROOT / 'output' / 'next_preview.csv')

        draws3 = load3(temp)
        for target in SPECS:
            rows3, metrics3 = next3(draws3, target_date, target)
            metrics3.update({'forecast_kind': 'bootstrap_post_data_preview', 'data_cutoff_date': end.isoformat(), 'history_calendar_start': start.isoformat(), 'history_calendar_end': end.isoformat()})
            write3(rows3, metrics3, ROOT / 'output' / f'next_preview_{target}.csv')
    finally:
        temp.unlink(missing_ok=True)

    print(f'Rebuilt rolling window: {start} -> {end}; {len(rolling)} observed draws')
    print(f'Preview target: {target_date}')
    print(f"2-digit feature_set={metrics2['selected_feature_set']} blend={metrics2['selected_blend']}")
    for target in SPECS:
        mp = ROOT / 'output' / f'next_preview_{target}_metrics.csv'
        print(f'3-digit output: {mp}')


if __name__ == '__main__':
    main()
