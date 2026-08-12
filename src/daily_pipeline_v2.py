#!/usr/bin/env python3
"""Three-year rolling wrapper for the existing 2-digit daily pipeline.

The live 2D model remains unchanged. After a brand-new pre-draw forecast is
created, descriptive full-history xAB metadata is appended to each 2D row and
the manifest hash is refreshed. Existing immutable forecasts are never edited.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import subprocess
import sys

import daily_pipeline as base

base.WINDOW_DAYS = 1095


def lazy_model_v2():
    from xsmb_probability_v2 import load_draws, matrices, next_probabilities, write_prediction
    return load_draws, matrices, next_probabilities, write_prediction


base.lazy_model = lazy_model_v2


def run_statistics_3y():
    subprocess.run(
        [sys.executable, str(base.ROOT / "src" / "build_statistics_3y.py")],
        cwd=base.ROOT,
        check=True,
    )


base.run_statistics = run_statistics_3y


def refresh_manifest_forecast_hash(target_date: dt.date, forecast_path) -> None:
    manifest = base.ROOT / "forecasts" / "manifest.csv"
    rows = base.read_csv(manifest)
    if not rows:
        raise RuntimeError("Forecast manifest missing after forecast creation")
    fields = list(rows[0])
    found = False
    for row in rows:
        if row.get("target_date") == target_date.isoformat():
            row["forecast_sha256"] = base.sha256_file(forecast_path)
            found = True
            break
    if not found:
        raise RuntimeError(f"Manifest row missing for {target_date}")
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def forecast_with_prefix3(target_date: dt.date) -> int:
    forecast_path = base.ROOT / "forecasts" / f"{target_date.isoformat()}.csv"
    existed_before = forecast_path.exists()
    rc = base.forecast(target_date)
    if rc != 0 or existed_before or not forecast_path.exists():
        return rc

    # Full descriptive history only; this does not feed back into the 2D model.
    from prefix3_history import annotate_csv, build_prefix3_stats

    upstream = base.fetch_upstream()
    historical_rows = [row for row in upstream if row["date"] < target_date.isoformat()]
    stats = build_prefix3_stats(historical_rows)
    annotate_csv(forecast_path, stats)
    refresh_manifest_forecast_hash(target_date, forecast_path)
    print("Added historical best xAB prefix annotation to the immutable live forecast")
    return rc


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("forecast")
    a.add_argument("--target-date")
    b = sub.add_parser("settle")
    b.add_argument("--date")
    args = p.parse_args()
    if args.cmd == "forecast":
        target = dt.date.fromisoformat(args.target_date) if args.target_date else base.local_today()
        return forecast_with_prefix3(target)
    target = dt.date.fromisoformat(args.date) if args.date else base.local_today()
    return base.settle(target)


if __name__ == "__main__":
    raise SystemExit(main())
