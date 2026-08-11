#!/usr/bin/env python3
"""Daily XSMB live-forecast and settlement pipeline.

Design goals:
- Live forecasts are immutable snapshots committed before the draw.
- Forecast training always excludes the target date (anti-look-ahead).
- Settlement only evaluates a forecast after an observed draw is available.
- The canonical training dataset is a rolling 730-calendar-day window.
- Live model quality is tracked with Brier/log-loss over 30/60/90 settled draws.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import subprocess
import sys
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_URL = (
    "https://raw.githubusercontent.com/khiemdoan/"
    "vietnam-lottery-xsmb-analysis/refs/heads/main/data/xsmb.csv"
)
LOCAL_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
WINDOW_DAYS = 730
PART_SIZE = 91

PRIZE_LENGTHS = {
    "special": 5,
    "prize1": 5,
    "prize2_1": 5,
    "prize2_2": 5,
    **{f"prize3_{i}": 5 for i in range(1, 7)},
    **{f"prize4_{i}": 4 for i in range(1, 5)},
    **{f"prize5_{i}": 4 for i in range(1, 7)},
    **{f"prize6_{i}": 3 for i in range(1, 4)},
    **{f"prize7_{i}": 2 for i in range(1, 5)},
}
PRIZE_COLS = list(PRIZE_LENGTHS)
RAW_FIELDS = ["date", *PRIZE_COLS]
P0 = 1.0 - (99.0 / 100.0) ** 27


def local_today() -> dt.date:
    return dt.datetime.now(LOCAL_TZ).date()


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch_upstream(url: str = UPSTREAM_URL) -> list[dict[str, str]]:
    request = urllib.request.Request(url, headers={"User-Agent": "xsmb-probability-pipeline/1.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        text = response.read().decode("utf-8-sig")
    return parse_upstream_csv(text)


def parse_upstream_csv(text: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or any(name not in reader.fieldnames for name in RAW_FIELDS):
        raise ValueError("Upstream CSV schema does not contain the expected XSMB columns")

    rows: list[dict[str, str]] = []
    seen_dates: set[str] = set()
    for raw in reader:
        date_text = (raw.get("date") or "").strip()
        if not date_text:
            continue
        dt.date.fromisoformat(date_text)
        if date_text in seen_dates:
            raise ValueError(f"Duplicate upstream draw date: {date_text}")
        seen_dates.add(date_text)

        row = {"date": date_text}
        for col in PRIZE_COLS:
            value = (raw.get(col) or "").strip()
            if not value or not value.isdigit():
                raise ValueError(f"Invalid value for {col} on {date_text}: {value!r}")
            row[col] = value
        rows.append(row)

    rows.sort(key=lambda row: row["date"])
    return rows


def rows_between(
    rows: list[dict[str, str]], start: dt.date, end: dt.date
) -> list[dict[str, str]]:
    if end < start:
        return []
    return [row for row in rows if start <= dt.date.fromisoformat(row["date"]) <= end]


def write_raw_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_parts(parts_dir: Path, rows: list[dict[str, str]], part_size: int = PART_SIZE) -> None:
    parts_dir.mkdir(parents=True, exist_ok=True)
    for old in parts_dir.glob("xsmb_part_*.csv"):
        old.unlink()
    for idx, start in enumerate(range(0, len(rows), part_size), start=1):
        write_raw_csv(parts_dir / f"xsmb_part_{idx:02d}.csv", rows[start : start + part_size])


def missing_dates(start: dt.date, end: dt.date, rows: list[dict[str, str]]) -> list[str]:
    observed = {row["date"] for row in rows}
    result = []
    day = start
    while day <= end:
        if day.isoformat() not in observed:
            result.append(day.isoformat())
        day += dt.timedelta(days=1)
    return result


def write_dataset_summary(path: Path, rows: list[dict[str, str]], start: dt.date, end: dt.date) -> None:
    summary = {
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "calendar_days": (end - start).days + 1,
        "observed_draws": len(rows),
        "missing_calendar_dates": missing_dates(start, end, rows),
        "upstream_url": UPSTREAM_URL,
        "updated_at_utc": utc_now_iso(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def lazy_model():
    from xsmb_probability import load_draws, matrices, next_probabilities, write_prediction
    return load_draws, matrices, next_probabilities, write_prediction


def generate_prediction(
    history_rows: list[dict[str, str]],
    target_date: dt.date,
    out_path: Path,
    metadata: dict[str, object],
) -> tuple[list[dict], dict]:
    load_draws, _matrices, next_probabilities, write_prediction = lazy_model()
    runtime_dir = ROOT / ".runtime"
    runtime_dir.mkdir(exist_ok=True)
    temp_csv = runtime_dir / f"history_{target_date.isoformat()}.csv"
    write_raw_csv(temp_csv, history_rows)
    try:
        draws = load_draws(temp_csv)
        prediction_rows, metrics = next_probabilities(draws, target_date)
        metrics.update(metadata)
        write_prediction(prediction_rows, metrics, out_path)
        return prediction_rows, metrics
    finally:
        temp_csv.unlink(missing_ok=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def append_unique_csv(path: Path, row: dict[str, object], key: str, fieldnames: list[str]) -> bool:
    existing = read_csv(path)
    key_value = str(row[key])
    if any(existing_row.get(key) == key_value for existing_row in existing):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})
    return True


def update_manifest(target_date: dt.date, forecast_path: Path, metrics_path: Path, metrics: dict) -> None:
    manifest = ROOT / "forecasts" / "manifest.csv"
    row = {
        "target_date": target_date.isoformat(),
        "generated_at_utc": metrics["generated_at_utc"],
        "data_cutoff_date": metrics["data_cutoff_date"],
        "observed_draws": metrics["observed_draws"],
        "selected_l2": metrics["selected_l2"],
        "selected_blend": metrics["selected_blend"],
        "forecast_sha256": sha256_file(forecast_path),
        "metrics_sha256": sha256_file(metrics_path),
    }
    append_unique_csv(manifest, row, "target_date", list(row))


def forecast(target_date: dt.date) -> int:
    forecast_dir = ROOT / "forecasts"
    forecast_path = forecast_dir / f"{target_date.isoformat()}.csv"
    metrics_path = forecast_dir / f"{target_date.isoformat()}_metrics.csv"
    if forecast_path.exists():
        print(f"Live forecast already exists and is immutable: {forecast_path}")
        return 0

    upstream = fetch_upstream()
    if any(row["date"] == target_date.isoformat() for row in upstream):
        raise RuntimeError(
            f"Draw {target_date} is already present upstream; refusing to create a retrospective live forecast"
        )
    cutoff = target_date - dt.timedelta(days=1)
    history_start = target_date - dt.timedelta(days=WINDOW_DAYS)
    history = rows_between(upstream, history_start, cutoff)
    if len(history) < 240:
        raise RuntimeError(f"Only {len(history)} historical draws available; need at least 240")

    generated_at = utc_now_iso()
    prediction_rows, metrics = generate_prediction(
        history,
        target_date,
        forecast_path,
        {
            "forecast_kind": "live_pre_draw",
            "generated_at_utc": generated_at,
            "data_cutoff_date": cutoff.isoformat(),
            "history_calendar_start": history_start.isoformat(),
            "history_calendar_end": cutoff.isoformat(),
            "upstream_max_date_seen": upstream[-1]["date"] if upstream else "",
        },
    )
    update_manifest(target_date, forecast_path, metrics_path, metrics)
    print(f"Created immutable live forecast for {target_date} from {len(history)} observed draws")
    print(f"Data cutoff: {cutoff}")
    print("Top 10:")
    for row in prediction_rows[:10]:
        print(f"  {row['number']}: {row['probability']:.4%}")
    return 0


def actual_presence(draw_row: dict[str, str]) -> set[str]:
    result = set()
    for col, width in PRIZE_LENGTHS.items():
        value = str(draw_row[col]).zfill(width)
        result.add(value[-2:])
    return result


def safe_log_loss(y: int, p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def evaluate_forecast_rows(
    forecast_rows: list[dict[str, str]], draw_row: dict[str, str]
) -> dict[str, float | int]:
    if len(forecast_rows) != 100:
        raise ValueError(f"Expected 100 forecast rows, got {len(forecast_rows)}")
    actual = actual_presence(draw_row)
    ordered = sorted(forecast_rows, key=lambda r: (-float(r["probability"]), int(r["number"])))
    squared_errors = []
    log_losses = []
    probabilities = []
    for row in ordered:
        number = f"{int(row['number']):02d}"
        p = float(row["probability"])
        y = 1 if number in actual else 0
        squared_errors.append((y - p) ** 2)
        log_losses.append(safe_log_loss(y, p))
        probabilities.append(p)

    baseline_sq = []
    baseline_ll = []
    for n in range(100):
        y = 1 if f"{n:02d}" in actual else 0
        baseline_sq.append((y - P0) ** 2)
        baseline_ll.append(safe_log_loss(y, P0))

    brier = sum(squared_errors) / 100.0
    base_brier = sum(baseline_sq) / 100.0
    logloss = sum(log_losses) / 100.0
    base_logloss = sum(baseline_ll) / 100.0
    return {
        "brier": brier,
        "baseline_brier": base_brier,
        "brier_improvement": base_brier - brier,
        "log_loss": logloss,
        "baseline_log_loss": base_logloss,
        "log_loss_improvement": base_logloss - logloss,
        "mean_probability": sum(probabilities) / 100.0,
        "actual_unique_numbers": len(actual),
        "top10_hits": sum(1 for row in ordered[:10] if f"{int(row['number']):02d}" in actual),
        "top20_hits": sum(1 for row in ordered[:20] if f"{int(row['number']):02d}" in actual),
    }


def record_evaluation(as_of_date: dt.date, draw_row: dict[str, str]) -> None:
    forecast_path = ROOT / "forecasts" / f"{as_of_date.isoformat()}.csv"
    daily_path = ROOT / "evaluation" / "daily_metrics.csv"
    fields = [
        "date", "status", "brier", "baseline_brier", "brier_improvement",
        "log_loss", "baseline_log_loss", "log_loss_improvement", "mean_probability",
        "actual_unique_numbers", "top10_hits", "top20_hits", "forecast_sha256",
    ]
    if forecast_path.exists():
        metrics = evaluate_forecast_rows(read_csv(forecast_path), draw_row)
        row: dict[str, object] = {
            "date": as_of_date.isoformat(),
            "status": "evaluated",
            **metrics,
            "forecast_sha256": sha256_file(forecast_path),
        }
    else:
        row = {"date": as_of_date.isoformat(), "status": "missing_forecast", "forecast_sha256": ""}
    append_unique_csv(daily_path, row, "date", fields)


def write_rolling_summary() -> None:
    evaluated = [
        row for row in read_csv(ROOT / "evaluation" / "daily_metrics.csv")
        if row.get("status") == "evaluated"
    ]
    out_path = ROOT / "evaluation" / "rolling_summary.csv"
    fields = [
        "window", "evaluated_draws", "model_brier", "baseline_brier", "brier_improvement",
        "model_log_loss", "baseline_log_loss", "log_loss_improvement", "mean_top10_hits",
        "mean_top20_hits", "brier_wins_vs_baseline",
    ]
    rows = []
    for window in (30, 60, 90):
        sample = evaluated[-window:]
        if not sample:
            rows.append({"window": window, "evaluated_draws": 0})
            continue

        def avg(name: str) -> float:
            return sum(float(row[name]) for row in sample) / len(sample)

        model_brier = avg("brier")
        baseline_brier = avg("baseline_brier")
        model_logloss = avg("log_loss")
        baseline_logloss = avg("baseline_log_loss")
        rows.append({
            "window": window,
            "evaluated_draws": len(sample),
            "model_brier": model_brier,
            "baseline_brier": baseline_brier,
            "brier_improvement": baseline_brier - model_brier,
            "model_log_loss": model_logloss,
            "baseline_log_loss": baseline_logloss,
            "log_loss_improvement": baseline_logloss - model_logloss,
            "mean_top10_hits": avg("top10_hits"),
            "mean_top20_hits": avg("top20_hits"),
            "brier_wins_vs_baseline": sum(
                1 for row in sample if float(row["brier"]) < float(row["baseline_brier"])
            ),
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def append_model_run(as_of_date: dt.date, target_date: dt.date, metrics: dict) -> None:
    path = ROOT / "evaluation" / "model_runs.csv"
    row = {
        "as_of_date": as_of_date.isoformat(),
        "target_date": target_date.isoformat(),
        "observed_draws": metrics["observed_draws"],
        "selected_l2": metrics["selected_l2"],
        "selected_blend": metrics["selected_blend"],
        "cv_brier": metrics["cv_brier"],
        "baseline_brier": metrics["baseline_brier"],
        "brier_improvement": metrics["brier_improvement"],
        "generated_at_utc": metrics["generated_at_utc"],
    }
    append_unique_csv(path, row, "as_of_date", list(row))


def run_statistics() -> None:
    script = ROOT / "src" / "build_statistics.py"
    if script.exists():
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)


def settle(as_of_date: dt.date) -> int:
    state_path = ROOT / "output" / "pipeline_state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {}
        if state.get("last_settled_draw") == as_of_date.isoformat():
            print(f"Draw {as_of_date} is already settled; no changes needed.")
            return 0

    upstream = fetch_upstream()
    draw_map = {row["date"]: row for row in upstream}
    draw_row = draw_map.get(as_of_date.isoformat())
    if draw_row is None:
        print(f"No upstream XSMB draw for {as_of_date}; nothing to settle (holiday or source delay).")
        return 0

    record_evaluation(as_of_date, draw_row)
    window_start = as_of_date - dt.timedelta(days=WINDOW_DAYS - 1)
    rolling_rows = rows_between(upstream, window_start, as_of_date)
    if len(rolling_rows) < 240:
        raise RuntimeError(f"Only {len(rolling_rows)} draws in rolling window")
    write_parts(ROOT / "data" / "parts", rolling_rows)
    write_dataset_summary(ROOT / "data" / "dataset_summary.json", rolling_rows, window_start, as_of_date)
    run_statistics()

    next_date = as_of_date + dt.timedelta(days=1)
    generated_at = utc_now_iso()
    preview_path = ROOT / "output" / "next_preview.csv"
    prediction_rows, metrics = generate_prediction(
        rolling_rows,
        next_date,
        preview_path,
        {
            "forecast_kind": "post_draw_preview",
            "generated_at_utc": generated_at,
            "data_cutoff_date": as_of_date.isoformat(),
            "history_calendar_start": window_start.isoformat(),
            "history_calendar_end": as_of_date.isoformat(),
            "upstream_max_date_seen": upstream[-1]["date"] if upstream else "",
        },
    )
    append_model_run(as_of_date, next_date, metrics)
    write_rolling_summary()
    state = {
        "last_settled_draw": as_of_date.isoformat(),
        "next_preview_target": next_date.isoformat(),
        "observed_draws_in_window": len(rolling_rows),
        "selected_blend": metrics["selected_blend"],
        "updated_at_utc": generated_at,
    }
    (ROOT / "output" / "pipeline_state.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Settled draw {as_of_date}; rolling window contains {len(rolling_rows)} draws")
    print(f"Next preview target: {next_date}, blend={metrics['selected_blend']}")
    print("Preview top 10:")
    for row in prediction_rows[:10]:
        print(f"  {row['number']}: {row['probability']:.4%}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p_forecast = sub.add_parser("forecast", help="Create an immutable pre-draw forecast")
    p_forecast.add_argument("--target-date", help="YYYY-MM-DD; defaults to today in Vietnam")
    p_settle = sub.add_parser("settle", help="Evaluate observed draw and roll the dataset")
    p_settle.add_argument("--date", help="YYYY-MM-DD; defaults to today in Vietnam")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "forecast":
        target = dt.date.fromisoformat(args.target_date) if args.target_date else local_today()
        return forecast(target)
    if args.command == "settle":
        as_of = dt.date.fromisoformat(args.date) if args.date else local_today()
        return settle(as_of)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
