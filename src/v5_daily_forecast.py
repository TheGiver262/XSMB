#!/usr/bin/env python3
"""Generate and immutably lock the daily V5 set10 forecast.

This wrapper is intended for the pre-draw scheduler. It refreshes upstream data,
runs the frozen V5 research pipeline, validates that the generated target is
today in Vietnam time and still before the draw lock deadline, then creates a
write-once dated snapshot under forecasts/v5_set10/YYYY-MM-DD.json.

Existing dated snapshots are never overwritten. A rerun is accepted only when
the existing snapshot has the same target date, data cutoff, model, scheme and
top10; otherwise the job fails loudly.
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
FORECAST = ROOT / "forecasts"
NEXT_PATH = FORECAST / "set10_v5_next.json"
DATED_DIR = FORECAST / "v5_set10"
TZ = ZoneInfo("Asia/Ho_Chi_Minh")
LOCK_TIME = dt.time(18, 0)


def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def stable_identity(payload: dict) -> dict:
    return {
        "target_date": payload.get("target_date"),
        "data_cutoff": payload.get("data_cutoff"),
        "model": payload.get("model"),
        "selected_scheme": payload.get("selected_scheme"),
        "top10": payload.get("top10"),
        "strict_success_rule": payload.get("strict_success_rule"),
    }


def main() -> None:
    now = dt.datetime.now(TZ)
    today = now.date()
    deadline = dt.datetime.combine(today, LOCK_TIME, tzinfo=TZ)
    if now >= deadline:
        raise RuntimeError(
            f"Refusing V5 daily forecast at {now.isoformat()}: "
            f"daily lock deadline is {deadline.isoformat()}"
        )

    run("src/sync_upstream.py")
    run("src/walkforward_set10_v5.py")

    if not NEXT_PATH.exists():
        raise RuntimeError(f"V5 pipeline did not create {NEXT_PATH}")
    forecast = json.loads(NEXT_PATH.read_text(encoding="utf-8"))

    target = forecast.get("target_date")
    if target != today.isoformat():
        raise RuntimeError(
            "V5 target mismatch after upstream refresh: "
            f"expected {today.isoformat()}, got {target}. "
            "Do not backfill or lock a forecast for another date."
        )

    generated_at = dt.datetime.fromisoformat(forecast["generated_at_ict"])
    if generated_at.tzinfo is None:
        raise RuntimeError("generated_at_ict must be timezone-aware")
    if generated_at.astimezone(TZ) >= deadline:
        raise RuntimeError("Generated V5 forecast is not pre-draw valid")

    if len(forecast.get("top10", [])) != 10 or len(set(forecast["top10"])) != 10:
        raise RuntimeError("V5 forecast must contain exactly 10 distinct suffixes")
    for suffix in forecast["top10"]:
        if len(suffix) != 2 or not suffix.isdigit():
            raise RuntimeError(f"Invalid two-digit suffix serialization: {suffix!r}")

    DATED_DIR.mkdir(parents=True, exist_ok=True)
    dated_path = DATED_DIR / f"{target}.json"
    if dated_path.exists():
        existing = json.loads(dated_path.read_text(encoding="utf-8"))
        if stable_identity(existing) != stable_identity(forecast):
            raise RuntimeError(
                f"Immutable V5 forecast already exists with different content: {dated_path}"
            )
        print(f"V5 forecast already locked and unchanged: {dated_path}")
        return

    # Exclusive creation is the final write-once guard.
    with dated_path.open("x", encoding="utf-8") as handle:
        json.dump(forecast, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Locked prospective V5 forecast: {dated_path}")
    print("top10=" + "-".join(forecast["top10"]))


if __name__ == "__main__":
    main()
