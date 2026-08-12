#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import daily_pipeline as base

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "challenger"


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def append_unique(path: Path, row: dict):
    existing = []
    if path.exists():
        existing = read_csv(path)
        if any(r.get("date") == str(row["date"]) for r in existing):
            print(f"Already settled: {row['date']}")
            return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(row)
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        if not existing:
            w.writeheader()
        w.writerow(row)
    return True


def settle(target: dt.date):
    upstream = base.fetch_upstream()
    draw = next((r for r in upstream if r["date"] == target.isoformat()), None)
    if draw is None:
        raise RuntimeError(f"Draw {target} is not available upstream yet")

    fprob = PAPER / "forecasts" / f"{target.isoformat()}_probability.csv"
    frank = PAPER / "forecasts" / f"{target.isoformat()}_ranking.csv"
    if not fprob.exists() or not frank.exists():
        raise RuntimeError(f"Missing immutable challenger forecast for {target}")

    prob_metrics = base.evaluate_forecast_rows(read_csv(fprob), draw)
    ranking = sorted(read_csv(frank), key=lambda r: int(r["rank"]))
    actual = base.actual_presence(draw)

    def top_hits(k: int):
        picked = [f"{int(r['number']):02d}" for r in ranking[:k]]
        hits = [n for n in picked if n in actual]
        return picked, hits

    p1, h1 = top_hits(1)
    p3, h3 = top_hits(3)
    p5, h5 = top_hits(5)
    row = {
        "date": target.isoformat(),
        "status": "evaluated",
        "brier": prob_metrics["brier"],
        "baseline_brier": prob_metrics["baseline_brier"],
        "brier_improvement": prob_metrics["brier_improvement"],
        "log_loss": prob_metrics["log_loss"],
        "baseline_log_loss": prob_metrics["baseline_log_loss"],
        "log_loss_improvement": prob_metrics["log_loss_improvement"],
        "actual_unique_numbers": prob_metrics["actual_unique_numbers"],
        "top1_hits": len(h1),
        "top1_pick": " ".join(p1),
        "top1_actual_hits": " ".join(h1),
        "top3_hits": len(h3),
        "top3_picks": " ".join(p3),
        "top3_actual_hits": " ".join(h3),
        "top5_hits": len(h5),
        "top5_picks": " ".join(p5),
        "top5_actual_hits": " ".join(h5),
        "number62_hit": int("62" in actual),
    }
    append_unique(PAPER / "evaluation" / "daily.csv", row)
    write_summary()
    print(json.dumps(row, ensure_ascii=False, indent=2))


def write_summary():
    path = PAPER / "evaluation" / "daily.csv"
    if not path.exists():
        return
    rows = [r for r in read_csv(path) if r.get("status") == "evaluated"]
    if not rows:
        return
    n = len(rows)
    avg = lambda key: sum(float(r[key]) for r in rows) / n
    summary = {
        "evaluated_days": n,
        "start_date": rows[0]["date"],
        "end_date": rows[-1]["date"],
        "mean_brier": avg("brier"),
        "mean_baseline_brier": avg("baseline_brier"),
        "brier_improvement": avg("baseline_brier") - avg("brier"),
        "mean_log_loss": avg("log_loss"),
        "mean_baseline_log_loss": avg("baseline_log_loss"),
        "log_loss_improvement": avg("baseline_log_loss") - avg("log_loss"),
        "top1_hit_rate": avg("top1_hits"),
        "top3_hit_rate_per_pick": sum(int(r["top3_hits"]) for r in rows) / (3 * n),
        "top5_hit_rate_per_pick": sum(int(r["top5_hits"]) for r in rows) / (5 * n),
        "brier_win_days": sum(float(r["brier_improvement"]) > 0 for r in rows),
        "logloss_win_days": sum(float(r["log_loss_improvement"]) > 0 for r in rows),
        "note": "Paper-trading diagnostics only; sample is intentionally immutable and must not be used to retune the same day's forecast.",
    }
    out = PAPER / "evaluation" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    args = ap.parse_args()
    target = dt.date.fromisoformat(args.date) if args.date else base.local_today()
    settle(target)


if __name__ == "__main__":
    main()
