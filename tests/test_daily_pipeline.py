import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import daily_pipeline as pipeline


def synthetic_draw(date="2026-08-11"):
    row = {"date": date}
    for i, (col, width) in enumerate(pipeline.PRIZE_LENGTHS.items()):
        row[col] = str(i % 100).zfill(width)
    return row


def test_rows_between_and_missing_dates():
    rows = [synthetic_draw("2026-08-09"), synthetic_draw("2026-08-11")]
    got = pipeline.rows_between(rows, dt.date(2026, 8, 9), dt.date(2026, 8, 11))
    assert len(got) == 2
    assert pipeline.missing_dates(dt.date(2026, 8, 9), dt.date(2026, 8, 11), got) == ["2026-08-10"]


def test_evaluate_perfect_forecast_beats_baseline():
    draw = synthetic_draw()
    actual = pipeline.actual_presence(draw)
    forecasts = []
    for n in range(100):
        number = f"{n:02d}"
        forecasts.append({"number": number, "probability": "0.99" if number in actual else "0.01"})
    metrics = pipeline.evaluate_forecast_rows(forecasts, draw)
    assert metrics["brier"] < metrics["baseline_brier"]
    assert metrics["log_loss"] < metrics["baseline_log_loss"]
    assert metrics["actual_unique_numbers"] == len(actual)


def test_parse_upstream_rejects_duplicate_date():
    header = ",".join(pipeline.RAW_FIELDS)
    values = ["2026-08-11"] + ["1"] * len(pipeline.PRIZE_COLS)
    text = header + "\n" + ",".join(values) + "\n" + ",".join(values) + "\n"
    try:
        pipeline.parse_upstream_csv(text)
    except ValueError as exc:
        assert "Duplicate" in str(exc)
    else:
        raise AssertionError("duplicate date should fail")
