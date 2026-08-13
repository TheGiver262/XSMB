import csv
import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import sync_upstream


def raw_csv(rows):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=sync_upstream.RAW_FIELDS, lineterminator="\n")
    writer.writeheader()
    for date_text, seed in rows:
        row = {"date": date_text}
        for index, (col, width) in enumerate(sync_upstream.PRIZE_LENGTHS.items()):
            row[col] = str((seed + index) % (10**width)).zfill(width)
        writer.writerow(row)
    return buf.getvalue().encode()


def test_append_only_accepts_new_draws():
    old = raw_csv([("2026-08-10", 1), ("2026-08-11", 2)])
    new = raw_csv([("2026-08-10", 1), ("2026-08-11", 2), ("2026-08-12", 3)])
    sync_upstream.validate_append_only(old, new)


def test_append_only_rejects_historical_mutation():
    old = raw_csv([("2026-08-10", 1), ("2026-08-11", 2)])
    mutated = raw_csv([("2026-08-10", 99), ("2026-08-11", 2), ("2026-08-12", 3)])
    with pytest.raises(ValueError, match="Historical upstream mutation"):
        sync_upstream.validate_append_only(old, mutated)


def test_append_only_rejects_history_shrink():
    old = raw_csv([("2026-08-10", 1), ("2026-08-11", 2)])
    shrunk = raw_csv([("2026-08-10", 1)])
    with pytest.raises(ValueError, match="history shrank"):
        sync_upstream.validate_append_only(old, shrunk)


def test_decode_rows_rejects_future_draw():
    payload = b"date,value\n2999-01-01,1\n"
    with pytest.raises(ValueError, match="Future draw date"):
        sync_upstream.decode_rows(payload, "future.csv")
