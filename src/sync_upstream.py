#!/usr/bin/env python3
"""Mirror canonical XSMB CSV datasets with append-only integrity checks."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://raw.githubusercontent.com/khiemdoan/vietnam-lottery-xsmb-analysis/refs/heads/main/data"
FILES = ["xsmb.csv", "xsmb-2-digits.csv", "xsmb-sparse.csv"]
DEST = ROOT / "data" / "upstream"
MAX_FILE_BYTES = 10 * 1024 * 1024
LOCAL_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

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
RAW_FIELDS = ["date", *PRIZE_LENGTHS]


def download(name: str) -> bytes:
    req = urllib.request.Request(
        f"{BASE}/{name}", headers={"User-Agent": "xsmb-research-mirror/1.0"}
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        data = response.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"Upstream file {name} exceeds {MAX_FILE_BYTES} bytes")
    return data


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decode_rows(data: bytes, name: str) -> tuple[list[dict[str, str]], list[str]]:
    text = data.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or "date" not in reader.fieldnames:
        raise ValueError(f"{name} does not contain a date column")
    rows = list(reader)
    dates: list[str] = []
    today = dt.datetime.now(LOCAL_TZ).date()
    for row in rows:
        date_text = (row.get("date") or "")[:10]
        try:
            parsed = dt.date.fromisoformat(date_text)
        except ValueError as exc:
            raise ValueError(f"Invalid date in {name}: {date_text!r}") from exc
        if parsed > today:
            raise ValueError(f"Future draw date in {name}: {date_text}")
        dates.append(date_text)
    if dates != sorted(dates):
        raise ValueError(f"Upstream dates are not sorted in {name}")
    if len(dates) != len(set(dates)):
        raise ValueError(f"Duplicate dates in upstream dataset {name}")
    return rows, dates


def normalized_raw_rows(data: bytes) -> list[tuple[str, ...]]:
    rows, _dates = decode_rows(data, "xsmb.csv")
    if not rows:
        raise ValueError("xsmb.csv is empty")
    fieldnames = set(rows[0])
    missing = [field for field in RAW_FIELDS if field not in fieldnames]
    if missing:
        raise ValueError(f"xsmb.csv is missing expected columns: {missing}")

    normalized: list[tuple[str, ...]] = []
    for row in rows:
        date_text = (row.get("date") or "")[:10]
        values: list[str] = [date_text]
        for col, width in PRIZE_LENGTHS.items():
            value = (row.get(col) or "").strip()
            if not value or not value.isdigit() or len(value) > width:
                raise ValueError(f"Invalid {col} value on {date_text}: {value!r}")
            values.append(value.zfill(width))
        normalized.append(tuple(values))
    return normalized


def validate_append_only(existing: bytes, incoming: bytes) -> None:
    """Reject deletion or mutation of any previously accepted historical draw."""
    old_rows = normalized_raw_rows(existing)
    new_rows = normalized_raw_rows(incoming)
    if len(new_rows) < len(old_rows):
        raise ValueError("Upstream history shrank; refusing mirror update")
    if new_rows[: len(old_rows)] != old_rows:
        for index, (old, new) in enumerate(zip(old_rows, new_rows)):
            if old != new:
                raise ValueError(
                    f"Historical upstream mutation at row {index + 1} / date {old[0]}"
                )
        raise ValueError("Historical upstream prefix changed; refusing mirror update")


def validate_payloads(payloads: dict[str, bytes]) -> list[str]:
    date_sets: dict[str, list[str]] = {}
    for name, payload in payloads.items():
        _rows, dates = decode_rows(payload, name)
        date_sets[name] = dates
    raw_dates = date_sets["xsmb.csv"]
    normalized_raw_rows(payloads["xsmb.csv"])
    for name, dates in date_sets.items():
        if dates != raw_dates:
            raise ValueError(f"Date mismatch between xsmb.csv and {name}")
    return raw_dates


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    payloads = {name: download(name) for name in FILES}
    raw_dates = validate_payloads(payloads)

    existing_raw = DEST / "xsmb.csv"
    if existing_raw.exists() and existing_raw.stat().st_size:
        validate_append_only(existing_raw.read_bytes(), payloads["xsmb.csv"])

    for name, payload in payloads.items():
        (DEST / name).write_bytes(payload)

    meta = {
        "source_repository": "khiemdoan/vietnam-lottery-xsmb-analysis",
        "source_ref": "refs/heads/main",
        "integrity_policy": "append-only historical xsmb.csv; cross-file date equality; schema/value checks",
        "license": "MIT",
        "first_date": raw_dates[0],
        "last_date": raw_dates[-1],
        "observed_draws": len(raw_dates),
        "files": {
            name: {"sha256": sha(payload), "bytes": len(payload)}
            for name, payload in payloads.items()
        },
    }
    (DEST / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
