import datetime as dt

from prefix3_history import build_prefix3_stats
from xsmb_probability import PRIZE_LENGTHS


def make_draw(day, special="00000", prize1="00000"):
    row = {"date": day}
    for col, width in PRIZE_LENGTHS.items():
        row[col] = "0" * width
    row["special"] = special
    row["prize1"] = prize1
    return row


def test_best_prefix_for_two_digit_suffix_uses_historical_occurrences():
    draws = [
        make_draw(dt.date(2024, 1, 1), special="00562", prize1="00662"),
        make_draw(dt.date(2024, 1, 2), special="00562"),
        make_draw(dt.date(2024, 1, 3), special="00662", prize1="00562"),
    ]
    stats = {row["number"]: row for row in build_prefix3_stats(draws)}
    row = stats["62"]
    assert row["historical_best_prefix_digit"] == "5"
    assert row["historical_best_3digit"] == "562"
    assert row["historical_best_3digit_count"] == 3
    assert row["historical_best_3digit_hit_draws"] == 3


def test_tie_break_prefers_more_distinct_draw_hits():
    draws = [
        make_draw(dt.date(2024, 1, 1), special="00123", prize1="00123"),
        make_draw(dt.date(2024, 1, 2), special="00223"),
        make_draw(dt.date(2024, 1, 3), special="00223"),
    ]
    stats = {row["number"]: row for row in build_prefix3_stats(draws)}
    row = stats["23"]
    assert row["historical_best_3digit_count"] == 2
    assert row["historical_best_prefix_digit"] == "2"
    assert row["historical_best_3digit"] == "223"
    assert row["historical_best_3digit_hit_draws"] == 2
