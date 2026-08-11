import datetime as dt
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from xsmb_probability import P0, load_draws, matrices, next_probabilities


def test_baseline():
    assert 0.237 < P0 < 0.238


def test_data_shape():
    draws = load_draws(ROOT / 'data' / 'parts')
    presence, counts = matrices(draws)
    assert len(draws) == 722
    assert draws[0]['date'] == dt.date(2024, 8, 11)
    assert draws[-1]['date'] == dt.date(2026, 8, 10)
    assert len({d['date'] for d in draws}) == len(draws)
    assert presence.shape == (722, 100)
    assert counts.shape == (722, 100)
    assert (counts.sum(axis=1) == 27).all()


def test_prediction_bounds():
    draws = load_draws(ROOT / 'data' / 'parts')
    rows, metrics = next_probabilities(draws, dt.date(2026, 8, 11))
    assert len(rows) == 100
    assert all(0 < r['probability'] < 1 for r in rows)
    assert 0 <= metrics['selected_blend'] <= 1
    assert metrics['observed_draws'] == 722
