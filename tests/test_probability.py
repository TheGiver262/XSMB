import datetime as dt
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from xsmb_probability import P0, load_draws, matrices, next_probabilities


def dataset_summary():
    return json.loads((ROOT / 'data' / 'dataset_summary.json').read_text(encoding='utf-8'))


def test_baseline():
    assert 0.237 < P0 < 0.238


def test_data_shape():
    draws = load_draws(ROOT / 'data' / 'parts')
    presence, counts = matrices(draws)
    summary = dataset_summary()
    expected = int(summary['observed_draws'])
    assert len(draws) == expected
    assert draws[0]['date'] >= dt.date.fromisoformat(summary['window_start'])
    assert draws[-1]['date'] == dt.date.fromisoformat(summary['window_end'])
    assert len({d['date'] for d in draws}) == len(draws)
    assert presence.shape == (expected, 100)
    assert counts.shape == (expected, 100)
    assert (counts.sum(axis=1) == 27).all()


def test_prediction_bounds():
    draws = load_draws(ROOT / 'data' / 'parts')
    target_date = draws[-1]['date'] + dt.timedelta(days=1)
    rows, metrics = next_probabilities(draws, target_date)
    assert len(rows) == 100
    assert all(0 < r['probability'] < 1 for r in rows)
    assert 0 <= metrics['selected_blend'] <= 1
    assert metrics['observed_draws'] == len(draws)
