import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from xsmb_probability_v2 import FEATURE_SETS
from xsmb_3digit import SPECS

def test_two_digit_feature_sets_nested():
    assert set(FEATURE_SETS['long_only']).issubset(FEATURE_SETS['recency'])
    assert set(FEATURE_SETS['recency']).issubset(FEATURE_SETS['recency_gap'])
    assert set(FEATURE_SETS['recency_gap']).issubset(FEATURE_SETS['full'])

def test_three_digit_targets_distinct():
    assert SPECS['suffix3_any'].positions==23
    assert SPECS['g6_exact'].positions==3
    assert SPECS['suffix3_any'].baseline>SPECS['g6_exact'].baseline
