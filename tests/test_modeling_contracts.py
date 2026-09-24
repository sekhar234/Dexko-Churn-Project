import numpy as np
import pandas as pd

from src.features.contracts import CATEGORY_FEATURE_COLS
from src.modeling.contracts import CATEGORICAL_FEATURES, TEST_DAYS, VALID_DAYS
from src.modeling.pipeline import lift_at_fraction


def test_phase7_feature_contract():
    assert len(CATEGORY_FEATURE_COLS) == 44
    assert CATEGORICAL_FEATURES == ["customergroupname_current", "mgrl1"]
    assert VALID_DAYS == 90
    assert TEST_DAYS == 90


def test_lift_at_10pct_is_above_one_for_perfect_ranking():
    y = np.array([1] * 10 + [0] * 90)
    p = np.linspace(1.0, 0.0, 100)
    assert lift_at_fraction(y, p, 0.10) == 10.0
