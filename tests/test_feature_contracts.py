from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.contracts import BILLTO_FEATURE_COLS, CATEGORY_FEATURE_COLS


def test_authoritative_feature_counts():
    assert len(BILLTO_FEATURE_COLS) == 29
    assert len(CATEGORY_FEATURE_COLS) == 44
    assert len(set(BILLTO_FEATURE_COLS)) == 29
    assert len(set(CATEGORY_FEATURE_COLS)) == 44
