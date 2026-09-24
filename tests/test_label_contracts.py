from src.labels.contracts import (
    HORIZON_DAYS,
    OBS_WINDOW_DAYS,
    LABEL_MATURITY_DAYS,
    MIN_GAPS,
    MIN_ACTIVE_MONTHS_12,
    SCORING_RECENCY_MAX_DAYS,
)


def test_phase6_label_contract():
    assert HORIZON_DAYS == 14
    assert OBS_WINDOW_DAYS == 180
    assert LABEL_MATURITY_DAYS == 194
    assert MIN_GAPS == 2
    assert MIN_ACTIVE_MONTHS_12 == 2
    assert SCORING_RECENCY_MAX_DAYS == 90
