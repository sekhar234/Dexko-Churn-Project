import numpy as np

from src.monitoring.contracts import (
    CRITICAL_DRIFT_FEATURES,
    LABEL_MATURITY_DAYS,
    MONITORED_CATEGORICAL_FEATURES,
    MONITORED_NUMERIC_FEATURES,
)
from src.monitoring.pipeline import _psi, _status


def test_phase9_monitoring_contract():
    assert LABEL_MATURITY_DAYS == 194
    assert len(MONITORED_NUMERIC_FEATURES) == 14
    assert len(MONITORED_CATEGORICAL_FEATURES) == 2
    assert CRITICAL_DRIFT_FEATURES == [
        "recency_days",
        "median_cycle_days",
        "spend_0_30_g",
    ]


def test_phase9_psi_thresholds():
    assert _status(0.0999) == "STABLE"
    assert _status(0.10) == "WATCH"
    assert _status(0.2499) == "WATCH"
    assert _status(0.25) == "ACT"


def test_phase9_identical_distribution_has_zero_psi():
    p = np.array([0.2, 0.3, 0.5])
    assert abs(_psi(p, p)) < 1e-12
