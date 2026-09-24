"""Phase 9 monitoring contracts."""

PSI_STABLE_MAX = 0.10
PSI_WATCH_MAX = 0.25
PSI_EPSILON = 1e-6
PSI_NUMERIC_BINS = 10

LABEL_MATURITY_DAYS = 194
MIN_PERFORMANCE_ROWS = 100
DRIFT_CONFIRMATION_RUNS = 2

CRITICAL_DRIFT_FEATURES = [
    "recency_days",
    "median_cycle_days",
    "spend_0_30_g",
]

# The frozen production documentation confirms 14 numeric + 2 categorical
# monitored inputs, but does not enumerate the complete 14-name list in the
# surfaced baseline. This offline contract uses the key behavioral inputs from
# the approved 44-feature model and preserves all confirmed critical features.
MONITORED_NUMERIC_FEATURES = [
    "recency_days",
    "median_cycle_days",
    "ratio_recency_to_cycle",
    "n_gaps_running",
    "active_months_12_g",
    "freq_0_30_g",
    "spend_0_30_g",
    "spend_logratio_3m_g",
    "freq_logratio_3m_g",
    "net_price_per_unit_ratio_3m_g",
    "order_to_invoice_days_3m_g",
    "recency_days_all",
    "active_months_12_all",
    "spend_logratio_3m_all",
]

MONITORED_CATEGORICAL_FEATURES = [
    "customergroupname_current",
    "mgrl1",
]
