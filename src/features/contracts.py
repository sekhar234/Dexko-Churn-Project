"""Authoritative local feature contracts mirrored from the latest churn config."""

CUSTOMER_COLUMN = "harmonizedsoldtocustomername"
CATEGORY_COLUMN = "mgrl1"

BILLTO_ID_COLS = [CUSTOMER_COLUMN, "bulevel1", "zipcode"]
CATEGORY_ID_COLS = [CUSTOMER_COLUMN, "bulevel1", CATEGORY_COLUMN, "zipcode"]
BILLTO_KEYS = BILLTO_ID_COLS + ["snapshot_dt"]
CATEGORY_KEYS = CATEGORY_ID_COLS + ["snapshot_dt"]

BILLTO_FEATURE_COLS = [
    "recency_days_all", "ratio_recency_to_cycle_all", "log_ratio_all",
    "median_cycle_days_all", "n_gaps_running_all", "active_months_12_all",
    "freq_0_30_all", "freq_181_365_all",
    "spend_0_30_all", "spend_181_365_all",
    "freq_ratio_recent_vs_prior_all", "freq_ratio_mid_vs_older_all",
    "spend_ratio_recent_vs_prior_all", "freq_share_0_30_in_90_all",
    "creditmax_current", "customergroupname_current", "bulevel1",
    "order_to_invoice_days_3m_all",
    "net_price_per_unit_3m_all", "net_price_per_unit_ratio_3m_all",
    "tenure_days_all", "distinct_mgrl1_365_all", "avg_order_value_365_all",
    "spend_volatility_6m_all",
    "spend_logratio_3m_all", "spend_logratio_6m_all",
    "freq_logratio_3m_all", "freq_logratio_6m_all",
    "snapshot_month_idx",
]

CATEGORY_FEATURE_COLS = [
    "recency_days", "ratio_recency_to_cycle", "log_ratio",
    "median_cycle_days", "n_gaps_running", "active_months_12_g",
    "order_to_invoice_days_3m_g",
    "net_price_per_unit_3m_g", "net_price_per_unit_ratio_3m_g",
    "freq_0_30_g", "freq_181_365_g",
    "spend_0_30_g", "spend_181_365_g",
    "freq_ratio_recent_vs_prior_g", "freq_ratio_mid_vs_older_g",
    "spend_ratio_recent_vs_prior_g", "freq_share_0_30_in_90_g",
    "spend_logratio_3m_g", "spend_logratio_6m_g",
    "freq_logratio_3m_g", "freq_logratio_6m_g",
    "recency_days_all", "ratio_recency_to_cycle_all",
    "median_cycle_days_all", "n_gaps_running_all", "active_months_12_all",
    "creditmax_current", "customergroupname_current",
    "order_to_invoice_days_3m_all",
    "net_price_per_unit_3m_all", "net_price_per_unit_ratio_3m_all",
    "tenure_days_all", "distinct_mgrl1_365_all", "avg_order_value_365_all",
    "spend_volatility_6m_all", "max_mgrl1_share_365_all",
    "spend_logratio_3m_all", "freq_logratio_3m_all",
    "spend_share_365", "spend_share_3m", "mix_shift_share_3m", "recency_delta",
    "mgrl1", "snapshot_month_idx",
]

# Validation uses the category-specific (non-account-context) subset.
CATEGORY_CORE_FEATURE_COLS = [
    "recency_days", "ratio_recency_to_cycle", "log_ratio",
    "median_cycle_days", "n_gaps_running", "active_months_12_g",
    "order_to_invoice_days_3m_g",
    "net_price_per_unit_3m_g", "net_price_per_unit_ratio_3m_g",
    "freq_0_30_g", "freq_181_365_g",
    "spend_0_30_g", "spend_181_365_g",
    "freq_ratio_recent_vs_prior_g", "freq_ratio_mid_vs_older_g",
    "spend_ratio_recent_vs_prior_g", "freq_share_0_30_in_90_g",
    "spend_logratio_3m_g", "spend_logratio_6m_g",
    "freq_logratio_3m_g", "freq_logratio_6m_g",
    "spend_share_365", "spend_share_3m", "mix_shift_share_3m", "recency_delta",
]

FREQ_SPEND_BANDS = [
    ("0_30", 0, 29),
    ("31_90", 30, 89),
    ("91_180", 90, 179),
    ("181_365", 180, 364),
]

DEFAULT_GROUP_MEDIAN_DAYS = 30.0
MIN_GAPS = 2
MIN_ACTIVE_MONTHS_12 = 2
