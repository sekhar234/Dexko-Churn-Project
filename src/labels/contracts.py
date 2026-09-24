"""Phase 6 label and eligibility contracts."""

HORIZON_DAYS = 14
OBS_WINDOW_DAYS = 180
LABEL_MATURITY_DAYS = HORIZON_DAYS + OBS_WINDOW_DAYS

MIN_GAPS = 2
MIN_ACTIVE_MONTHS_12 = 2
SCORING_RECENCY_MAX_DAYS = 90

CATEGORY_TRAIN_GATES = [
    "gate_label_observable_14",
    "gate_has_last_purchase_14",
    "gate_observed_cycle_14",
    "gate_min_gaps_14",
    "gate_min_active_months_14",
]

CATEGORY_SCORE_GATES = [
    "score_gate_has_last_purchase_14",
    "score_gate_recency_lt_90_14",
    "score_gate_min_gaps_14",
    "score_gate_min_active_months_14",
]

BILLTO_TRAIN_GATES = [
    "gate_billto_label_observable_14",
    "gate_billto_has_last_purchase_14",
    "gate_billto_observed_cycle_14",
    "gate_billto_min_gaps_14",
    "gate_billto_min_active_months_14",
]

BILLTO_SCORE_GATES = [
    "score_gate_billto_has_last_purchase_14",
    "score_gate_billto_recency_lt_90_14",
    "score_gate_billto_min_gaps_14",
    "score_gate_billto_min_active_months_14",
]
