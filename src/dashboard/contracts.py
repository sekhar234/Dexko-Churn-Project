"""Phase 11 dashboard contracts."""

REQUIRED_TABLES = [
    "category_scores",
    "customer_rollup",
    "reason_long",
    "scoring_history",
    "drift_metrics",
    "drift_summary",
    "retraining_decisions",
    "cdata_output",
    "customer_attrs",
    "peak_months",
    "top_categories",
    "reject_log",
    "dashboard_metadata",
    "pipeline_stage_status",
]

OPTIONAL_TABLES = [
    "performance_history",
]

DASHBOARD_PAGES = [
    "Data Upload",
    "Executive Overview",
    "Customer Workbench",
    "Customer 360",
    "Category Intelligence",
    "Risk Movement",
    "Model Health",
    "Pipeline Health",
]
