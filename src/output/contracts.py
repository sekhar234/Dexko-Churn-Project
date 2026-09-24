"""Phase 10 business-output contracts."""

CALL_LIST_COLUMNS = [
    "Customer ID",
    "Customer",
    "Customer Group",
    "Ship-To",
    "Branch",
    "Sales Rep",
    "Category",
    "Risk Score",
    "Risk Tier",
    "Action",
    "Annual Spend",
    "Rev at Risk",
    "Who They Are",
    "Bold Signals",
    "Grey Signals",
    "Scoring Run ID",
]

VALID_ACTIONS = {
    "Urgent Save",
    "Protect",
    "Monitor",
    "Retain",
    "Engage",
    "Maintain",
    "Evaluate",
    "Watch",
    "Automate",
}

RISK_TIER_ORDER = ["High", "Medium", "Low"]

COLUMN_DESCRIPTIONS = [
    ("Customer ID", "Customer identifier(s) propagated from the certified customer attributes table."),
    ("Customer", "Harmonized sold-to customer name."),
    ("Customer Group", "Current customer group used by the churn pipeline: OEM or Stocking Dealer."),
    ("Ship-To", "Normalized ship-to ZIP code."),
    ("Branch", "Servicing branch carried from the certified customer attributes."),
    ("Sales Rep", "Synthetic/local sales representative. This is not the external SharePoint Account Owner mapping."),
    ("Category", "Product category (mgrl1) scored by the H14 churn model."),
    ("Risk Score", "Category churn probability p_14 multiplied by 100 and rounded to an integer."),
    ("Risk Tier", "High >= 0.40, Medium >= 0.20 and < 0.40, Low < 0.20."),
    ("Action", "3x3 business action derived from customer value band and category risk tier."),
    ("Annual Spend", "Trailing 365-day spend for this scored customer x ZIP x category."),
    ("Rev at Risk", "Category Annual Spend multiplied by p_14."),
    ("Who They Are", "Customer group, tenure, credit limit, peak buying months, and top annual-spend categories."),
    ("Bold Signals", "Strongest SHAP contributors that increase modeled churn risk."),
    ("Grey Signals", "Additional SHAP context, including risk-lowering or secondary contributors."),
    ("Scoring Run ID", "Local Phase 8 scoring lineage identifier."),
]
