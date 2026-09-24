"""Phase 8 scoring contracts."""

HIGH_RISK_THRESHOLD = 0.40
MEDIUM_RISK_THRESHOLD = 0.20
TOP_REASON_COUNT = 3

ACTION_GRID = {
    ("High", "High"): "Urgent Save",
    ("High", "Medium"): "Protect",
    ("High", "Low"): "Monitor",
    ("Mid", "High"): "Retain",
    ("Mid", "Medium"): "Engage",
    ("Mid", "Low"): "Maintain",
    ("Low", "High"): "Evaluate",
    ("Low", "Medium"): "Watch",
    ("Low", "Low"): "Automate",
}

NON_ACTIONABLE_REASON_FEATURES = {
    "customergroupname_current",
    "mgrl1",
}
