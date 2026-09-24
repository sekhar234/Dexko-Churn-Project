from pathlib import Path
import json
import math
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.local_pipeline.io import read_df, resolve_table
from src.monitoring.contracts import (
    CRITICAL_DRIFT_FEATURES,
    MONITORED_CATEGORICAL_FEATURES,
    MONITORED_NUMERIC_FEATURES,
    PSI_STABLE_MAX,
    PSI_WATCH_MAX,
)

base = ROOT/"data"/"monitoring"
baseline = read_df(resolve_table(base/"dex_v2_drift_baseline"))
metrics = read_df(resolve_table(base/"dex_v2_drift_metrics"))
summary = read_df(resolve_table(base/"dex_v2_drift_summary"))
retrain = read_df(resolve_table(base/"dex_v2_retraining_decisions"))
feature_dist = read_df(resolve_table(base/"dex_v2_monitoring_feature_distributions"))
score_dist = read_df(resolve_table(base/"dex_v2_monitoring_score_distributions"))
scoring_history = read_df(resolve_table(base/"dex_v2_scoring_history"))

manifest = json.loads(
    (ROOT/"data"/"local_pipeline_manifest.json").read_text(encoding="utf-8")
)
monitoring = manifest["monitoring"]
latest = pd.to_datetime(metrics["scoring_snapshot"]).max()
latest_metrics = metrics[
    pd.to_datetime(metrics["scoring_snapshot"]) == latest
].copy()

expected_features = set(
    MONITORED_NUMERIC_FEATURES
    + MONITORED_CATEGORICAL_FEATURES
    + ["__SCORE_P14__"]
)
actual_features = set(latest_metrics["feature_name"].astype(str))

expected_status = latest_metrics["metric_value"].apply(
    lambda x: "STABLE"
    if x < PSI_STABLE_MAX
    else ("WATCH" if x < PSI_WATCH_MAX else "ACT")
)

performance_status = monitoring["performance"]["status"]
performance_ok = performance_status in {
    "EVALUATED",
    "BLOCKED_WAITING_FOR_LABELS",
    "BLOCKED_INSUFFICIENT_MATURE_ROWS",
}

checks = {
    "baseline_nonempty": bool(len(baseline) > 0),
    "all_monitored_features_present": bool(actual_features == expected_features),
    "metric_count_17": bool(len(latest_metrics) == 17),
    "psi_finite": bool(latest_metrics["metric_value"].map(math.isfinite).all()),
    "psi_nonnegative": bool((latest_metrics["metric_value"] >= 0).all()),
    "status_matches_thresholds": bool(
        (latest_metrics["status"].astype(str) == expected_status.astype(str)).all()
    ),
    "critical_features_present": bool(
        set(CRITICAL_DRIFT_FEATURES) <= actual_features
    ),
    "feature_distributions_present": bool(len(feature_dist) > 0),
    "score_distribution_present": bool(len(score_dist) > 0),
    "summary_has_latest": bool(
        latest in set(pd.to_datetime(summary["scoring_snapshot"]))
    ),
    "retraining_decision_has_latest": bool(
        latest in set(pd.to_datetime(retrain["scoring_snapshot"]))
    ),
    "automatic_retraining_disabled": bool(
        (retrain["automatic_retraining_enabled"] == False).all()
    ),
    "scoring_history_present": bool(len(scoring_history) > 0),
    "performance_state_valid": bool(performance_ok),
}

result = {
    "status": "PASS" if all(checks.values()) else "FAIL",
    "checks": checks,
    "snapshot_dt": str(latest.date()),
    "drift": monitoring["drift"],
    "retraining": monitoring["retraining"],
    "performance": monitoring["performance"],
}

out = ROOT/"data"/"qa"
out.mkdir(parents=True, exist_ok=True)
payload = json.dumps(result, indent=2)
(out/"monitoring_qa.json").write_text(payload, encoding="utf-8")
print(payload)

if result["status"] != "PASS":
    raise SystemExit(1)
