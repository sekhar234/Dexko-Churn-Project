from pathlib import Path
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.local_pipeline.io import read_df, resolve_table
from src.features.contracts import CATEGORY_KEYS
from src.scoring.contracts import HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD

base = ROOT/"data"/"scoring"
scores = read_df(resolve_table(base/"dex_v2_category_loss_scores_wide_zipcode"))
viz = read_df(resolve_table(base/"dex_v2_category_loss_viz_latest_zipcode"))
reason_latest = read_df(resolve_table(base/"dex_v2_category_reason_latest_zipcode"))
reason_long = read_df(resolve_table(base/"dex_v2_category_reason_long_zipcode"))
rollup = read_df(resolve_table(base/"dex_v2_customer_zip_risk_rollup"))
reject = read_df(resolve_table(base/"dex_v2_weekly_reject_log"))

scores["snapshot_dt"] = pd.to_datetime(scores["snapshot_dt"])
latest = scores["snapshot_dt"].max()

expected_tier = pd.Series(
    pd.Categorical(
        pd.cut(
            scores["p_14"],
            bins=[-float("inf"), MEDIUM_RISK_THRESHOLD, HIGH_RISK_THRESHOLD, float("inf")],
            labels=["Low", "Medium", "High"],
            right=False,
        )
    ).astype(str),
    index=scores.index,
)

reason_counts = reason_long.groupby(CATEGORY_KEYS, dropna=False).size()
checks = {
    "scores_nonempty": bool(len(scores) > 0),
    "scores_key_unique": bool(int(scores.duplicated(CATEGORY_KEYS).sum()) == 0),
    "probability_in_range": bool(scores["p_14"].between(0, 1).all()),
    "risk_tiers_match_thresholds": bool((scores["risk_tier"].astype(str) == expected_tier).all()),
    "reason_latest_one_per_score": bool(len(reason_latest) == len(scores)),
    "viz_one_per_score": bool(len(viz) == len(scores)),
    "reason_long_has_all_scores": bool(len(reason_counts) == len(scores)),
    "reason_long_many_per_score": bool((reason_counts >= 3).all()),
    "rollup_nonempty": bool(len(rollup) > 0),
    "weighted_risk_in_range": bool(rollup["weighted_risk"].between(0, 1).all()),
    "risk_score_in_range": bool(rollup["risk_score"].between(0, 100).all()),
    "value_bands_valid": bool(set(rollup["value_band"].dropna().astype(str)) <= {"Low", "Mid", "High"}),
    "actions_present": bool(rollup["action"].notna().all()),
    "latest_snapshot_only": bool(scores["snapshot_dt"].nunique() == 1),
    "scored_plus_rejected_reconciles": bool(
        len(scores) + len(reject)
        == len(
            read_df(
                resolve_table(
                    ROOT/"data"/"labels"/"dex_v2_category_labeled_panel_zipcode"
                )
            ).query("snapshot_dt == @latest")
        )
    ),
}

result = {
    "status": "PASS" if all(checks.values()) else "FAIL",
    "checks": checks,
    "snapshot_dt": str(latest.date()),
    "rows": {
        "scores": int(len(scores)),
        "rejected": int(len(reject)),
        "rollup": int(len(rollup)),
        "reason_latest": int(len(reason_latest)),
        "reason_long": int(len(reason_long)),
    },
    "risk_counts": {
        str(k): int(v)
        for k, v in scores["risk_tier"].value_counts().sort_index().items()
    },
}

out = ROOT/"data"/"qa"
out.mkdir(parents=True, exist_ok=True)
payload = json.dumps(result, indent=2)
(out/"weekly_scoring_qa.json").write_text(payload, encoding="utf-8")
print(payload)

if result["status"] != "PASS":
    raise SystemExit(1)
