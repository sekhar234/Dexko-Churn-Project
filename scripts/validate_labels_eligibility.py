from pathlib import Path
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.local_pipeline.io import read_df, resolve_table
from src.features.contracts import CATEGORY_KEYS, BILLTO_KEYS
from src.labels.contracts import LABEL_MATURITY_DAYS

cat = read_df(resolve_table(ROOT/"data"/"labels"/"dex_v2_category_labeled_panel_zipcode"))
bill = read_df(resolve_table(ROOT/"data"/"labels"/"dex_v2_billto_labeled_panel_zipcode"))

cat["snapshot_dt"] = pd.to_datetime(cat["snapshot_dt"])
bill["snapshot_dt"] = pd.to_datetime(bill["snapshot_dt"])

latest = cat["snapshot_dt"].max()
mature_cutoff = latest - pd.Timedelta(days=LABEL_MATURITY_DAYS)
latest_cat = cat[cat["snapshot_dt"] == latest]
train_labeled = cat[cat["y_cat_activeacct_14"].notna()]

checks = {
    "category_nonempty": bool(len(cat) > 0),
    "billto_nonempty": bool(len(bill) > 0),
    "category_key_unique": bool(int(cat.duplicated(CATEGORY_KEYS).sum()) == 0),
    "billto_key_unique": bool(int(bill.duplicated(BILLTO_KEYS).sum()) == 0),
    "target_binary_or_null": bool(cat["y_cat_activeacct_14"].dropna().isin([0, 1]).all()),
    "latest_target_unresolved": bool(latest_cat["y_cat_activeacct_14"].isna().all()),
    "latest_training_not_eligible": bool((latest_cat["train_eligible_14"] == 0).all()),
    "latest_has_score_eligible_rows": bool((latest_cat["score_eligible_cat_14"] == 1).any()),
    "labels_only_on_mature_rows": bool((train_labeled["snapshot_dt"] <= mature_cutoff).all()),
    "labels_require_active_account": bool((train_labeled["account_retained_post_horizon_14"] == 1).all()),
    "positive_means_category_not_retained": bool(
        (train_labeled.loc[
            train_labeled["y_cat_activeacct_14"] == 1,
            "category_retained_post_horizon_14",
        ] == 0).all()
    ),
    "negative_means_category_retained": bool(
        (train_labeled.loc[
            train_labeled["y_cat_activeacct_14"] == 0,
            "category_retained_post_horizon_14",
        ] == 1).all()
    ),
    "score_gate_independent_of_label_maturity": bool(
        (
            (latest_cat["score_eligible_cat_14"] == 1)
            & (latest_cat["gate_label_observable_14"] == 0)
        ).any()
    ),
    "eligible_cat_is_training_alias": bool(
        (latest_cat["eligible_cat_14"] == latest_cat["train_eligible_14"]).all()
    ),
}

label_counts = {
    str(int(k)): int(v)
    for k, v in train_labeled["y_cat_activeacct_14"].astype(int).value_counts().sort_index().items()
}

result = {
    "status": "PASS" if all(checks.values()) else "FAIL",
    "checks": checks,
    "latest_snapshot": str(latest.date()),
    "mature_cutoff": str(mature_cutoff.date()),
    "rows": {
        "category": int(len(cat)),
        "billto": int(len(bill)),
        "training_labeled_category": int(len(train_labeled)),
        "latest_score_eligible_category": int((latest_cat["score_eligible_cat_14"] == 1).sum()),
    },
    "label_counts": label_counts,
}

out = ROOT/"data"/"qa"
out.mkdir(parents=True, exist_ok=True)
payload = json.dumps(result, indent=2)
(out/"labels_eligibility_qa.json").write_text(payload, encoding="utf-8")
print(payload)

if result["status"] != "PASS":
    raise SystemExit(1)
