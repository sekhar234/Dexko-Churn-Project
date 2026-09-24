from pathlib import Path
import json, sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from src.local_pipeline.io import read_df, resolve_table
from src.features.contracts import BILLTO_FEATURE_COLS, BILLTO_KEYS, CATEGORY_KEYS, CATEGORY_CORE_FEATURE_COLS

bill=read_df(resolve_table(ROOT/"data"/"features"/"dex_v2_checkpoint_billto_panel_zipcode"))
cat=read_df(resolve_table(ROOT/"data"/"features"/"dex_v2_checkpoint_category_panel_zipcode"))

checks={
    "billto_nonempty": len(bill)>0,
    "category_nonempty": len(cat)>0,
    "billto_key_unique": int(bill.duplicated(BILLTO_KEYS).sum())==0,
    "category_key_unique": int(cat.duplicated(CATEGORY_KEYS).sum())==0,
    "billto_29_features_present": set(BILLTO_FEATURE_COLS).issubset(bill.columns),
    "category_core_features_present": set(CATEGORY_CORE_FEATURE_COLS).issubset(cat.columns),
    "no_labels_in_billto": not any(c.startswith("y_") for c in bill.columns),
    "no_labels_in_category": not any(c.startswith("y_") for c in cat.columns),
    "no_eligibility_in_category": not any("eligible" in c for c in cat.columns),
    "zip_preserved_billto": bill["zipcode"].notna().all(),
    "zip_preserved_category": cat["zipcode"].notna().all(),
    "snapshot_mondays_billto": bool((pd.to_datetime(bill["snapshot_dt"]).dt.dayofweek==0).all()),
    "snapshot_mondays_category": bool((pd.to_datetime(cat["snapshot_dt"]).dt.dayofweek==0).all()),
}
result={
    "status":"PASS" if all(checks.values()) else "FAIL",
    "checks":checks,
    "rows":{"billto":len(bill),"category":len(cat)},
    "snapshot_range":{
        "min":str(pd.to_datetime(cat["snapshot_dt"]).min().date()) if len(cat) else None,
        "max":str(pd.to_datetime(cat["snapshot_dt"]).max().date()) if len(cat) else None,
    },
}
out=ROOT/"data"/"qa"
out.mkdir(parents=True,exist_ok=True)
(out/"feature_engineering_qa.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))
if result["status"]!="PASS":
    raise SystemExit(1)
