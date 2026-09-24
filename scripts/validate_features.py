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
    "billto_nonempty": bool(len(bill)>0),
    "category_nonempty": bool(len(cat)>0),
    "billto_key_unique": bool(int(bill.duplicated(BILLTO_KEYS).sum())==0),
    "category_key_unique": bool(int(cat.duplicated(CATEGORY_KEYS).sum())==0),
    "billto_29_features_present": bool(set(BILLTO_FEATURE_COLS).issubset(bill.columns)),
    "category_core_features_present": bool(set(CATEGORY_CORE_FEATURE_COLS).issubset(cat.columns)),
    "no_labels_in_billto": bool(not any(c.startswith("y_") for c in bill.columns)),
    "no_labels_in_category": bool(not any(c.startswith("y_") for c in cat.columns)),
    "no_eligibility_in_category": bool(not any("eligible" in c for c in cat.columns)),
    "zip_preserved_billto": bool(bill["zipcode"].notna().all()),
    "zip_preserved_category": bool(cat["zipcode"].notna().all()),
    "snapshot_mondays_billto": bool((pd.to_datetime(bill["snapshot_dt"]).dt.dayofweek==0).all()),
    "snapshot_mondays_category": bool((pd.to_datetime(cat["snapshot_dt"]).dt.dayofweek==0).all()),
}

result={
    "status":"PASS" if all(checks.values()) else "FAIL",
    "checks":checks,
    "rows":{"billto":int(len(bill)),"category":int(len(cat))},
    "snapshot_range":{
        "min":str(pd.to_datetime(cat["snapshot_dt"]).min().date()) if len(cat) else None,
        "max":str(pd.to_datetime(cat["snapshot_dt"]).max().date()) if len(cat) else None,
    },
}

out=ROOT/"data"/"qa"
out.mkdir(parents=True,exist_ok=True)
payload=json.dumps(result,indent=2)
(out/"feature_engineering_qa.json").write_text(payload,encoding="utf-8")
print(payload)

if result["status"]!="PASS":
    raise SystemExit(1)
