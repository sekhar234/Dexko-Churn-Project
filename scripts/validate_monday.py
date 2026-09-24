from pathlib import Path
import json, sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.local_pipeline.io import read_df, resolve_table

base=read_df(resolve_table(ROOT/"data"/"certified"/"dex_v2_base"))
attrs=read_df(resolve_table(ROOT/"data"/"certified"/"dex_v2_cust_attrs"))
funnel=read_df(resolve_table(ROOT/"data"/"reports"/"load_funnel"))
registry=read_df(resolve_table(ROOT/"data"/"edw_cache"/"snapshot_registry"))

checks={
    "latest_snapshot_success": bool(((registry["dataset_name"].astype(str)=="master_weekly") & (registry["status"].astype(str)=="SUCCESS")).any()),
    "base_nonempty": len(base)>0,
    "base_all_dex": bool(base["dataareaid"].astype(str).str.strip().str.lower().eq("dex").all()),
    "base_positive_revenue": bool((pd.to_numeric(base["netamountextended"],errors="coerce")>0).all()),
    "base_positive_quantity": bool((pd.to_numeric(base["quantity"],errors="coerce")>0).all()),
    "base_dates_valid": bool(pd.to_datetime(base["invoicedate"],errors="coerce").notna().all()),
    "attrs_unique_customer_zip": int(attrs.duplicated(["harmonizedsoldtocustomername","zipcode"]).sum())==0,
    "attrs_scope_only": bool(attrs["customergroupname"].astype(str).str.upper().isin(["OEM","STOCKING DEALER"]).all()),
    "funnel_monotonic": bool((funnel["rows"].diff().fillna(0)<=0).all()),
}
result={
    "status":"PASS" if all(checks.values()) else "FAIL",
    "checks":checks,
    "rows":{"dex_v2_base":len(base),"dex_v2_cust_attrs":len(attrs)},
    "funnel":funnel.to_dict("records"),
}
qa=ROOT/"data"/"qa"
qa.mkdir(parents=True,exist_ok=True)
(qa/"local_monday_qa.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")

lines=[
    "# Local Monday Pipeline QA",
    "",
    f"**Status:** {result['status']}",
    "",
    "## Checks",
]
lines += [f"- {'PASS' if v else 'FAIL'} — {k}" for k,v in checks.items()]
lines += ["","## Funnel","","| Stage | Rows | Sales | Customers | ZIPs |","|---|---:|---:|---:|---:|"]
for r in result["funnel"]:
    sales="" if pd.isna(r.get("sales")) else f"{r.get('sales'):,.2f}"
    customers="" if pd.isna(r.get("customers")) else str(int(r.get("customers")))
    zipcodes="" if pd.isna(r.get("zipcodes")) else str(int(r.get("zipcodes")))
    lines.append(f"| {r['stage']} | {int(r['rows']):,} | {sales} | {customers} | {zipcodes} |")

(qa/"local_monday_qa.md").write_text("\n".join(lines),encoding="utf-8")
print(json.dumps(result,indent=2,default=str))
