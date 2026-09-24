from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

from .generator import SyntheticBundle


def run_qa(bundle: SyntheticBundle) -> dict:
    c=bundle.dimcustomer; i=bundle.factsalesinvoice; p=bundle.dimproduct; w=bundle.dimwarehouselocation; s=bundle.dimcustomershipto; gt=bundle.ground_truth
    cust_keys=set(c["Customer Key"].astype(str))
    prod_keys=set(p["Product Key"].astype(str))
    wh_keys=set(w["Warehouse Location Key"].astype(str))
    ship_keys=set(s["CustomerShipTokey"].astype(str))
    inv_nonnull=i[i["InvoiceDate"].notna()]
    report={
        "row_counts": {"dimcustomer":len(c),"factsalesinvoice":len(i),"dimproduct":len(p),"dimwarehouselocation":len(w),"dimcustomershipto":len(s),"ground_truth":len(gt)},
        "referential_coverage": {
            "customer_key_pct": round(i["InvoiceAccountCustomerKey"].astype(str).isin(cust_keys).mean()*100,3) if len(i) else 0,
            "product_key_pct": round(i["Product Key"].astype(str).isin(prod_keys).mean()*100,3) if len(i) else 0,
            "warehouse_key_pct": round(i["Warehouse Location Key"].astype(str).isin(wh_keys).mean()*100,3) if len(i) else 0,
            "shipto_key_pct": round(i["CustomerShipTokey"].astype(str).isin(ship_keys).mean()*100,3) if len(i) else 0,
        },
        "date_range": {
            "min_invoice_date": None if inv_nonnull.empty else str(pd.to_datetime(inv_nonnull["InvoiceDate"]).min().date()),
            "max_invoice_date": None if inv_nonnull.empty else str(pd.to_datetime(inv_nonnull["InvoiceDate"]).max().date()),
        },
        "customer_groups": c["Customer Group Name"].value_counts(dropna=False).to_dict(),
        "behaviors": gt["scenario"].value_counts(dropna=False).to_dict(),
        "dirty_cases": {
            "non_dex_invoices": int((i["DataAreaId"].astype(str).str.lower().str.strip()!="dex").sum()),
            "nonpositive_revenue": int((pd.to_numeric(i["NetAmountExtended"],errors="coerce")<=0).sum()),
            "zero_quantity": int((pd.to_numeric(i["Quantity"],errors="coerce")<=0).sum()),
            "null_invoice_date": int(i["InvoiceDate"].isna().sum()),
            "intercompany_customers": int((c["Intercompany Flag"]=="Y").sum()),
            "junk_harmonized_names": int(c["Harmonized Sold To Customer Name"].fillna("").astype(str).str.strip().str.upper().isin(["","0","UNASSIGNED"]).sum()),
            "junk_product_rows": int(p["MGR L1"].fillna("").astype(str).str.upper().isin(["UNASSIGNED","N/A","I/N","I/C"]).sum()),
        },
        "grain": {
            "distinct_customer_zip_category_truth": int(gt[["synthetic_customer_name","zipcode","mgrl1"]].drop_duplicates().shape[0]),
            "multi_zip_customers": int((s.groupby("CustomerName")["ZipCode"].nunique()>1).sum()),
        },
    }
    report["checks"]={
        "all_invoice_fk_coverage_100pct": all(v==100.0 for v in report["referential_coverage"].values()),
        "has_oem": "OEM" in report["customer_groups"],
        "has_stocking_dealer": "Stocking Dealer" in report["customer_groups"],
        "has_category_churn": report["behaviors"].get("category_churn",0)>0,
        "has_account_churn": report["behaviors"].get("account_churn",0)>0,
        "has_stable": report["behaviors"].get("stable",0)>0,
        "has_dirty_rows": sum(report["dirty_cases"].values())>0,
        "has_multi_zip_customers": report["grain"]["multi_zip_customers"]>0,
    }
    return report


def write_qa(report: dict, root: str | Path):
    root=Path(root); out=root/"data"/"qa"; out.mkdir(parents=True,exist_ok=True)
    (out/"synthetic_qa.json").write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
    lines=["# Synthetic Dataset QA Report","", "## Checks"]
    for k,v in report["checks"].items(): lines.append(f"- [{'x' if v else ' '}] `{k}`")
    lines += ["", "## Row counts"] + [f"- **{k}**: {v:,}" for k,v in report["row_counts"].items()]
    lines += ["", "## Referential coverage"] + [f"- **{k}**: {v:.3f}%" for k,v in report["referential_coverage"].items()]
    lines += ["", "## Date range", f"- Min invoice date: {report['date_range']['min_invoice_date']}", f"- Max invoice date: {report['date_range']['max_invoice_date']}"]
    lines += ["", "## Dirty cases"] + [f"- **{k}**: {v:,}" for k,v in report["dirty_cases"].items()]
    lines += ["", "## Behaviour distribution"] + [f"- **{k}**: {v:,}" for k,v in report["behaviors"].items()]
    (out/"synthetic_qa.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
