from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import json
import warnings

import numpy as np
import pandas as pd
import yaml

from .contracts import SOURCE_CONTRACTS

CATEGORIES = [
    "AXLE GROUP", "BRAKES", "WHEELS & TIRES", "HUB & HUB-DRUMS",
    "SPRING & SUSPENSION", "COUPLERS", "LIGHTING", "CARGO CONTROL",
    "FENDER GROUP", "BRAKE CONTROLS", "VENT, DOORS, AND WINDOWS",
    "MISC COMPONENTS",
]
JUNK_CATEGORIES = ["UNASSIGNED", "N/A", "I/N", "I/C"]
US_LOCATIONS = [
    ("Dallas", "TX", "75001"), ("Houston", "TX", "77002"),
    ("Milwaukee", "WI", "53202"), ("Madison", "WI", "53703"),
    ("Omaha", "NE", "68102"), ("Chicago", "IL", "60601"),
    ("Columbus", "OH", "43215"), ("Phoenix", "AZ", "85001"),
    ("Atlanta", "GA", "30303"), ("Nashville", "TN", "37201"),
    ("Boston", "MA", "02108"), ("Portland", "ME", "04101"),
]
CA_LOCATIONS = [
    ("Toronto", "ON", "M5V 2T6"), ("Calgary", "AB", "T2P 1J9"),
    ("Vancouver", "BC", "V6B 1A1"),
]
FIRST = ["ALPHA", "NORTHSTAR", "MIDWEST", "PREMIER", "CENTRAL", "ROADMASTER",
         "SUMMIT", "PIONEER", "RIVERBEND", "HORIZON", "METRO", "HERITAGE"]
LAST = ["TRAILER SYSTEMS", "EQUIPMENT", "WELDING", "TRAILER PARTS", "UTILITY",
        "REPAIR", "FABRICATION", "TRANSPORT", "INDUSTRIES", "TRAILER SALES"]


@dataclass
class SyntheticBundle:
    dimcustomer: pd.DataFrame
    factsalesinvoice: pd.DataFrame
    dimproduct: pd.DataFrame
    dimwarehouselocation: pd.DataFrame
    dimcustomershipto: pd.DataFrame
    ground_truth: pd.DataFrame
    account_owners: pd.DataFrame


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _weighted_choice(rng: np.random.Generator, mapping: dict, n: int) -> np.ndarray:
    keys = list(mapping)
    probs = np.array([mapping[k] for k in keys], dtype=float)
    probs = probs / probs.sum()
    return rng.choice(keys, size=n, p=probs)


def _scenario_dates(rng, scenario: str, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    span = (end - start).days
    if scenario in {"category_churn", "account_churn", "migration"}:
        churn_start = start + pd.Timedelta(days=int(rng.integers(max(500, span//3), max(501, int(span*0.72)))))
    else:
        churn_start = pd.NaT
    shock = start + pd.Timedelta(days=int(rng.integers(max(300, span//4), max(301, int(span*0.78))))) if scenario in {"price_shock", "lead_time_problem", "declining"} else pd.NaT
    false_gap_start = start + pd.Timedelta(days=int(rng.integers(max(400, span//3), max(401, int(span*0.75))))) if scenario == "false_alarm" else pd.NaT
    return {"churn_date": churn_start, "shock_date": shock, "false_gap_start": false_gap_start}


def _make_event_dates(rng, start, end, cadence, scenario, dates, seasonal_months=None):
    if scenario == "new_customer":
        start = max(start, end - pd.Timedelta(days=int(rng.integers(100, 280))))
    cur = start + pd.Timedelta(days=int(rng.integers(0, max(cadence, 2))))
    out = []
    while cur <= end:
        if scenario in {"category_churn", "account_churn"} and pd.notna(dates["churn_date"]) and cur > dates["churn_date"]:
            break
        if scenario == "false_alarm" and pd.notna(dates["false_gap_start"]):
            gap_end = dates["false_gap_start"] + pd.Timedelta(days=int(rng.integers(90, 145)))
            if dates["false_gap_start"] <= cur <= gap_end:
                cur = gap_end + pd.Timedelta(days=int(rng.integers(1, 8)))
                continue
        if scenario == "seasonal" and seasonal_months and cur.month not in seasonal_months:
            cur += pd.Timedelta(days=7)
            continue
        out.append(cur)
        c = cadence
        if scenario == "sporadic":
            c = int(rng.integers(max(35, cadence), max(70, cadence*3)))
        elif scenario == "declining" and pd.notna(dates["shock_date"]) and cur >= dates["shock_date"]:
            elapsed = min((cur - dates["shock_date"]).days / 365.0, 1.5)
            c = int(cadence * (1.4 + 0.8 * elapsed))
        jitter = int(rng.normal(0, max(2.0, c * 0.15)))
        cur += pd.Timedelta(days=max(5, c + jitter))
    return out


def generate(config: dict) -> SyntheticBundle:
    rng = np.random.default_rng(int(config.get("seed", 42)))
    scale_name = config.get("scale", "small")
    scale = config["scales"][scale_name]
    n_customers = int(scale["customers"])
    n_products = int(scale["products"])
    n_wh = int(scale["warehouses"])
    start = pd.Timestamp(config["dates"]["start"])
    end = pd.Timestamp(config["dates"]["end"])
    dirty = config["dirty_data"]
    pb = config["purchase_behavior"]

    bu1 = ["DDG", "AXLE", "HEAVY DUTY", "DMP"]
    wh_rows=[]
    for i in range(n_wh):
        wkey=f"WH{i+1:04d}"; site=f"BR{(i+1)*10:03d}"
        b1=bu1[i % len(bu1)]
        wh_rows.append({
            "Warehouse Location Key":wkey,"BU Level 1":b1,"BU Level 2":f"{b1}-L2-{i%3+1}",
            "BU Level 3":f"{b1}-L3-{i%5+1}","InventSiteId":site,"Invent Location Id":f"LOC{i+1:03d}",
            "Business Unit":f"{b1} BU","InventSideIdName":f"Synthetic Site {i+1}",
        })
    dimwh=pd.DataFrame(wh_rows)

    product_rows=[]; product_by_cat={c:[] for c in CATEGORIES+JUNK_CATEGORIES}
    n_junk=max(2, int(n_products*dirty["junk_product_rate"]))
    for i in range(n_products):
        cat = JUNK_CATEGORIES[i % len(JUNK_CATEGORIES)] if i >= n_products-n_junk else CATEGORIES[i % len(CATEGORIES)]
        pkey=f"P{i+1:05d}"; item=f"SKU{i+1:06d}"
        row={"Product Key":pkey,"MGR L1":cat,"MGR L2":f"{cat} SUBGROUP",
             "Product Name":f"Synthetic {cat.title()} Product {i+1}","Item Id":item,
             "Primary Vendor Group":f"VENDOR-{i%7+1}","Item Group Id":f"IG{i%12+1:02d}",
             "Item Group Name":f"ITEM GROUP {i%12+1}","Product Description":f"Synthetic test product {i+1}",
             "Product Type":"Finished Good"}
        product_rows.append(row); product_by_cat[cat].append(pkey)
    n_dup=max(1,int(n_products*dirty["product_scd_duplicate_rate"]))
    for idx in rng.choice(np.arange(n_products-n_junk),size=min(n_dup,n_products-n_junk),replace=False):
        r=product_rows[int(idx)].copy(); r["Product Description"] += " (historical version)"; product_rows.append(r)
    dimprod=pd.DataFrame(product_rows)

    groups=_weighted_choice(rng, config["customer_groups"], n_customers)
    intercompany = rng.random(n_customers) < dirty["intercompany_customer_rate"]
    customer_rows=[]; shipto_rows=[]; canonical=[]; customer_keys_by_customer={}; shiptos_by_customer={}
    owner_names=["Sarah Miller","Daniel Lee","Priya Sharma","Jordan Davis","Alex Morgan","Taylor Wilson"]
    owner_rows=[]
    used_names=set()
    for i in range(n_customers):
        base_name=f"{FIRST[i%len(FIRST)]} {LAST[(i*5)%len(LAST)]} {i+1:04d}"
        while base_name in used_names: base_name += "X"
        used_names.add(base_name)
        canonical.append(base_name)
        n_keys = 2 if rng.random() < 0.16 else 1
        keys=[]
        for k in range(n_keys):
            ck=f"CUSTKEY{i+1:06d}_{k+1}"
            cid=f"{i+1:06d}" if k == n_keys-1 else (f"TCM-{i+1:06d}" if rng.random()<0.45 else f"{i+1:06d}{k}")
            keys.append(ck)
            harm=base_name
            if rng.random() < dirty["junk_customer_name_rate"]:
                harm=rng.choice(["UNASSIGNED","0",""])
            site=dimwh.iloc[(i+k)%len(dimwh)]["InventSiteId"]
            row={
                "Harmonized Sold To Customer Name":harm,"Customer Key":ck,"Customer Name":base_name,
                "Customer Group":f"G{(i%9)+1}","Customer Group Name":groups[i],
                "Credit Max":float(np.round(rng.uniform(10000,800000),2)),"Customer Id":cid,
                "Sales Rep":f"REP-{i%20+1:02d}","Data Area Id":"dex" if not str(cid).startswith("TCM-") else "tcm",
                "Intercompany Flag":"Y" if intercompany[i] else "N","Region":f"REGION-{i%5+1}",
                "Reporting Segment":f"SEGMENT-{i%4+1}","InventSiteId":site,
            }
            customer_rows.append(row)
            if rng.random() < dirty["customer_scd_duplicate_rate"]:
                old=row.copy(); old["Credit Max"] = max(5000.0, row["Credit Max"]*0.85); old["Sales Rep"]=f"REP-{(i+7)%20+1:02d}"; customer_rows.append(old)
        customer_keys_by_customer[i]=keys
        n_ship=int(rng.integers(pb["min_shiptos_per_customer"], pb["max_shiptos_per_customer"]+1))
        ships=[]
        for s in range(n_ship):
            if rng.random()<0.08:
                city,state,zipc=rng.choice(CA_LOCATIONS); country="CA"
            else:
                city,state,zipc=US_LOCATIONS[(i*3+s)%len(US_LOCATIONS)]; country="US"
            sk=f"SHIP{i+1:06d}_{s+1}"
            ships.append({"key":sk,"zip":zipc,"city":city,"state":state,"country":country,"seq":f"{s+1:03d}"})
            shipto_rows.append({"CustomerShipTokey":sk,"CustomerName":base_name,"City":city,"State":state,"ZipCode":zipc,
                                "Country":country,"Territory":f"TERR-{i%8+1}","SalesRep":f"REP-{i%20+1:02d}",
                                "Region":f"REGION-{i%5+1}","RegionalManager":f"RM-{i%6+1}","VP":f"VP-{i%3+1}",
                                "ServicingBranch":dimwh.iloc[i%len(dimwh)]["InventSiteId"],"ReportingSegment":f"SEGMENT-{i%4+1}"})
        shiptos_by_customer[i]=ships
        owner_rows.append({"Customer Id":f"{i+1:06d}","Account Owner":owner_names[i%len(owner_names)],"Synthetic Customer Name":base_name})
    dimcust=pd.DataFrame(customer_rows)
    dimship=pd.DataFrame(shipto_rows)

    customer_behavior=_weighted_choice(rng, config["behaviors"], n_customers)
    invoice_rows=[]; gt_rows=[]; inv_num=1
    base_prices={cat:float(rng.uniform(12,350)) for cat in CATEGORIES}

    for ci in range(n_customers):
        cust_name=canonical[ci]
        account_scenario=str(customer_behavior[ci])
        ships=shiptos_by_customer[ci]
        keys=customer_keys_by_customer[ci]
        account_dates=_scenario_dates(rng, account_scenario, start, end)
        for ship_idx, sh in enumerate(ships):
            n_cat=int(rng.integers(pb["min_categories_per_shipto"], pb["max_categories_per_shipto"]+1))
            cats=list(rng.choice(CATEGORIES,size=min(n_cat,len(CATEGORIES)),replace=False))
            for cat_idx,cat in enumerate(cats):
                if account_scenario == "account_churn": scenario="account_churn"
                elif account_scenario == "migration" and cat_idx==0: scenario="migration"
                elif account_scenario in {"stable","growing","declining","seasonal","sporadic","new_customer","price_shock","lead_time_problem","false_alarm"}:
                    scenario=account_scenario if cat_idx==0 or rng.random()<0.35 else str(_weighted_choice(rng, config["behaviors"],1)[0])
                    if scenario=="account_churn": scenario="category_churn"
                else: scenario="category_churn" if rng.random()<0.15 else "stable"
                dates=account_dates if scenario=="account_churn" else _scenario_dates(rng,scenario,start,end)
                cadence=int(rng.integers(pb["cadence_days_min"],pb["cadence_days_max"]+1))
                base_value=float(np.exp(rng.uniform(np.log(pb["base_order_value_min"]),np.log(pb["base_order_value_max"]))))
                seasonal_months=None
                if scenario=="seasonal":
                    start_month=int(rng.integers(1,9)); seasonal_months={((start_month+j-1)%12)+1 for j in range(int(rng.integers(3,6)))}
                event_dates=_make_event_dates(rng,start,end,cadence,scenario,dates,seasonal_months)
                migration_date=dates["churn_date"] if scenario=="migration" else pd.NaT
                for event_i,dt in enumerate(event_dates):
                    progress=(dt-start).days/max((end-start).days,1)
                    multiplier=1.0
                    if scenario=="growing": multiplier=0.75+0.8*progress
                    elif scenario=="declining": multiplier=max(0.18,1.25-1.0*progress)
                    elif scenario=="category_churn": multiplier=max(0.35,1.15-0.45*progress)
                    elif scenario=="false_alarm" and pd.notna(dates["false_gap_start"]) and dt > dates["false_gap_start"]+pd.Timedelta(days=80): multiplier=1.7
                    spend=max(25.0,base_value*multiplier*rng.lognormal(mean=0,sigma=pb["noise_sigma"]))
                    unit=base_prices[cat]*(1+0.025*((dt.year-start.year)))
                    if scenario=="price_shock" and pd.notna(dates["shock_date"]) and dt>=dates["shock_date"]:
                        unit*=float(rng.uniform(1.6,2.8)); spend*=0.65
                    qty=max(1,int(round(spend/max(unit,1))))
                    spend=float(round(qty*unit*rng.uniform(0.96,1.04),2))
                    lead=max(0,int(rng.normal(5,2)))
                    if scenario=="lead_time_problem" and pd.notna(dates["shock_date"]) and dt>=dates["shock_date"]:
                        lead+=int(rng.integers(10,28))
                    salesorder_dt=dt-pd.Timedelta(days=lead)
                    pkey=str(rng.choice(product_by_cat[cat]))
                    wh=dimwh.iloc[(ci+ship_idx+event_i)%len(dimwh)]
                    if scenario=="migration" and len(keys)>1 and pd.notna(migration_date): ck=keys[0] if dt<=migration_date else keys[-1]
                    else: ck=keys[-1]
                    cid_row=dimcust[dimcust["Customer Key"]==ck].iloc[0]
                    dataarea="tcm" if str(cid_row["Customer Id"]).startswith("TCM-") else "dex"
                    invoice_rows.append({
                        "InvoiceDate":dt,"NetAmountExtended":spend,"Quantity":qty,"DataAreaId":dataarea,
                        "SalesOrderCreatedDate":salesorder_dt,"Warehouse Location Key":wh["Warehouse Location Key"],
                        "Product Key":pkey,"InvoiceAccountCustomerKey":ck,"CustomerShipTokey":sh["key"],
                        "ShiptoSeq":sh["seq"],"InvoiceId":f"INV-{inv_num:09d}","City":sh["city"],"State":sh["state"],
                        "ZipCode":sh["zip"],"Country":sh["country"],"Street":f"{100+ci} Synthetic Way",
                        "SalesId":f"SO-{inv_num:09d}","SalesOrderDate":salesorder_dt,"InterCompanyPosted":"N",
                    }); inv_num+=1
                gt_rows.append({
                    "synthetic_customer_name":cust_name,"customer_index":ci+1,"customershiptokey":sh["key"],
                    "zipcode":sh["zip"],"mgrl1":cat,"scenario":scenario,
                    "planned_churn_date":None if pd.isna(dates["churn_date"]) else dates["churn_date"],
                    "shock_date":None if pd.isna(dates["shock_date"]) else dates["shock_date"],
                    "expected_risk_direction":("higher" if scenario in {"declining","category_churn","account_churn","price_shock","lead_time_problem"} else "lower_or_variable"),
                })

    inv=pd.DataFrame(invoice_rows)
    n=len(inv)
    if n:
        def choose(rate):
            k=max(1,int(n*rate)) if rate>0 else 0
            return rng.choice(inv.index,size=min(k,n),replace=False) if k else []
        for idx in choose(dirty["non_dex_invoice_rate"]): inv.loc[idx,"DataAreaId"]="other"
        for idx in choose(dirty["nonpositive_revenue_rate"]): inv.loc[idx,"NetAmountExtended"]=float(rng.choice([0.0,-25.0,-100.0]))
        for idx in choose(dirty["zero_quantity_rate"]): inv.loc[idx,"Quantity"]=0
        for idx in choose(dirty["null_invoice_date_rate"]): inv.loc[idx,"InvoiceDate"]=pd.NaT
        badz=["00000","123","ABCD","12WH","999999",""]
        for idx in choose(dirty["invalid_zip_rate"]): inv.loc[idx,"ZipCode"]=str(rng.choice(badz))

    gt=pd.DataFrame(gt_rows)
    owners=pd.DataFrame(owner_rows)
    dimcust=dimcust[SOURCE_CONTRACTS["dimcustomer"]]
    inv=inv[SOURCE_CONTRACTS["factsalesinvoice"]]
    dimprod=dimprod[SOURCE_CONTRACTS["dimproduct"]]
    dimwh=dimwh[SOURCE_CONTRACTS["dimwarehouselocation"]]
    dimship=dimship[SOURCE_CONTRACTS["dimcustomershipto"]]
    return SyntheticBundle(dimcust,inv,dimprod,dimwh,dimship,gt,owners)


def _write_df(df: pd.DataFrame, path_without_ext: Path, fmt: str) -> Path:
    path_without_ext.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        try:
            path=path_without_ext.with_suffix(".parquet")
            df.to_parquet(path,index=False)
            return path
        except Exception as e:
            warnings.warn(f"Parquet unavailable ({e}); falling back to CSV. Install pyarrow to enable Parquet.")
    path=path_without_ext.with_suffix(".csv")
    df.to_csv(path,index=False)
    return path


def write_bundle(bundle: SyntheticBundle, root: str | Path, fmt: str="parquet") -> Dict[str,str]:
    root=Path(root)
    files={}
    source=root/"data"/"source"/"edw"
    for name,df in [
        ("dimcustomer",bundle.dimcustomer),("factsalesinvoice",bundle.factsalesinvoice),
        ("dimproduct",bundle.dimproduct),("dimwarehouselocation",bundle.dimwarehouselocation),
        ("dimcustomershipto",bundle.dimcustomershipto),
    ]:
        files[name]=str(_write_df(df,source/name,fmt))
    files["ground_truth"]=str(_write_df(bundle.ground_truth,root/"data"/"ground_truth"/"synthetic_ground_truth",fmt))
    files["account_owner_mapping"]=str(_write_df(bundle.account_owners,root/"mappings"/"synthetic_account_owner",fmt))
    with open(root/"data"/"manifest.json","w",encoding="utf-8") as f:
        json.dump(files,f,indent=2,default=str)
    return files
