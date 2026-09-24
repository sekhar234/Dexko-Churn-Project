from __future__ import annotations
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd

from .io import read_table, write_partition, write_df, read_df, resolve_table, update_manifest
from .common import sanitize_columns

SOURCE_TABLES = {
    "dimcustomer": "jdbc_customer_weekly",
    "factsalesinvoice": "jdbc_invoice_weekly",
    "dimproduct": "jdbc_product_weekly",
    "dimwarehouselocation": "jdbc_warehouse_weekly",
    "dimcustomershipto": "jdbc_shiptokey_weekly",
}


def monday_of(value) -> str:
    d=pd.Timestamp(value).date()
    return (d-timedelta(days=d.weekday())).isoformat()


def _source(root: Path, name: str) -> pd.DataFrame:
    return read_table(root/"data"/"source"/"edw"/name)


def _dedupe(df: pd.DataFrame, key: str) -> pd.DataFrame:
    return df.drop_duplicates(subset=[key], keep="first").copy()


def _build_master(cust, inv, prod, wh, shipto, snapshot_week: str) -> pd.DataFrame:
    cust_d=_dedupe(cust,"Customer Key")
    prod_d=_dedupe(prod,"Product Key")
    wh_key="Warehouse Location Key" if "Warehouse Location Key" in wh.columns else "Warehouse_Location_Key"
    wh_d=_dedupe(wh,wh_key)
    ship_d=_dedupe(shipto,"CustomerShipTokey")

    csel=cust_d[["Customer Key","Harmonized Sold To Customer Name","Customer Name","Intercompany Flag"]].copy()
    csel=csel.rename(columns={"Customer Key":"InvoiceAccountCustomerKey", "Customer Name":"DimCustomerName"})
    psel=prod_d[["Product Key","Product Name","MGR L1","Primary Vendor Group"]].copy()
    wcols=[wh_key]+[c for c in ["InventSiteId","Invent Location Id","Invent_Location_Id","Business Unit","Business_Unit","InventSideIdName"] if c in wh_d.columns]
    wsel=wh_d[wcols].copy().rename(columns={wh_key:"Warehouse Location Key"})
    ssel=ship_d[[c for c in ["CustomerShipTokey","CustomerName","City","State","ZipCode","Country","Territory","SalesRep","Region","RegionalManager","VP","ServicingBranch","ReportingSegment"] if c in ship_d.columns]].copy()
    ssel=ssel.rename(columns={"CustomerName":"ShipToCustomerName","City":"ShipTo_City","State":"ShipTo_State","ZipCode":"ShipTo_ZipCode","Country":"ShipTo_Country"})

    m=inv.merge(csel,on="InvoiceAccountCustomerKey",how="left")
    m=m.merge(psel,on="Product Key",how="left",suffixes=("","_prod"))
    m=m.merge(wsel,on="Warehouse Location Key",how="left",suffixes=("","_wh"))
    m=m.merge(ssel,on="CustomerShipTokey",how="left",suffixes=("","_shipto"))
    m["snapshot_week"]=snapshot_week
    return m


def run_snapshot(root: str|Path, as_of_date: str|None=None, preferred_format: str="parquet") -> dict:
    root=Path(root)
    if as_of_date is None:
        import yaml
        cfg=yaml.safe_load((root/"config"/"synthetic.yml").read_text())
        as_of_date=cfg["dates"]["end"]
    snapshot_week=monday_of(as_of_date)
    now=datetime.now(timezone.utc)
    batch_id=f"LOCAL_BATCH_{snapshot_week.replace('-','')}_{now.strftime('%H%M%S')}"
    cache=root/"data"/"edw_cache"
    cache.mkdir(parents=True,exist_ok=True)

    raw={name:_source(root,name) for name in SOURCE_TABLES}
    counts={}
    paths={}
    for src,target in SOURCE_TABLES.items():
        df=raw[src].copy()
        if src in {"dimproduct","dimwarehouselocation","dimcustomershipto"}:
            df=sanitize_columns(df)
        df["snapshot_week"]=snapshot_week
        paths[target]=str(write_partition(df,cache/target,snapshot_week,preferred_format))
        counts[target]=len(df)

    master=_build_master(raw["dimcustomer"],raw["factsalesinvoice"],raw["dimproduct"],raw["dimwarehouselocation"],raw["dimcustomershipto"],snapshot_week)
    paths["master_weekly"]=str(write_partition(master,cache/"master_weekly",snapshot_week,preferred_format))
    counts["master_weekly"]=len(master)
    inv_n=max(len(raw["factsalesinvoice"]),1)
    join_loss=abs(len(master)-len(raw["factsalesinvoice"]))/inv_n
    if join_loss>0.05:
        raise AssertionError(f"master_weekly join loss {join_loss:.1%} exceeds 5%")

    reg_base=cache/"snapshot_registry"
    try:
        reg=read_df(resolve_table(reg_base))
    except FileNotFoundError:
        reg=pd.DataFrame(columns=["dataset_name","snapshot_week","batch_id","status","row_count","source_watermark","started_at","completed_at","error_message"])
    if len(reg):
        reg=reg[~((reg["dataset_name"].astype(str)=="master_weekly") & (reg["snapshot_week"].astype(str).str[:10]==snapshot_week))].copy()
    row={"dataset_name":"master_weekly","snapshot_week":snapshot_week,"batch_id":batch_id,"status":"SUCCESS","row_count":len(master),"source_watermark":None,"started_at":now.isoformat(),"completed_at":datetime.now(timezone.utc).isoformat(),"error_message":None}
    reg=pd.concat([reg,pd.DataFrame([row])],ignore_index=True)
    paths["snapshot_registry"]=str(write_df(reg,reg_base,preferred_format))

    result={"snapshot_week":snapshot_week,"batch_id":batch_id,"counts":counts,"join_loss":join_loss,"paths":paths}
    update_manifest(root,"snapshot",result)
    return result
