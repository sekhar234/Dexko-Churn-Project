from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import re
import pandas as pd

from .common import sanitize_columns, rename_columns, is_bad_name
from .io import read_partition, read_df, resolve_table, write_df, update_manifest

BAD_MGRL1={"UNASSIGNED","N/A","I/N","I/C"}
OEM_STOCKING={"OEM","STOCKING DEALER"}
TRAIN_START_DATE=pd.Timestamp("2023-01-01")

COUNTRY_MAP={
    "USA":"US","UNITED STATES":"US","US":"US","CANADA":"CA","CA":"CA","CAN":"CA",
    "MEX":"MX","MEXICO":"MX","CHN":"CN","CHINA":"CN","TWN":"TW","TAIWAN":"TW",
    "IND":"IN","INDIA":"IN","THA":"TH","THAILAND":"TH","VNM":"VN","VIETNAM":"VN",
    "ROU":"RO","ROMANIA":"RO","AUS":"AU","AUSTRALIA":"AU","BRA":"BR","BRAZIL":"BR",
    "DEU":"DE","GERMANY":"DE","GER":"DE","GBR":"GB","UNITED KINGDOM":"GB","UK":"GB",
    "FRA":"FR","FRANCE":"FR","UNA":None,"UNASSIGNED":None,
}
FORMAT_RULES={
    "US":r"^\d{5}$","CA":r"^[A-Z]\d[A-Z]\s?\d[A-Z]\d$","MX":r"^\d{5}$","CN":r"^\d{6}$",
    "TW":r"^\d{3,6}$","IN":r"^\d{6}$","TH":r"^\d{5}$","VN":r"^\d{5,6}$","RO":r"^\d{6}$",
    "AU":r"^\d{4}$","BR":r"^\d{5}-?\d{3}$","DE":r"^\d{5}$","GB":r"^[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}$","FR":r"^\d{5}$",
}


def _norm_zip(v):
    if pd.isna(v): return None
    s=str(v).strip().upper()
    if re.fullmatch(r"\d+\.0",s): s=s[:-2]
    if re.fullmatch(r"\d{5}-\d{4}",s): s=s[:5]
    if re.fullmatch(r"[A-Z]\d[A-Z]\d[A-Z]\d",s): s=s[:3]+" "+s[3:]
    return s


def _country_iso(v):
    if pd.isna(v): return None
    return COUNTRY_MAP.get(str(v).strip().upper())


def _zip_eval(raw,country):
    z=_norm_zip(raw); iso=_country_iso(country)
    if z is None or z.strip()=="": return z,"ANOMALY","Missing ZIP/postal code","RULE_NULL_BLANK_ZIP"
    structural=(re.fullmatch(r"0+",z) or z=="." or z.startswith("+") or z=="-" or "/" in z or "(" in z or re.fullmatch(r"\d{1,2}",z) or re.fullmatch(r"[A-Z]+",z) or re.fullmatch(r"\d{2}WH",z,re.I) or re.search(r"[*^]",z) or re.fullmatch(r"\d+\+",z) or re.fullmatch(r"\d{3}-\d{3}-\d{2,}",z))
    if structural:
        if re.fullmatch(r"0+",z): reason="Invalid character pattern (all zeros)"
        elif re.fullmatch(r"\d{1,2}",z): reason="Invalid character pattern (too short)"
        elif re.fullmatch(r"[A-Z]+",z): reason="Invalid character pattern (pure alpha)"
        elif re.fullmatch(r"\d{2}WH",z,re.I): reason="Invalid character pattern (warehouse code)"
        elif re.fullmatch(r"\d{3}-\d{3}-\d{2,}",z): reason="Invalid character pattern (phone-like)"
        else: reason="Invalid character pattern"
        return z,"ANOMALY",reason,"RULE_STRUCTURAL_PATTERN"
    if iso=="US" and re.fullmatch(r"\d{4}",z):
        return z,"ANOMALY","Suspicious four-digit US ZIP","RULE_4DIGIT_US_SUSPICIOUS"
    if iso in FORMAT_RULES and not re.fullmatch(FORMAT_RULES[iso],z):
        return z,"ANOMALY","Invalid format for supplied country",f"FORMAT_{iso}"
    return z,"VALID",None,"PASS"


def _latest_success(root: Path) -> str:
    reg=read_df(resolve_table(root/"data"/"edw_cache"/"snapshot_registry"))
    reg=reg[(reg["dataset_name"].astype(str)=="master_weekly") & (reg["status"].astype(str)=="SUCCESS")].copy()
    if reg.empty: raise RuntimeError("No SUCCESS master_weekly snapshot in local snapshot_registry")
    return pd.to_datetime(reg["snapshot_week"]).max().date().isoformat()


def _read_weekly(root: Path, table: str, week: str) -> pd.DataFrame:
    df=read_partition(root/"data"/"edw_cache"/table,week)
    if "snapshot_week" in df.columns: df=df.drop(columns=["snapshot_week"])
    return rename_columns(sanitize_columns(df))


def _funnel(rows, stage, df, note=""):
    rows.append({"stage":stage,"rows":int(len(df)),"sales":float(pd.to_numeric(df.get("netamountextended",pd.Series(dtype=float)),errors="coerce").sum()) if "netamountextended" in df else None,
                 "customers":int(df["harmonizedsoldtocustomername"].nunique(dropna=True)) if "harmonizedsoldtocustomername" in df else None,
                 "zipcodes":int(df["zipcode"].nunique(dropna=True)) if "zipcode" in df else None,"note":note})


def run_load_data(root: str|Path, preferred_format: str="parquet") -> dict:
    root=Path(root); week=_latest_success(root); funnel=[]
    inv=_read_weekly(root,"jdbc_invoice_weekly",week)
    cust=_read_weekly(root,"jdbc_customer_weekly",week)
    prod=_read_weekly(root,"jdbc_product_weekly",week)
    wh=_read_weekly(root,"jdbc_warehouse_weekly",week)
    _ship=_read_weekly(root,"jdbc_shiptokey_weekly",week)

    for c in ["invoicedate","salesordercreateddate","salesorderdate"]:
        if c in inv: inv[c]=pd.to_datetime(inv[c],errors="coerce")
    for c in ["netamountextended","quantity"]:
        inv[c]=pd.to_numeric(inv[c],errors="coerce")
    if "creditmax" in cust: cust["creditmax"]=pd.to_numeric(cust["creditmax"],errors="coerce")
    for df,cols in [(inv,["invoiceaccountcustomerkey","productkey","warehouselocationkey","customershiptokey","zipcode"]),(cust,["customerkey","customerid"]),(prod,["productkey"]),(wh,["warehouselocationkey"])]:
        for c in cols:
            if c in df: df[c]=df[c].astype("string")

    bridge=cust[["customerkey","harmonizedsoldtocustomername","customername","intercompanyflag"]].copy()
    bridge["harmonizedsoldtocustomername"]=bridge["harmonizedsoldtocustomername"].astype("string").str.strip()
    bridge=bridge.drop(columns=["customername"]).drop_duplicates("customerkey",keep="first")

    prod_sel=prod[["productkey","productname","itemid","mgrl1","primaryvendorgroup"]].drop_duplicates("productkey",keep="first").copy()
    cat=prod_sel["mgrl1"].astype("string").str.strip()
    prod_sel=prod_sel[prod_sel["mgrl1"].notna() & cat.ne("") & ~cat.str.upper().isin(BAD_MGRL1)].copy()
    wh_sel=wh[["warehouselocationkey","bulevel1","bulevel2","bulevel3"]].drop_duplicates("warehouselocationkey",keep="first").copy()

    _funnel(funnel,"Invoice",inv)
    a=inv.merge(bridge,left_on="invoiceaccountcustomerkey",right_on="customerkey",how="inner")
    _funnel(funnel,"After adding Customer table",a,"Drops customers not in Customer table")
    b=a.merge(prod_sel,on="productkey",how="inner")
    _funnel(funnel,"After adding Product table",b)
    c=b.merge(wh_sel,on="warehouselocationkey",how="inner")
    _funnel(funnel,"After adding Warehouse table",c)
    d=c[~is_bad_name(c["harmonizedsoldtocustomername"])].copy()
    _funnel(funnel,"Dropping Blank/Unassigned names",d)

    country_col="Country" if "Country" in d.columns else ("country" if "country" in d.columns else None)
    if country_col is None: raise RuntimeError("Country column missing before ZIP validation")
    vals=d.apply(lambda r:_zip_eval(r.get("zipcode"),r.get(country_col)),axis=1,result_type="expand")
    vals.columns=["zipcode_normalized","zipcode_validation_status","zipcode_anomaly_reason","zipcode_validation_rule"]
    zv=pd.concat([d.reset_index(drop=True),vals],axis=1)
    zv["zipcode_raw"]=zv["zipcode"]
    anomalies=zv[zv["zipcode_validation_status"]=="ANOMALY"].copy()
    e=zv[zv["zipcode_validation_status"]=="VALID"].copy()
    e["zipcode"]=e["zipcode_normalized"].astype("string")
    _funnel(funnel,"After ZIP Validation (VALID only)",e,"Anomalies routed to audit")
    after_f=e

    dex=after_f[after_f["dataareaid"].astype("string").str.strip().str.lower().eq("dex")].copy()
    rev=dex[(dex["netamountextended"]>0)&(dex["quantity"]>0)].groupby(["harmonizedsoldtocustomername","zipcode","customerkey"],dropna=False,as_index=False)["netamountextended"].sum().rename(columns={"customerkey":"_ck","netamountextended":"key_net"})
    attrs=cust[cust["intercompanyflag"].astype("string").str.strip().eq("N")].copy()
    attrs=attrs[["customerkey","customergroup","customergroupname","creditmax","salesrep","InventSiteId","customerid"]].rename(columns={"customerkey":"_ck","InventSiteId":"inventsiteid"}).drop_duplicates("_ck",keep="first")
    kwa=rev.merge(attrs,on="_ck",how="inner")
    ko=kwa[kwa["customergroupname"].astype("string").str.strip().str.upper().isin(OEM_STOCKING)].copy()
    grp=ko.groupby(["harmonizedsoldtocustomername","zipcode","customergroupname"],dropna=False,as_index=False)["key_net"].sum().rename(columns={"key_net":"grp_net"})
    grp=grp.sort_values(["harmonizedsoldtocustomername","zipcode","grp_net","customergroupname"],ascending=[True,True,False,True])
    win_grp=grp.drop_duplicates(["harmonizedsoldtocustomername","zipcode"],keep="first")[["harmonizedsoldtocustomername","zipcode","customergroupname"]]
    wk=ko.merge(win_grp,on=["harmonizedsoldtocustomername","zipcode","customergroupname"],how="inner")
    wk=wk.sort_values(["harmonizedsoldtocustomername","zipcode","key_net","_ck"],ascending=[True,True,False,True])
    rows=[]
    for (name,z,group),g in wk.groupby(["harmonizedsoldtocustomername","zipcode","customergroupname"],dropna=False,sort=False):
        top=g.iloc[0]
        ids=sorted({str(x).strip() for x in g["customerid"].dropna() if str(x).strip()})
        rows.append({"harmonizedsoldtocustomername":name,"zipcode":z,"customergroupname":group,
                     "customergroup":str(top.get("customergroup","")).strip() or None,"creditmax":top.get("creditmax"),
                     "salesrep":str(top.get("salesrep","")).strip() or None,"servicingbranch":str(top.get("inventsiteid","")).strip() or None,
                     "customer_ids":", ".join(ids)})
    cust_attrs=pd.DataFrame(rows,columns=["harmonizedsoldtocustomername","zipcode","customergroupname","customergroup","creditmax","salesrep","servicingbranch","customer_ids"])
    if not cust_attrs.empty and cust_attrs.duplicated(["harmonizedsoldtocustomername","zipcode"]).any():
        raise AssertionError("cust_attrs fan-out detected")

    joined=after_f.merge(cust_attrs,on=["harmonizedsoldtocustomername","zipcode"],how="inner")
    _funnel(funnel,"Only OEM and Stocking Dealers",joined)

    filt=joined.copy()
    steps=[
        ("Invoicedate not null", lambda x:x["invoicedate"].notna()),
        ("Invoicedate >= 1 Jan, 2023", lambda x:x["invoicedate"]>=TRAIN_START_DATE),
        ("Intercompany = N", lambda x:x["intercompanyflag"].astype("string").str.strip().eq("N")),
        ("Netamountextended > 0", lambda x:x["netamountextended"]>0),
        ("Quantity > 0", lambda x:x["quantity"]>0),
        ("Dataareaid == dex (required)", lambda x:x["dataareaid"].astype("string").str.strip().str.lower().eq("dex")),
    ]
    for label,fn in steps:
        filt=filt[fn(filt)].copy(); _funnel(funnel,label,filt)
    base=filt.copy()
    base["purchase_dt"]=pd.to_datetime(base["invoicedate"],errors="coerce").dt.date
    base["salesorder_dt"]=pd.to_datetime(base["salesordercreateddate"],errors="coerce").dt.date
    base["is_positive_purchase"]=(base["netamountextended"]>0).astype(int)

    assert base["dataareaid"].astype("string").str.strip().str.lower().eq("dex").all()
    assert (base["netamountextended"]>0).all() and (base["quantity"]>0).all()
    assert base["invoicedate"].notna().all() and (base["invoicedate"]>=TRAIN_START_DATE).all()
    assert (~is_bad_name(base["harmonizedsoldtocustomername"])).all()
    if not cust_attrs.empty:
        assert cust_attrs["customergroupname"].astype("string").str.upper().isin(OEM_STOCKING).all()

    certified=root/"data"/"certified"; certified.mkdir(parents=True,exist_ok=True)
    p_base=write_df(base,certified/"dex_v2_base",preferred_format)
    p_attrs=write_df(cust_attrs,certified/"dex_v2_cust_attrs",preferred_format)
    process_date=datetime.now(timezone.utc).date().isoformat()
    if len(anomalies):
        anomaly_out=anomalies[[c for c in ["invoiceaccountcustomerkey","harmonizedsoldtocustomername","zipcode_raw","zipcode_normalized","city","state",country_col,"zipcode_validation_status","zipcode_anomaly_reason","zipcode_validation_rule","invoiceid","invoicedate","dataareaid"] if c in anomalies.columns]].copy()
        anomaly_out=anomaly_out.rename(columns={"invoiceaccountcustomerkey":"customer_key","harmonizedsoldtocustomername":"customer_name",country_col:"country","invoiceid":"invoice_id","invoicedate":"invoice_date"})
        anomaly_out["processing_timestamp"]=datetime.now(timezone.utc).isoformat(); anomaly_out["processing_date"]=process_date
    else:
        anomaly_out=pd.DataFrame()
    p_anom=write_df(anomaly_out,root/"data"/"edw_cache"/"zipcode_anomaly_eda"/f"processing_date={process_date}"/"data",preferred_format)
    funnel_df=pd.DataFrame(funnel)
    p_funnel=write_df(funnel_df,root/"data"/"reports"/"load_funnel",preferred_format)

    panel_end=str(max(base["purchase_dt"])) if len(base) else None
    result={"snapshot_week":week,"rows":{"dex_v2_base":len(base),"dex_v2_cust_attrs":len(cust_attrs),"zipcode_anomaly_eda":len(anomaly_out)},
            "panel_end_dt":panel_end,"paths":{"dex_v2_base":str(p_base),"dex_v2_cust_attrs":str(p_attrs),"zipcode_anomaly_eda":str(p_anom),"load_funnel":str(p_funnel)}}
    update_manifest(root,"load_data",result)
    return result
