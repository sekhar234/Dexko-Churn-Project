from pathlib import Path
import shutil
import sys
import tempfile
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from src.synthetic.generator import load_config, generate, write_bundle
from src.local_pipeline.snapshot import run_snapshot
from src.local_pipeline.load_data import run_load_data
from src.local_pipeline.io import read_df, resolve_table, read_partition

@pytest.fixture(scope="module")
def local_run():
    src=Path(__file__).resolve().parents[1]
    root=Path(tempfile.mkdtemp(prefix="dexko_local_test_"))
    (root/"config").mkdir(parents=True)
    shutil.copy(src/"config"/"synthetic.yml",root/"config"/"synthetic.yml")
    cfg=load_config(root/"config"/"synthetic.yml")
    cfg["scales"]["test"]={"customers":40,"products":36,"warehouses":6}
    cfg["scale"]="test"; cfg["output_format"]="csv"; cfg["seed"]=42
    bundle=generate(cfg); write_bundle(bundle,root,"csv")
    snap=run_snapshot(root,preferred_format="csv")
    load=run_load_data(root,preferred_format="csv")
    yield root,snap,load
    shutil.rmtree(root,ignore_errors=True)


def test_snapshot_alignment_and_registry(local_run):
    root,res,_=local_run
    week=res["snapshot_week"]
    assert res["join_loss"] == 0
    assert res["counts"]["master_weekly"] == res["counts"]["jdbc_invoice_weekly"]
    for t in ["jdbc_customer_weekly","jdbc_invoice_weekly","jdbc_product_weekly","jdbc_warehouse_weekly","jdbc_shiptokey_weekly","master_weekly"]:
        df=read_partition(root/"data"/"edw_cache"/t,week)
        assert len(df)>0
        assert set(pd.to_datetime(df["snapshot_week"]).dt.date.astype(str)) == {week}
    reg=read_df(resolve_table(root/"data"/"edw_cache"/"snapshot_registry"))
    row=reg[(reg["dataset_name"]=="master_weekly") & (reg["status"]=="SUCCESS")]
    assert len(row)==1


def test_load_data_guardrails(local_run):
    root,_,res=local_run
    base=read_df(resolve_table(root/"data"/"certified"/"dex_v2_base"))
    attrs=read_df(resolve_table(root/"data"/"certified"/"dex_v2_cust_attrs"))
    assert len(base)==res["rows"]["dex_v2_base"] and len(base)>0
    assert len(attrs)==res["rows"]["dex_v2_cust_attrs"] and len(attrs)>0
    assert base["dataareaid"].astype(str).str.strip().str.lower().eq("dex").all()
    assert (pd.to_numeric(base["netamountextended"])>0).all()
    assert (pd.to_numeric(base["quantity"])>0).all()
    assert pd.to_datetime(base["invoicedate"],errors="coerce").notna().all()
    assert attrs["customergroupname"].astype(str).str.upper().isin(["OEM","STOCKING DEALER"]).all()
    assert attrs.duplicated(["harmonizedsoldtocustomername","zipcode"]).sum()==0
    assert attrs["customer_ids"].fillna("").astype(str).str.contains(r"[A-Za-z]",regex=True).sum()==0
    assert res["rows"]["zipcode_anomaly_eda"]>0
