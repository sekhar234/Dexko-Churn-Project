from pathlib import Path
import argparse, sys, json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.local_pipeline.snapshot import run_snapshot
from src.local_pipeline.load_data import run_load_data

p=argparse.ArgumentParser(description="Run offline Monday snapshot + load-data pipeline")
p.add_argument("--as-of-date", default=None)
p.add_argument("--format", choices=["parquet","csv"], default="parquet")
a=p.parse_args()
print("[1/2] Building weekly snapshot...")
s=run_snapshot(ROOT,a.as_of_date,a.format)
print(json.dumps(s,indent=2,default=str))
print("\n[2/2] Building certified churn inputs...")
l=run_load_data(ROOT,a.format)
print(json.dumps(l,indent=2,default=str))
print("\nSUCCESS: offline Monday pipeline complete")
