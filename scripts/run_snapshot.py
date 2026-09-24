from pathlib import Path
import argparse,sys,json
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.local_pipeline.snapshot import run_snapshot
p=argparse.ArgumentParser(); p.add_argument('--as-of-date',default=None); p.add_argument('--format',choices=['parquet','csv'],default='parquet'); a=p.parse_args()
print(json.dumps(run_snapshot(ROOT,a.as_of_date,a.format),indent=2,default=str))
