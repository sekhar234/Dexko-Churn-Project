from pathlib import Path
import argparse,sys,json
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.local_pipeline.load_data import run_load_data
p=argparse.ArgumentParser(); p.add_argument('--format',choices=['parquet','csv'],default='parquet'); a=p.parse_args()
print(json.dumps(run_load_data(ROOT,a.format),indent=2,default=str))
