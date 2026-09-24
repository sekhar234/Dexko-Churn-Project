from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scoring.pipeline import run_weekly_scoring

p = argparse.ArgumentParser(description="Run offline weekly category churn scoring")
p.add_argument("--format", choices=["parquet", "csv"], default="parquet")
a = p.parse_args()

print(json.dumps(run_weekly_scoring(ROOT, a.format), indent=2, default=str))
