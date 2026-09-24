from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.monitoring.pipeline import run_monitoring

p = argparse.ArgumentParser(description="Run offline churn drift/performance monitoring")
p.add_argument("--format", choices=["parquet", "csv"], default="parquet")
a = p.parse_args()

print(json.dumps(run_monitoring(ROOT, a.format), indent=2, default=str))
