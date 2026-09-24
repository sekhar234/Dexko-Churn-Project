from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.labels.pipeline import build_labels_eligibility

p = argparse.ArgumentParser(description="Build offline H14 labels and eligibility gates")
p.add_argument("--format", choices=["parquet", "csv"], default="parquet")
a = p.parse_args()

print(json.dumps(build_labels_eligibility(ROOT, a.format), indent=2, default=str))
