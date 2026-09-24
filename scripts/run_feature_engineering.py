from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.pipeline import build_features

p = argparse.ArgumentParser(description="Build offline bill-to and category feature panels")
p.add_argument("--format", choices=["parquet", "csv"], default="parquet")
a = p.parse_args()

print(json.dumps(build_features(ROOT, a.format), indent=2, default=str))
