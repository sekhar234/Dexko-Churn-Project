from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.output.pipeline import generate_business_output

p = argparse.ArgumentParser(description="Generate offline DexKo business outputs")
p.add_argument("--format", choices=["parquet", "csv"], default="parquet")
a = p.parse_args()

print(json.dumps(generate_business_output(ROOT, a.format), indent=2, default=str))
