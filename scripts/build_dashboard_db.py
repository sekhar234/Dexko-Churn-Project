from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dashboard.builder import build_dashboard_database

print(json.dumps(build_dashboard_database(ROOT), indent=2, default=str))
