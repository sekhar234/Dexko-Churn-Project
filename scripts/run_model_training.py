from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modeling.pipeline import train_category_model

print(json.dumps(train_category_model(ROOT), indent=2, default=str))
