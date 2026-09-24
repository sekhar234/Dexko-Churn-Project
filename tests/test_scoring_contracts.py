import pandas as pd

from src.scoring.contracts import ACTION_GRID
from src.scoring.pipeline import _risk_tier


def test_phase8_action_grid_complete():
    assert len(ACTION_GRID) == 9
    assert ACTION_GRID[("High", "High")] == "Urgent Save"
    assert ACTION_GRID[("Low", "Low")] == "Automate"


def test_phase8_risk_thresholds():
    p = pd.Series([0.00, 0.1999, 0.20, 0.3999, 0.40, 0.95])
    assert _risk_tier(p).tolist() == [
        "Low", "Low", "Medium", "Medium", "High", "High"
    ]
