from src.output.contracts import CALL_LIST_COLUMNS, VALID_ACTIONS


def test_phase10_call_list_contract():
    assert CALL_LIST_COLUMNS[:7] == [
        "Customer ID",
        "Customer",
        "Customer Group",
        "Ship-To",
        "Branch",
        "Sales Rep",
        "Category",
    ]
    assert "Risk Score" in CALL_LIST_COLUMNS
    assert "Annual Spend" in CALL_LIST_COLUMNS
    assert "Rev at Risk" in CALL_LIST_COLUMNS
    assert "Who They Are" in CALL_LIST_COLUMNS
    assert "Bold Signals" in CALL_LIST_COLUMNS
    assert "Grey Signals" in CALL_LIST_COLUMNS


def test_phase10_action_grid_values():
    assert len(VALID_ACTIONS) == 9
    assert "Urgent Save" in VALID_ACTIONS
    assert "Automate" in VALID_ACTIONS
