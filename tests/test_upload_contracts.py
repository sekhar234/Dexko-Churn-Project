import pandas as pd

from src.dashboard.upload import (
    SOURCE_TABLE_ORDER,
    stage_upload_bundle,
    template_csv,
    validate_source_table,
    validate_upload_bundle,
)
from src.synthetic.contracts import SOURCE_CONTRACTS


def _valid_frame(table: str) -> pd.DataFrame:
    row = {column: "VALUE" for column in SOURCE_CONTRACTS[table]}

    if table == "dimcustomer":
        row.update(
            {
                "Customer Key": "C1",
                "Customer Id": "1001",
                "Credit Max": 10000,
                "Intercompany Flag": "N",
                "Data Area Id": "dex",
            }
        )
    elif table == "factsalesinvoice":
        row.update(
            {
                "InvoiceDate": "2026-09-21",
                "SalesOrderCreatedDate": "2026-09-20",
                "SalesOrderDate": "2026-09-20",
                "NetAmountExtended": 100.0,
                "Quantity": 2,
                "InvoiceAccountCustomerKey": "C1",
                "Product Key": "P1",
                "Warehouse Location Key": "W1",
                "CustomerShipTokey": "S1",
                "ZipCode": "77001",
                "Country": "USA",
                "DataAreaId": "dex",
            }
        )
    elif table == "dimproduct":
        row["Product Key"] = "P1"
        row["MGR L1"] = "AXLE GROUP"
    elif table == "dimwarehouselocation":
        row["Warehouse Location Key"] = "W1"
        row["BU Level 1"] = "DDG - Ex - TWA"
    elif table == "dimcustomershipto":
        row["CustomerShipTokey"] = "S1"
        row["ZipCode"] = "77001"
        row["Country"] = "USA"

    return pd.DataFrame([row])


def _csv_uploads() -> dict[str, tuple[str, bytes]]:
    uploads = {}
    for table in SOURCE_TABLE_ORDER:
        frame = _valid_frame(table)
        uploads[table] = (
            f"{table}.csv",
            frame.to_csv(index=False).encode("utf-8"),
        )
    return uploads


def test_upload_template_matches_source_contract():
    header = template_csv("dimcustomer").decode("utf-8").strip().split(",")
    assert header == SOURCE_CONTRACTS["dimcustomer"]


def test_valid_source_table_is_ready():
    result = validate_source_table("dimproduct", _valid_frame("dimproduct"))
    assert result["status"] == "READY"
    assert result["missing_columns"] == []
    assert result["rows"] == 1


def test_missing_required_columns_fail():
    result = validate_source_table(
        "dimproduct",
        pd.DataFrame([{"Product Key": "P1"}]),
    )
    assert result["status"] == "FAIL"
    assert "MGR L1" in result["missing_columns"]


def test_complete_upload_bundle_is_ready():
    summary, frames = validate_upload_bundle(_csv_uploads())
    assert summary["ready_to_stage"] is True
    assert summary["tables_received"] == 5
    assert set(frames) == set(SOURCE_TABLE_ORDER)


def test_stage_upload_bundle_is_isolated(tmp_path):
    uploads = _csv_uploads()
    summary, _ = validate_upload_bundle(uploads)

    staged = stage_upload_bundle(tmp_path, uploads, summary)

    assert staged["run_id"].startswith("UPLOAD_")
    assert staged["ready_to_promote"] is True
    assert "data" in staged["source_dir"]
    assert "uploads" in staged["source_dir"]

    current_source = tmp_path / "data" / "source" / "edw"
    assert not current_source.exists()

    for table in SOURCE_TABLE_ORDER:
        assert (tmp_path / staged["files"][table]).exists()
