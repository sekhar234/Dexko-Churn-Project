from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import duckdb
import pandas as pd

from src.local_pipeline.io import read_df, resolve_table
from .contracts import REQUIRED_TABLES, OPTIONAL_TABLES


TABLE_SOURCES = {
    "category_scores": ("data", "scoring", "dex_v2_category_loss_scores_wide_zipcode"),
    "customer_rollup": ("data", "scoring", "dex_v2_customer_zip_risk_rollup"),
    "reason_long": ("data", "scoring", "dex_v2_category_reason_long_zipcode"),
    "scoring_history": ("data", "monitoring", "dex_v2_scoring_history"),
    "drift_metrics": ("data", "monitoring", "dex_v2_drift_metrics"),
    "drift_summary": ("data", "monitoring", "dex_v2_drift_summary"),
    "retraining_decisions": ("data", "monitoring", "dex_v2_retraining_decisions"),
    "cdata_output": ("data", "output", "dex_v2_cdata_output_snapshot"),
    "customer_attrs": ("data", "certified", "dex_v2_cust_attrs"),
    "peak_months": ("data", "features", "dex_v2_customer_peak_months_zipcode"),
    "top_categories": ("data", "features", "dex_v2_customer_top_categories_zipcode"),
    "reject_log": ("data", "scoring", "dex_v2_weekly_reject_log"),
    "performance_history": ("data", "monitoring", "dex_v2_performance_history"),
}


def _load_table(root: Path, parts: tuple[str, ...], required: bool) -> pd.DataFrame | None:
    base = root.joinpath(*parts)
    try:
        return read_df(resolve_table(base))
    except FileNotFoundError:
        if required:
            raise
        return None


def _stage_rows(manifest: dict) -> pd.DataFrame:
    rows = []
    for stage, payload in manifest.items():
        if not isinstance(payload, dict):
            continue
        rows.append(
            {
                "stage": stage,
                "status": payload.get("status", "COMPLETE"),
                "snapshot_dt": payload.get("snapshot_dt") or payload.get("panel_end_dt") or payload.get("snapshot_week"),
                "rows_json": json.dumps(payload.get("rows", {}), default=str),
                "details_json": json.dumps(payload, default=str),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["stage", "status", "snapshot_dt", "rows_json", "details_json"],
    )


def build_dashboard_database(root: str | Path) -> dict:
    root = Path(root)
    dashboard_dir = root / "data" / "dashboard"
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    db_path = dashboard_dir / "dexko_churn.duckdb"

    manifest_path = root / "data" / "local_pipeline_manifest.json"
    qa_path = root / "data" / "qa" / "business_output_qa.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    business_qa = json.loads(qa_path.read_text(encoding="utf-8")) if qa_path.exists() else {}

    loaded: dict[str, pd.DataFrame] = {}
    for table_name, parts in TABLE_SOURCES.items():
        required = table_name not in OPTIONAL_TABLES
        frame = _load_table(root, parts, required=required)
        if frame is not None:
            loaded[table_name] = frame

    category_scores = loaded["category_scores"].copy()
    category_scores["snapshot_dt"] = pd.to_datetime(category_scores["snapshot_dt"])
    latest_snapshot = category_scores["snapshot_dt"].max()

    metadata = pd.DataFrame(
        [
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "latest_snapshot_dt": str(latest_snapshot.date()),
                "business_output_status": business_qa.get("status", "UNKNOWN"),
                "monitoring_status": manifest.get("monitoring", {}).get("status", "UNKNOWN"),
                "performance_status": manifest.get("monitoring", {}).get("performance", {}).get("status", "UNKNOWN"),
                "scoring_rows": int(len(category_scores)),
                "customer_zip_rollup_rows": int(len(loaded["customer_rollup"])),
                "reject_rows": int(len(loaded["reject_log"])),
                "main_workbook": business_qa.get("paths", {}).get("main_workbook"),
                "audit_workbook": business_qa.get("paths", {}).get("audit_workbook"),
            }
        ]
    )
    stage_status = _stage_rows(manifest)

    con = duckdb.connect(str(db_path))
    try:
        con.execute("BEGIN TRANSACTION")
        for table_name, frame in loaded.items():
            con.register("_frame", frame)
            con.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM _frame')
            con.unregister("_frame")

        con.register("_metadata", metadata)
        con.execute('CREATE OR REPLACE TABLE "dashboard_metadata" AS SELECT * FROM _metadata')
        con.unregister("_metadata")

        con.register("_stage_status", stage_status)
        con.execute('CREATE OR REPLACE TABLE "pipeline_stage_status" AS SELECT * FROM _stage_status')
        con.unregister("_stage_status")

        con.execute(
            """
            CREATE OR REPLACE VIEW latest_category_scores AS
            SELECT *
            FROM category_scores
            WHERE snapshot_dt = (SELECT MAX(snapshot_dt) FROM category_scores)
            """
        )
        con.execute(
            """
            CREATE OR REPLACE VIEW latest_customer_rollup AS
            SELECT *
            FROM customer_rollup
            WHERE snapshot_dt = (SELECT MAX(snapshot_dt) FROM customer_rollup)
            """
        )
        con.execute(
            """
            CREATE OR REPLACE VIEW latest_drift_metrics AS
            SELECT *
            FROM drift_metrics
            WHERE scoring_snapshot = (SELECT MAX(scoring_snapshot) FROM drift_metrics)
            """
        )
        con.execute(
            """
            CREATE OR REPLACE VIEW latest_drift_summary AS
            SELECT *
            FROM drift_summary
            WHERE scoring_snapshot = (SELECT MAX(scoring_snapshot) FROM drift_summary)
            """
        )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()

    result = {
        "status": "PASS",
        "database": str(db_path),
        "latest_snapshot_dt": str(latest_snapshot.date()),
        "tables": sorted([*loaded.keys(), "dashboard_metadata", "pipeline_stage_status"]),
        "required_tables": REQUIRED_TABLES,
        "optional_tables_present": sorted(set(loaded) & set(OPTIONAL_TABLES)),
        "rows": {name: int(len(df)) for name, df in loaded.items()},
    }

    out = root / "data" / "qa"
    out.mkdir(parents=True, exist_ok=True)
    (out / "dashboard_db_qa.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
