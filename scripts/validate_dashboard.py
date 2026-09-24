from pathlib import Path
import json
import sys

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dashboard.contracts import REQUIRED_TABLES

qa_dir = ROOT / "data" / "qa"
db_qa = json.loads((qa_dir / "dashboard_db_qa.json").read_text(encoding="utf-8"))
business_qa = json.loads((qa_dir / "business_output_qa.json").read_text(encoding="utf-8"))
db_path = Path(db_qa["database"])

con = duckdb.connect(str(db_path), read_only=True)
try:
    tables = {
        r[0]
        for r in con.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
            """
        ).fetchall()
    }
    views = {
        r[0]
        for r in con.execute(
            """
            SELECT table_name
            FROM information_schema.views
            WHERE table_schema = 'main'
            """
        ).fetchall()
    }

    latest_score_rows = int(
        con.execute("SELECT COUNT(*) FROM latest_category_scores").fetchone()[0]
    )
    rollup_rows = int(
        con.execute("SELECT COUNT(*) FROM latest_customer_rollup").fetchone()[0]
    )
    reject_rows = int(
        con.execute("SELECT COUNT(*) FROM reject_log").fetchone()[0]
    )
    snapshot_count = int(
        con.execute(
            "SELECT COUNT(DISTINCT snapshot_dt) FROM scoring_history"
        ).fetchone()[0]
    )
    invalid_prob = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM latest_category_scores
            WHERE p_14 IS NULL OR p_14 < 0 OR p_14 > 1
            """
        ).fetchone()[0]
    )
    invalid_tier = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM latest_category_scores
            WHERE risk_tier NOT IN ('Low','Medium','High') OR risk_tier IS NULL
            """
        ).fetchone()[0]
    )
    duplicate_keys = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT
                    harmonizedsoldtocustomername,
                    bulevel1,
                    zipcode,
                    mgrl1,
                    snapshot_dt,
                    COUNT(*) AS n
                FROM latest_category_scores
                GROUP BY 1,2,3,4,5
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
    )
    reason_coverage = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM latest_category_scores s
            LEFT JOIN (
                SELECT DISTINCT
                    harmonizedsoldtocustomername,
                    bulevel1,
                    zipcode,
                    mgrl1,
                    snapshot_dt
                FROM reason_long
            ) r
              ON s.harmonizedsoldtocustomername = r.harmonizedsoldtocustomername
             AND s.bulevel1 = r.bulevel1
             AND s.zipcode = r.zipcode
             AND s.mgrl1 = r.mgrl1
             AND s.snapshot_dt = r.snapshot_dt
            WHERE r.harmonizedsoldtocustomername IS NULL
            """
        ).fetchone()[0]
    )
    metadata = con.execute(
        """
        SELECT
            latest_snapshot_dt,
            business_output_status,
            monitoring_status,
            scoring_rows,
            customer_zip_rollup_rows,
            reject_rows
        FROM dashboard_metadata
        LIMIT 1
        """
    ).fetchone()
finally:
    con.close()

required_views = {
    "latest_category_scores",
    "latest_customer_rollup",
    "latest_drift_metrics",
    "latest_drift_summary",
}

checks = {
    "dashboard_db_build_pass": db_qa["status"] == "PASS",
    "database_exists": db_path.exists(),
    "all_required_tables_present": set(REQUIRED_TABLES) <= tables,
    "all_required_views_present": required_views <= views,
    "latest_score_rows_match_business_output": (
        latest_score_rows == int(business_qa["rows"]["call_list"])
    ),
    "rollup_rows_match_business_output": (
        rollup_rows == int(business_qa["rows"]["customer_zip_rollup"])
    ),
    "reject_rows_match_business_output": (
        reject_rows == int(business_qa["rows"]["rejected"])
    ),
    "at_least_one_scoring_snapshot": snapshot_count >= 1,
    "probabilities_valid": invalid_prob == 0,
    "risk_tiers_valid": invalid_tier == 0,
    "score_grain_unique": duplicate_keys == 0,
    "all_scores_have_reason_rows": reason_coverage == 0,
    "metadata_snapshot_matches_business_output": (
        str(metadata[0]) == str(business_qa["snapshot_dt"])
    ),
    "metadata_business_output_pass": str(metadata[1]) == "PASS",
    "metadata_scoring_rows_match": int(metadata[3]) == latest_score_rows,
    "metadata_rollup_rows_match": int(metadata[4]) == rollup_rows,
    "metadata_reject_rows_match": int(metadata[5]) == reject_rows,
    "streamlit_app_exists": (ROOT / "dashboard" / "app.py").exists(),
}

result = {
    "status": "PASS" if all(checks.values()) else "FAIL",
    "checks": checks,
    "database": str(db_path),
    "snapshot_dt": business_qa["snapshot_dt"],
    "rows": {
        "latest_category_scores": latest_score_rows,
        "latest_customer_rollup": rollup_rows,
        "reject_log": reject_rows,
        "scoring_history_snapshots": snapshot_count,
    },
    "risk_movement_ready": snapshot_count >= 2,
    "performance_history_present": "performance_history" in tables,
}

payload = json.dumps(result, indent=2)
(qa_dir / "dashboard_validation.json").write_text(payload, encoding="utf-8")
print(payload)

if result["status"] != "PASS":
    raise SystemExit(1)
