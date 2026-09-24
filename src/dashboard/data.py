from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "dashboard" / "dexko_churn.duckdb"


def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        return con.execute(sql, list(params)).df()
    finally:
        con.close()


def table_exists(name: str) -> bool:
    df = query(
        """
        SELECT COUNT(*) AS n
        FROM information_schema.tables
        WHERE table_name = ?
        """,
        (name,),
    )
    return bool(df.iloc[0]["n"])


def distinct_values(column: str) -> list[str]:
    allowed = {
        "customergroupname_current",
        "bulevel1",
        "risk_tier",
        "category_action",
        "mgrl1",
    }
    if column not in allowed:
        raise ValueError(column)
    df = query(
        f"""
        SELECT DISTINCT CAST({column} AS VARCHAR) AS value
        FROM latest_category_scores
        WHERE {column} IS NOT NULL
        ORDER BY value
        """
    )
    return df["value"].astype(str).tolist()


def latest_snapshot_count() -> int:
    return int(
        query("SELECT COUNT(*) AS n FROM latest_category_scores").iloc[0]["n"]
    )


def risk_movement() -> tuple[pd.DataFrame, object | None, object | None]:
    snapshots = query(
        """
        SELECT DISTINCT snapshot_dt
        FROM scoring_history
        ORDER BY snapshot_dt DESC
        """
    )
    if len(snapshots) < 2:
        return pd.DataFrame(), None, None

    latest = snapshots.iloc[0]["snapshot_dt"]
    previous = snapshots.iloc[1]["snapshot_dt"]
    movement = query(
        """
        WITH prev AS (
            SELECT *
            FROM scoring_history
            WHERE snapshot_dt = ?
        ),
        curr AS (
            SELECT *
            FROM scoring_history
            WHERE snapshot_dt = ?
        )
        SELECT
            COALESCE(curr.harmonizedsoldtocustomername, prev.harmonizedsoldtocustomername) AS Customer,
            COALESCE(curr.bulevel1, prev.bulevel1) AS "Business Unit",
            COALESCE(curr.zipcode, prev.zipcode) AS ZIP,
            COALESCE(curr.mgrl1, prev.mgrl1) AS Category,
            prev.p_14 * 100 AS "Previous Risk",
            curr.p_14 * 100 AS "Current Risk",
            (curr.p_14 - prev.p_14) * 100 AS "Risk Change",
            CASE
                WHEN prev.p_14 IS NULL THEN 'NEW'
                WHEN curr.p_14 IS NULL THEN 'DROPPED'
                WHEN curr.p_14 > prev.p_14 + 1e-12 THEN 'INCREASED'
                WHEN curr.p_14 < prev.p_14 - 1e-12 THEN 'DECREASED'
                ELSE 'NO CHANGE'
            END AS Movement
        FROM prev
        FULL OUTER JOIN curr
          ON prev.harmonizedsoldtocustomername = curr.harmonizedsoldtocustomername
         AND prev.bulevel1 = curr.bulevel1
         AND prev.zipcode = curr.zipcode
         AND prev.mgrl1 = curr.mgrl1
        """,
        (previous, latest),
    )
    return movement, previous, latest
