from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dashboard.data import DB_PATH, distinct_values, query, risk_movement, table_exists


st.set_page_config(
    page_title="DexKo Customer Churn",
    page_icon="📊",
    layout="wide",
)


def money(value) -> str:
    try:
        if pd.isna(value):
            return "$0"
        return "$" + f"{float(value):,.0f}"
    except Exception:
        return "$0"


def executive_overview() -> None:
    st.title("Executive Overview")
    meta = query("SELECT * FROM dashboard_metadata LIMIT 1").iloc[0]
    st.caption(
        f"Latest snapshot: {meta['latest_snapshot_dt']} · "
        f"Business output: {meta['business_output_status']} · "
        f"Monitoring: {meta['monitoring_status']}"
    )

    kpi = query(
        """
        SELECT
            COUNT(*) AS scored_categories,
            COUNT(DISTINCT harmonizedsoldtocustomername || '|' || bulevel1 || '|' || zipcode) AS customer_locations,
            SUM(spend_365) AS annual_spend,
            SUM(spend_365 * p_14) AS revenue_at_risk,
            AVG(p_14) AS mean_risk
        FROM latest_category_scores
        """
    ).iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Scored Categories", f"{int(kpi['scored_categories']):,}")
    c2.metric("Customer Locations", f"{int(kpi['customer_locations']):,}")
    c3.metric("Annual Category Spend", money(kpi["annual_spend"]))
    c4.metric("Category Rev at Risk", money(kpi["revenue_at_risk"]))
    c5.metric("Average Risk", f"{float(kpi['mean_risk']) * 100:.1f}%")

    left, right = st.columns(2)
    with left:
        st.subheader("Risk Tier Distribution")
        risk = query(
            """
            SELECT risk_tier, COUNT(*) AS rows
            FROM latest_category_scores
            GROUP BY risk_tier
            ORDER BY CASE risk_tier WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END
            """
        )
        st.bar_chart(risk.set_index("risk_tier")["rows"])
    with right:
        st.subheader("Action Distribution")
        action = query(
            """
            SELECT category_action AS action, COUNT(*) AS rows
            FROM latest_category_scores
            GROUP BY category_action
            ORDER BY rows DESC
            """
        )
        st.bar_chart(action.set_index("action")["rows"])

    st.subheader("Highest-Risk Customer Locations")
    top = query(
        """
        SELECT
            harmonizedsoldtocustomername AS Customer,
            bulevel1 AS "Business Unit",
            zipcode AS ZIP,
            risk_score AS "Risk Score",
            risk_tier AS "Risk Tier",
            value_band AS "Value Band",
            action AS Action,
            spend_365_all AS "Annual Spend",
            revenue_at_risk AS "Revenue at Risk",
            category_count_scored AS "Scored Categories"
        FROM latest_customer_rollup
        ORDER BY risk_score DESC, revenue_at_risk DESC
        LIMIT 25
        """
    )
    st.dataframe(
        top,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Annual Spend": st.column_config.NumberColumn(format="$%.0f"),
            "Revenue at Risk": st.column_config.NumberColumn(format="$%.0f"),
        },
    )


def customer_workbench() -> None:
    st.title("Customer Workbench")
    st.caption("Latest scored Customer × Business Unit × ZIP × Category population.")

    c1, c2, c3, c4 = st.columns(4)
    groups = c1.multiselect("Customer Group", distinct_values("customergroupname_current"))
    bus = c2.multiselect("Business Unit", distinct_values("bulevel1"))
    tiers = c3.multiselect("Risk Tier", distinct_values("risk_tier"))
    actions = c4.multiselect("Action", distinct_values("category_action"))
    categories = st.multiselect("Category", distinct_values("mgrl1"))

    clauses = []
    params: list[str] = []
    for column, values in [
        ("s.customergroupname_current", groups),
        ("s.bulevel1", bus),
        ("s.risk_tier", tiers),
        ("s.category_action", actions),
        ("s.mgrl1", categories),
    ]:
        if values:
            clauses.append(column + " IN (" + ",".join(["?"] * len(values)) + ")")
            params.extend(values)

    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    df = query(
        f"""
        SELECT
            COALESCE(a.customer_ids, '') AS "Customer ID",
            s.harmonizedsoldtocustomername AS Customer,
            s.customergroupname_current AS "Customer Group",
            s.bulevel1 AS "Business Unit",
            s.zipcode AS ZIP,
            COALESCE(a.servicingbranch, '') AS Branch,
            COALESCE(a.salesrep, s.salesrep_current, '') AS "Sales Rep",
            s.mgrl1 AS Category,
            ROUND(s.p_14 * 100) AS "Risk Score",
            s.risk_tier AS "Risk Tier",
            s.category_action AS Action,
            s.spend_365 AS "Annual Spend",
            s.spend_365 * s.p_14 AS "Category Rev at Risk",
            s.customer_risk_score AS "Customer Risk Score",
            s.customer_action AS "Customer Action"
        FROM latest_category_scores s
        LEFT JOIN customer_attrs a
          ON s.harmonizedsoldtocustomername = a.harmonizedsoldtocustomername
         AND s.zipcode = a.zipcode
        {where}
        ORDER BY s.p_14 DESC, s.spend_365 DESC
        """,
        tuple(params),
    )

    st.metric("Rows", f"{len(df):,}")
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Annual Spend": st.column_config.NumberColumn(format="$%.0f"),
            "Category Rev at Risk": st.column_config.NumberColumn(format="$%.0f"),
        },
    )


def customer_360() -> None:
    st.title("Customer 360")
    customers = query(
        """
        SELECT DISTINCT harmonizedsoldtocustomername AS customer
        FROM latest_category_scores
        ORDER BY customer
        """
    )["customer"].astype(str).tolist()
    customer = st.selectbox("Customer", customers)

    locations = query(
        """
        SELECT DISTINCT bulevel1, zipcode
        FROM latest_category_scores
        WHERE harmonizedsoldtocustomername = ?
        ORDER BY bulevel1, zipcode
        """,
        (customer,),
    )
    labels = [f"{r.bulevel1} | {r.zipcode}" for r in locations.itertuples(index=False)]
    selected = st.selectbox("Business Unit / ZIP", labels)
    bu, zipcode = [x.strip() for x in selected.split("|", 1)]

    roll = query(
        """
        SELECT *
        FROM latest_customer_rollup
        WHERE harmonizedsoldtocustomername = ?
          AND bulevel1 = ?
          AND CAST(zipcode AS VARCHAR) = ?
        LIMIT 1
        """,
        (customer, bu, zipcode),
    )
    if not roll.empty:
        r = roll.iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Customer Risk", f"{int(r['risk_score'])}")
        c2.metric("Risk Tier", str(r["risk_tier"]))
        c3.metric("Action", str(r["action"]))
        c4.metric("Annual Spend", money(r["spend_365_all"]))
        c5.metric("Revenue at Risk", money(r["revenue_at_risk"]))

    profile = query(
        """
        SELECT
            a.customer_ids,
            a.customergroupname AS customer_group,
            a.creditmax AS credit_limit,
            a.salesrep,
            a.servicingbranch,
            p.recent_peak_1,
            p.recent_peak_2,
            p.recent_peak_3,
            t.top_3_categories,
            t.top_80pct_categories
        FROM customer_attrs a
        LEFT JOIN peak_months p
          ON a.harmonizedsoldtocustomername = p.harmonizedsoldtocustomername
         AND a.zipcode = p.zipcode
        LEFT JOIN top_categories t
          ON a.harmonizedsoldtocustomername = t.harmonizedsoldtocustomername
         AND a.zipcode = t.zipcode
        WHERE a.harmonizedsoldtocustomername = ?
          AND CAST(a.zipcode AS VARCHAR) = ?
        LIMIT 1
        """,
        (customer, zipcode),
    )
    if not profile.empty:
        p = profile.iloc[0]
        peaks = ", ".join(
            [str(x) for x in [p["recent_peak_1"], p["recent_peak_2"], p["recent_peak_3"]] if pd.notna(x)]
        )
        st.subheader("Profile")
        st.write(
            "\n".join(
                [
                    f"**Customer ID:** {p['customer_ids']}",
                    f"**Customer Group:** {p['customer_group']}",
                    f"**Branch:** {p['servicingbranch']}",
                    f"**Sales Rep:** {p['salesrep']}",
                    f"**Credit Limit:** {money(p['credit_limit'])}",
                    f"**Peak Months:** {peaks or 'Unavailable'}",
                    f"**Top Categories:** {p['top_3_categories']}",
                ]
            )
        )

    st.subheader("Category Risk")
    cats = query(
        """
        SELECT
            mgrl1 AS Category,
            ROUND(p_14 * 100) AS "Risk Score",
            risk_tier AS "Risk Tier",
            category_action AS Action,
            spend_365 AS "Annual Spend",
            spend_365 * p_14 AS "Category Rev at Risk",
            recency_days AS "Recency Days",
            median_cycle_days AS "Median Cycle Days"
        FROM latest_category_scores
        WHERE harmonizedsoldtocustomername = ?
          AND bulevel1 = ?
          AND CAST(zipcode AS VARCHAR) = ?
        ORDER BY p_14 DESC, spend_365 DESC
        """,
        (customer, bu, zipcode),
    )
    st.dataframe(
        cats,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Annual Spend": st.column_config.NumberColumn(format="$%.0f"),
            "Category Rev at Risk": st.column_config.NumberColumn(format="$%.0f"),
        },
    )

    if not cats.empty:
        category = st.selectbox("Explain Category", cats["Category"].astype(str).tolist())
        reasons = query(
            """
            SELECT
                reason_rank AS Rank,
                reason_text AS Reason,
                feature_name AS Feature,
                feature_value AS Value,
                shap_value AS SHAP,
                direction AS Direction
            FROM reason_long
            WHERE harmonizedsoldtocustomername = ?
              AND bulevel1 = ?
              AND CAST(zipcode AS VARCHAR) = ?
              AND mgrl1 = ?
            ORDER BY reason_rank
            LIMIT 12
            """,
            (customer, bu, zipcode, category),
        )
        st.subheader("SHAP Reasons")
        st.dataframe(reasons, use_container_width=True, hide_index=True)


def category_intelligence() -> None:
    st.title("Category Intelligence")
    agg = query(
        """
        SELECT
            mgrl1 AS Category,
            COUNT(*) AS "Scored Rows",
            SUM(CASE WHEN risk_tier = 'High' THEN 1 ELSE 0 END) AS "High Risk",
            AVG(p_14) * 100 AS "Average Risk Score",
            SUM(spend_365) AS "Annual Spend",
            SUM(spend_365 * p_14) AS "Revenue at Risk"
        FROM latest_category_scores
        GROUP BY mgrl1
        ORDER BY "Revenue at Risk" DESC
        """
    )
    st.bar_chart(agg[["Category", "Revenue at Risk"]].set_index("Category"))
    st.dataframe(
        agg,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Average Risk Score": st.column_config.NumberColumn(format="%.1f"),
            "Annual Spend": st.column_config.NumberColumn(format="$%.0f"),
            "Revenue at Risk": st.column_config.NumberColumn(format="$%.0f"),
        },
    )

    if not agg.empty:
        category = st.selectbox("Category Detail", agg["Category"].astype(str).tolist())
        detail = query(
            """
            SELECT
                harmonizedsoldtocustomername AS Customer,
                bulevel1 AS "Business Unit",
                zipcode AS ZIP,
                customergroupname_current AS "Customer Group",
                ROUND(p_14 * 100) AS "Risk Score",
                risk_tier AS "Risk Tier",
                category_action AS Action,
                spend_365 AS "Annual Spend",
                spend_365 * p_14 AS "Revenue at Risk"
            FROM latest_category_scores
            WHERE mgrl1 = ?
            ORDER BY p_14 DESC, spend_365 DESC
            LIMIT 100
            """,
            (category,),
        )
        st.dataframe(
            detail,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Annual Spend": st.column_config.NumberColumn(format="$%.0f"),
                "Revenue at Risk": st.column_config.NumberColumn(format="$%.0f"),
            },
        )


def movement_page() -> None:
    st.title("Risk Movement")
    movement, previous, latest = risk_movement()
    if movement.empty:
        st.info(
            "Risk movement needs at least two distinct weekly scoring snapshots. "
            "Future weekly scoring runs will populate this view automatically."
        )
        return

    st.caption(f"Comparing {previous} → {latest}")
    counts = movement["Movement"].value_counts().rename_axis("Movement").reset_index(name="Rows")
    st.bar_chart(counts.set_index("Movement")["Rows"])
    st.dataframe(
        movement.sort_values("Risk Change", ascending=False, na_position="last"),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Previous Risk": st.column_config.NumberColumn(format="%.1f"),
            "Current Risk": st.column_config.NumberColumn(format="%.1f"),
            "Risk Change": st.column_config.NumberColumn(format="%+.1f"),
        },
    )


def model_health() -> None:
    st.title("Model Health")
    summary = query("SELECT * FROM latest_drift_summary LIMIT 1")
    if not summary.empty:
        s = summary.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Drift Status", str(s["overall_status"]))
        c2.metric("Max PSI", f"{float(s['max_psi']):.3f}")
        c3.metric("WATCH Features", f"{int(s['watch_count'])}")
        c4.metric("ACT Features", f"{int(s['act_count'])}")

    st.subheader("Latest Feature / Score PSI")
    drift = query(
        """
        SELECT
            feature_name AS Feature,
            metric_value AS PSI,
            status AS Status,
            is_critical AS Critical
        FROM latest_drift_metrics
        ORDER BY metric_value DESC
        """
    )
    st.bar_chart(drift.set_index("Feature")["PSI"])
    st.dataframe(drift, use_container_width=True, hide_index=True)

    retrain = query(
        """
        SELECT *
        FROM retraining_decisions
        ORDER BY scoring_snapshot DESC
        LIMIT 1
        """
    )
    if not retrain.empty:
        r = retrain.iloc[0]
        st.subheader("Retraining Governance")
        st.write(
            "\n".join(
                [
                    f"**Decision:** {r['decision']}",
                    f"**Confirmed trigger features:** {r['trigger_features'] or 'None'}",
                    f"**Automatic retraining enabled:** {bool(r['automatic_retraining_enabled'])}",
                ]
            )
        )

    st.subheader("Matured-Label Performance")
    if table_exists("performance_history"):
        perf = query("SELECT * FROM performance_history ORDER BY scoring_snapshot_dt")
        if not perf.empty:
            st.dataframe(perf, use_container_width=True, hide_index=True)
            st.line_chart(perf.set_index("scoring_snapshot_dt")[["auc_roc", "avg_precision"]])
        else:
            st.info("Performance history exists but has no evaluated snapshots yet.")
    else:
        status = query("SELECT performance_status FROM dashboard_metadata LIMIT 1").iloc[0, 0]
        st.info(
            f"Performance monitoring status: {status}. "
            "The H14 target needs 194 days of outcome maturity before AUC/AP can be evaluated."
        )


def pipeline_health() -> None:
    st.title("Pipeline Health")
    meta = query("SELECT * FROM dashboard_metadata LIMIT 1")
    st.dataframe(meta, use_container_width=True, hide_index=True)

    st.subheader("Pipeline Stages")
    stages = query(
        """
        SELECT stage AS Stage, status AS Status, snapshot_dt AS Snapshot, rows_json AS Rows
        FROM pipeline_stage_status
        ORDER BY stage
        """
    )
    st.dataframe(stages, use_container_width=True, hide_index=True)

    st.subheader("Reject Reasons")
    reject = query(
        """
        SELECT
            COALESCE(score_ineligible_reason_14, 'UNSPECIFIED') AS Reason,
            COUNT(*) AS Rows
        FROM reject_log
        GROUP BY 1
        ORDER BY Rows DESC
        """
    )
    st.bar_chart(reject.set_index("Reason")["Rows"])
    st.dataframe(reject, use_container_width=True, hide_index=True)

    if not meta.empty:
        m = meta.iloc[0]
        st.subheader("Generated Business Outputs")
        st.code(
            "Main workbook: " + str(m["main_workbook"]) + "\n"
            "Audit workbook: " + str(m["audit_workbook"])
        )


if not DB_PATH.exists():
    st.error("Dashboard database has not been built yet.")
    st.code("python scripts\\build_dashboard_db.py")
    st.stop()

st.sidebar.title("DexKo Churn")
page = st.sidebar.radio(
    "View",
    [
        "Executive Overview",
        "Customer Workbench",
        "Customer 360",
        "Category Intelligence",
        "Risk Movement",
        "Model Health",
        "Pipeline Health",
    ],
)
st.sidebar.caption("Offline synthetic-data replica")

if page == "Executive Overview":
    executive_overview()
elif page == "Customer Workbench":
    customer_workbench()
elif page == "Customer 360":
    customer_360()
elif page == "Category Intelligence":
    category_intelligence()
elif page == "Risk Movement":
    movement_page()
elif page == "Model Health":
    model_health()
else:
    pipeline_health()
