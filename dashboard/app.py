from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dashboard.data import DB_PATH, distinct_values, query, risk_movement, table_exists
from src.dashboard.ui import (
    MOVEMENT_COLORS,
    RISK_COLORS,
    COLORS,
    donut_chart,
    horizontal_bar,
    info_card,
    inject_css,
    kpi_card,
    meta_strip,
    page_header,
    psi_chart,
    risk_gauge,
    section_title,
    status_badge,
)


st.set_page_config(
    page_title="DexKo Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def money(value) -> str:
    try:
        if pd.isna(value):
            return "$0"
        return "$" + f"{float(value):,.0f}"
    except Exception:
        return "$0"


def number(value) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return "0"


def executive_overview() -> None:
    meta = query("SELECT * FROM dashboard_metadata LIMIT 1").iloc[0]
    page_header(
        "Executive Overview",
        "A business-first view of current churn exposure, customer value, and recommended retention actions.",
    )
    meta_strip(
        [
            ("Snapshot", str(meta["latest_snapshot_dt"])),
            ("Business Output", str(meta["business_output_status"])),
            ("Monitoring", str(meta["monitoring_status"])),
            ("Model", "H14 Category Churn"),
        ]
    )

    kpi = query(
        """
        SELECT
            COUNT(*) AS scored_categories,
            COUNT(DISTINCT harmonizedsoldtocustomername || '|' || bulevel1 || '|' || zipcode) AS customer_locations,
            SUM(spend_365) AS annual_spend,
            SUM(spend_365 * p_14) AS revenue_at_risk,
            SUM(CASE WHEN risk_tier = 'High' THEN 1 ELSE 0 END) AS high_risk,
            SUM(CASE WHEN category_action = 'Urgent Save' THEN 1 ELSE 0 END) AS urgent_save
        FROM latest_category_scores
        """
    ).iloc[0]

    cols = st.columns(6)
    cards = [
        ("Scored Categories", number(kpi["scored_categories"]), "Latest eligible category rows", "blue"),
        ("Customer Locations", number(kpi["customer_locations"]), "Customer × BU × ZIP rollups", "cyan"),
        ("Annual Spend", money(kpi["annual_spend"]), "Scored category spend", "teal"),
        ("Revenue at Risk", money(kpi["revenue_at_risk"]), "Category spend × p14", "red"),
        ("High Risk", number(kpi["high_risk"]), "Risk probability ≥ 40%", "amber"),
        ("Urgent Save", number(kpi["urgent_save"]), "High-value, high-risk actions", "purple"),
    ]
    for col, card in zip(cols, cards):
        with col:
            kpi_card(*card)

    st.write("")
    left, right = st.columns([1, 1.35])
    with left:
        section_title("Risk mix", "Distribution of latest category-level risk tiers.")
        risk = query(
            """
            SELECT risk_tier, COUNT(*) AS rows
            FROM latest_category_scores
            GROUP BY risk_tier
            ORDER BY CASE risk_tier WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END
            """
        )
        st.plotly_chart(
            donut_chart(risk, "risk_tier", "rows", RISK_COLORS),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with right:
        section_title("Recommended action mix", "How the 3×3 value/risk grid translates into next actions.")
        action = query(
            """
            SELECT category_action AS action, COUNT(*) AS rows
            FROM latest_category_scores
            GROUP BY category_action
            ORDER BY rows DESC
            """
        )
        st.plotly_chart(
            horizontal_bar(action, "action", "rows", color=COLORS["blue"], height=330),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    section_title(
        "Priority customer locations",
        "Highest customer-level risk scores, then highest revenue at risk.",
    )
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
        height=520,
        column_config={
            "Risk Score": st.column_config.ProgressColumn(
                "Risk Score",
                min_value=0,
                max_value=100,
                format="%d",
            ),
            "Annual Spend": st.column_config.NumberColumn(format="$%.0f"),
            "Revenue at Risk": st.column_config.NumberColumn(format="$%.0f"),
        },
    )


def customer_workbench() -> None:
    page_header(
        "Customer Workbench",
        "Slice the latest scored population and turn model output into a practical retention call list.",
    )

    with st.expander("Filters", expanded=True):
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

    high_risk = int((df["Risk Tier"] == "High").sum()) if not df.empty else 0
    unique_customers = int(df["Customer"].nunique()) if not df.empty else 0
    annual = float(df["Annual Spend"].sum()) if not df.empty else 0.0
    rar = float(df["Category Rev at Risk"].sum()) if not df.empty else 0.0

    cols = st.columns(4)
    summary_cards = [
        ("Filtered Rows", number(len(df)), "Category-level records", "blue"),
        ("Customers", number(unique_customers), "Distinct customer names", "cyan"),
        ("High Risk", number(high_risk), "Within current filters", "red"),
        ("Revenue at Risk", money(rar), f"Across {money(annual)} annual spend", "amber"),
    ]
    for col, card in zip(cols, summary_cards):
        with col:
            kpi_card(*card)

    st.write("")
    section_title("Filtered call list", "Sorted by category risk, then annual spend.")
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=650,
        column_config={
            "Risk Score": st.column_config.ProgressColumn(
                "Risk Score",
                min_value=0,
                max_value=100,
                format="%d",
            ),
            "Customer Risk Score": st.column_config.ProgressColumn(
                "Customer Risk Score",
                min_value=0,
                max_value=100,
                format="%d",
            ),
            "Annual Spend": st.column_config.NumberColumn(format="$%.0f"),
            "Category Rev at Risk": st.column_config.NumberColumn(format="$%.0f"),
        },
    )


def customer_360() -> None:
    page_header(
        "Customer 360",
        "One place to understand customer value, category risk, account context, and the model drivers behind the signal.",
    )

    customers = query(
        """
        SELECT DISTINCT harmonizedsoldtocustomername AS customer
        FROM latest_category_scores
        ORDER BY customer
        """
    )["customer"].astype(str).tolist()

    s1, s2 = st.columns([1.8, 1])
    customer = s1.selectbox("Customer", customers)

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
    selected = s2.selectbox("Business Unit / ZIP", labels)
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

    if not roll.empty:
        r = roll.iloc[0]
        left, right = st.columns([1, 1.8])
        with left:
            st.plotly_chart(
                risk_gauge(float(r["risk_score"])),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        with right:
            meta_strip(
                [
                    ("Risk Tier", str(r["risk_tier"])),
                    ("Value Band", str(r["value_band"])),
                    ("Recommended Action", str(r["action"])),
                ]
            )
            cols = st.columns(3)
            with cols[0]:
                kpi_card("Annual Spend", money(r["spend_365_all"]), "Customer-level trailing spend", "teal")
            with cols[1]:
                kpi_card("Revenue at Risk", money(r["revenue_at_risk"]), "Spend × weighted risk", "red")
            with cols[2]:
                kpi_card("Scored Categories", number(r["category_count_scored"]), "Eligible category signals", "blue")

    if not profile.empty:
        p = profile.iloc[0]
        peaks = ", ".join(
            [
                str(x)
                for x in [p["recent_peak_1"], p["recent_peak_2"], p["recent_peak_3"]]
                if pd.notna(x)
            ]
        )
        section_title("Account profile", "Business context carried from the certified customer attributes.")
        info_card(
            [
                ("Customer ID", p["customer_ids"]),
                ("Customer Group", p["customer_group"]),
                ("Branch", p["servicingbranch"]),
                ("Sales Rep", p["salesrep"]),
                ("Credit Limit", money(p["credit_limit"])),
                ("Peak Buying Months", peaks or "Unavailable"),
                ("Top 3 Categories", p["top_3_categories"]),
                ("80% Spend Categories", p["top_80pct_categories"]),
            ]
        )

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

    tab1, tab2 = st.tabs(["Category Risk", "Explainability"])
    with tab1:
        section_title("Category portfolio", "Risk, spend, and cadence at the customer-location level.")
        if not cats.empty:
            chart_df = cats.sort_values("Category Rev at Risk", ascending=False).head(12)
            st.plotly_chart(
                horizontal_bar(
                    chart_df,
                    "Category",
                    "Category Rev at Risk",
                    color=COLORS["red"],
                    height=max(320, 34 * len(chart_df)),
                    currency=True,
                ),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        st.dataframe(
            cats,
            use_container_width=True,
            hide_index=True,
            height=450,
            column_config={
                "Risk Score": st.column_config.ProgressColumn(
                    "Risk Score",
                    min_value=0,
                    max_value=100,
                    format="%d",
                ),
                "Annual Spend": st.column_config.NumberColumn(format="$%.0f"),
                "Category Rev at Risk": st.column_config.NumberColumn(format="$%.0f"),
            },
        )

    with tab2:
        if cats.empty:
            st.info("No scored categories are available for this customer-location.")
        else:
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
            section_title(
                "Why the model moved",
                "Top SHAP drivers for the selected customer-location-category prediction.",
            )
            st.dataframe(reasons, use_container_width=True, hide_index=True, height=470)


def category_intelligence() -> None:
    page_header(
        "Category Intelligence",
        "See where churn exposure is concentrated across product categories and which customers contribute most to revenue at risk.",
    )

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

    total_rar = float(agg["Revenue at Risk"].sum()) if not agg.empty else 0.0
    total_spend = float(agg["Annual Spend"].sum()) if not agg.empty else 0.0
    high_rows = int(agg["High Risk"].sum()) if not agg.empty else 0

    cols = st.columns(4)
    summary_cards = [
        ("Categories", number(len(agg)), "Scored product groups", "blue"),
        ("Annual Spend", money(total_spend), "Across scored categories", "teal"),
        ("Revenue at Risk", money(total_rar), "Category-level exposure", "red"),
        ("High-Risk Rows", number(high_rows), "p14 ≥ 40%", "amber"),
    ]
    for col, card in zip(cols, summary_cards):
        with col:
            kpi_card(*card)

    st.write("")
    section_title("Revenue at risk by category", "Largest contributors shown first.")
    chart_df = agg.head(15).copy()
    st.plotly_chart(
        horizontal_bar(
            chart_df,
            "Category",
            "Revenue at Risk",
            color=COLORS["red"],
            height=max(380, 34 * len(chart_df)),
            currency=True,
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.dataframe(
        agg,
        use_container_width=True,
        hide_index=True,
        height=420,
        column_config={
            "Average Risk Score": st.column_config.NumberColumn(format="%.1f"),
            "Annual Spend": st.column_config.NumberColumn(format="$%.0f"),
            "Revenue at Risk": st.column_config.NumberColumn(format="$%.0f"),
        },
    )

    if not agg.empty:
        section_title("Customer detail", "Drill into the customers contributing to a selected category.")
        category = st.selectbox("Category", agg["Category"].astype(str).tolist())
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
            height=520,
            column_config={
                "Risk Score": st.column_config.ProgressColumn(
                    "Risk Score",
                    min_value=0,
                    max_value=100,
                    format="%d",
                ),
                "Annual Spend": st.column_config.NumberColumn(format="$%.0f"),
                "Revenue at Risk": st.column_config.NumberColumn(format="$%.0f"),
            },
        )


def movement_page() -> None:
    page_header(
        "Risk Movement",
        "Track how category-level risk changes from one weekly scoring snapshot to the next.",
    )
    movement, previous, latest = risk_movement()
    if movement.empty:
        st.info(
            "Risk movement needs at least two distinct weekly scoring snapshots. "
            "Future weekly scoring runs will populate this view automatically."
        )
        return

    meta_strip([("Previous", str(previous)), ("Current", str(latest))])
    counts = (
        movement["Movement"]
        .value_counts()
        .rename_axis("Movement")
        .reset_index(name="Rows")
    )

    cols = st.columns(5)
    for col, movement_name in zip(
        cols,
        ["INCREASED", "DECREASED", "NEW", "DROPPED", "NO CHANGE"],
    ):
        row = counts[counts["Movement"] == movement_name]
        count = int(row["Rows"].iloc[0]) if not row.empty else 0
        accent = {
            "INCREASED": "red",
            "DECREASED": "green",
            "NEW": "blue",
            "DROPPED": "slate",
            "NO CHANGE": "cyan",
        }[movement_name]
        with col:
            kpi_card(movement_name.title(), number(count), "Category rows", accent)

    st.write("")
    left, right = st.columns([.85, 1.65])
    with left:
        section_title("Movement mix", "Row counts by weekly movement class.")
        st.plotly_chart(
            horizontal_bar(
                counts,
                "Movement",
                "Rows",
                color_col="Movement",
                color_map=MOVEMENT_COLORS,
                height=330,
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with right:
        section_title("Largest risk changes", "Sorted from strongest increase to strongest decrease.")
        st.dataframe(
            movement.sort_values("Risk Change", ascending=False, na_position="last"),
            use_container_width=True,
            hide_index=True,
            height=470,
            column_config={
                "Previous Risk": st.column_config.NumberColumn(format="%.1f"),
                "Current Risk": st.column_config.NumberColumn(format="%.1f"),
                "Risk Change": st.column_config.NumberColumn(format="%+.1f"),
            },
        )


def model_health() -> None:
    page_header(
        "Model Health",
        "Monitor feature drift, score stability, delayed performance, and retraining governance.",
    )

    summary = query("SELECT * FROM latest_drift_summary LIMIT 1")
    if not summary.empty:
        s = summary.iloc[0]
        cols = st.columns(4)
        cards = [
            ("Drift Status", str(s["overall_status"]), "Latest scoring snapshot", "green" if str(s["overall_status"]) == "STABLE" else "amber"),
            ("Maximum PSI", f"{float(s['max_psi']):.3f}", "Across monitored features", "blue"),
            ("WATCH Features", number(s["watch_count"]), "0.10 ≤ PSI < 0.25", "amber"),
            ("ACT Features", number(s["act_count"]), "PSI ≥ 0.25", "red"),
        ]
        for col, card in zip(cols, cards):
            with col:
                kpi_card(*card)

    st.write("")
    section_title(
        "Latest PSI by feature",
        "Dashed lines mark WATCH (0.10) and ACT (0.25) governance thresholds.",
    )
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
    st.plotly_chart(
        psi_chart(drift),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.dataframe(drift, use_container_width=True, hide_index=True, height=420)

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
        section_title("Retraining governance", "Recommendations are advisory; automatic retraining remains disabled.")
        meta_strip(
            [
                ("Decision", str(r["decision"])),
                ("Confirmed Triggers", str(r["trigger_features"] or "None")),
                ("Automatic Retraining", str(bool(r["automatic_retraining_enabled"]))),
            ]
        )

    section_title("Matured-label performance", "AUC/AP appear only after the full 194-day H14 maturity window.")
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
    page_header(
        "Pipeline Health",
        "Operational view of the local end-to-end pipeline, validation status, rejects, and generated business artifacts.",
    )
    meta = query("SELECT * FROM dashboard_metadata LIMIT 1")
    if not meta.empty:
        m = meta.iloc[0]
        cols = st.columns(4)
        cards = [
            ("Business Output", str(m["business_output_status"]), "Phase 10 QA status", "green"),
            ("Monitoring", str(m["monitoring_status"]), "Phase 9 monitoring status", "blue"),
            ("Scoring Rows", number(m["scoring_rows"]), "Latest category scores", "cyan"),
            ("Rejected Rows", number(m["reject_rows"]), "Excluded from latest scoring", "amber"),
        ]
        for col, card in zip(cols, cards):
            with col:
                kpi_card(*card)

    st.write("")
    section_title("Pipeline stages", "Status and lineage captured from the local pipeline manifest.")
    stages = query(
        """
        SELECT stage AS Stage, status AS Status, snapshot_dt AS Snapshot, rows_json AS Rows
        FROM pipeline_stage_status
        ORDER BY stage
        """
    )
    st.dataframe(stages, use_container_width=True, hide_index=True, height=380)

    section_title("Reject reasons", "Why rows did not enter the latest scoring population.")
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
    st.plotly_chart(
        horizontal_bar(
            reject.head(15),
            "Reason",
            "Rows",
            color=COLORS["amber"],
            height=max(320, 34 * min(len(reject), 15)),
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.dataframe(reject, use_container_width=True, hide_index=True, height=330)

    if not meta.empty:
        m = meta.iloc[0]
        section_title("Generated business outputs", "Latest Excel artifacts created by the validated output pipeline.")
        info_card(
            [
                ("Main Workbook", m["main_workbook"]),
                ("Audit Workbook", m["audit_workbook"]),
                ("Latest Snapshot", m["latest_snapshot_dt"]),
                ("Performance Status", m["performance_status"]),
            ]
        )


if not DB_PATH.exists():
    st.error("Dashboard database has not been built yet.")
    st.code("python scripts\\build_dashboard_db.py")
    st.stop()

inject_css(ROOT / "dashboard" / "styles.css")

meta = query("SELECT * FROM dashboard_metadata LIMIT 1").iloc[0]

st.sidebar.markdown("## ◈ DexKo Churn")
st.sidebar.caption("Customer Success Intelligence")
st.sidebar.markdown("---")

nav = {
    "◉ Executive Overview": "Executive Overview",
    "⌕ Customer Workbench": "Customer Workbench",
    "◎ Customer 360": "Customer 360",
    "▦ Category Intelligence": "Category Intelligence",
    "↕ Risk Movement": "Risk Movement",
    "◇ Model Health": "Model Health",
    "✓ Pipeline Health": "Pipeline Health",
}
selected_nav = st.sidebar.radio("Navigate", list(nav.keys()), label_visibility="collapsed")
page = nav[selected_nav]

st.sidebar.markdown("---")
st.sidebar.caption("LATEST SNAPSHOT")
st.sidebar.markdown(f"**{meta['latest_snapshot_dt']}**")
st.sidebar.markdown(
    status_badge(str(meta["business_output_status"])),
    unsafe_allow_html=True,
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
