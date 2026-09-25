from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from src.dashboard.ui import (
    kpi_card,
    meta_strip,
    page_header,
    section_title,
)
from src.dashboard.upload import (
    SOURCE_TABLE_ORDER,
    list_staged_uploads,
    stage_upload_bundle,
    template_csv,
    validate_upload_bundle,
)
from src.synthetic.contracts import SOURCE_CONTRACTS


TABLE_LABELS = {
    "dimcustomer": "Customer Dimension",
    "factsalesinvoice": "Sales Invoice Fact",
    "dimproduct": "Product Dimension",
    "dimwarehouselocation": "Warehouse / BU Dimension",
    "dimcustomershipto": "Customer Ship-To Dimension",
}

TABLE_HELP = {
    "dimcustomer": "Customer identity, group, credit, sales-rep and account attributes.",
    "factsalesinvoice": "Transaction history used to build recency, spend, frequency and churn signals.",
    "dimproduct": "Product hierarchy used for MGR L1 category-level scoring.",
    "dimwarehouselocation": "Warehouse and Business Unit hierarchy.",
    "dimcustomershipto": "Ship-to location, ZIP, territory and servicing context.",
}


def _bundle_hash(uploads: dict[str, tuple[str, bytes]]) -> str:
    h = hashlib.sha256()
    for table in SOURCE_TABLE_ORDER:
        if table not in uploads:
            h.update(f"{table}:MISSING".encode("utf-8"))
            continue
        filename, payload = uploads[table]
        h.update(table.encode("utf-8"))
        h.update(filename.encode("utf-8"))
        h.update(payload)
    return h.hexdigest()


def _validation_rows(summary: dict) -> pd.DataFrame:
    rows = []
    for table in SOURCE_TABLE_ORDER:
        result = summary["tables"][table]
        for check in result.get("checks", []):
            rows.append(
                {
                    "Source Table": table,
                    "Check": check["check"],
                    "Status": check["status"],
                    "Count": check.get("count"),
                    "Detail": check["detail"],
                }
            )
    return pd.DataFrame(rows)


def render_upload_center(root: str | Path) -> None:
    root = Path(root)
    page_header(
        "Data Upload",
        "Upload the five EDW source tables, validate them against the pipeline contract, and stage a safe input bundle before running churn scoring.",
        eyebrow="DexKo Churn Application",
    )

    st.info(
        "Uploads are staged under data/uploads first. This screen does not overwrite "
        "the currently working data/source/edw files."
    )

    section_title(
        "1. Upload source tables",
        "CSV and Parquet are supported. Column names must match the source contract.",
    )

    uploads: dict[str, tuple[str, bytes]] = {}
    for table in SOURCE_TABLE_ORDER:
        with st.container(border=True):
            left, right = st.columns([3.2, 1])
            with left:
                st.markdown(f"#### {TABLE_LABELS[table]}")
                st.caption(f"{table} · {TABLE_HELP[table]}")
                uploaded = st.file_uploader(
                    f"Upload {table}",
                    type=["csv", "parquet"],
                    key=f"source_upload_{table}",
                    label_visibility="collapsed",
                )
                if uploaded is not None:
                    payload = uploaded.getvalue()
                    uploads[table] = (uploaded.name, payload)
                    st.caption(
                        f"Selected: **{uploaded.name}** · "
                        f"{len(payload) / (1024 * 1024):,.2f} MB"
                    )
            with right:
                st.download_button(
                    "Download CSV template",
                    data=template_csv(table),
                    file_name=f"{table}_template.csv",
                    mime="text/csv",
                    key=f"template_{table}",
                    use_container_width=True,
                )

            with st.expander("Required columns"):
                st.write(", ".join(SOURCE_CONTRACTS[table]))

    received = len(uploads)
    cols = st.columns(3)
    with cols[0]:
        kpi_card("Files Selected", f"{received}/5", "All five are required", "blue")
    with cols[1]:
        total_mb = sum(len(payload) for _, payload in uploads.values()) / (1024 * 1024)
        kpi_card("Upload Size", f"{total_mb:,.1f} MB", "Current browser selection", "cyan")
    with cols[2]:
        kpi_card(
            "Staging Safety",
            "Isolated",
            "Current source data remains unchanged",
            "green",
        )

    st.write("")
    validate_clicked = st.button(
        "Validate Upload Bundle",
        type="primary",
        use_container_width=True,
        disabled=received == 0,
    )

    current_hash = _bundle_hash(uploads)

    if validate_clicked:
        with st.spinner("Validating source contracts and data quality..."):
            summary, _ = validate_upload_bundle(uploads)
        st.session_state["upload_validation_summary"] = summary
        st.session_state["upload_validation_hash"] = current_hash

    summary = st.session_state.get("upload_validation_summary")
    validation_hash = st.session_state.get("upload_validation_hash")

    if summary is not None and validation_hash != current_hash:
        st.warning(
            "The selected files changed after the last validation. "
            "Run validation again before staging."
        )
        summary = None

    if summary is not None:
        section_title(
            "2. Validation results",
            "FAIL blocks staging. WARN is surfaced for review but does not prevent staging.",
        )

        table_summary = pd.DataFrame(
            [
                {
                    "Source Table": table,
                    "Status": summary["tables"][table]["status"],
                    "Rows": summary["tables"][table]["rows"],
                    "Columns": summary["tables"][table]["columns"],
                    "Missing Columns": len(summary["tables"][table]["missing_columns"]),
                    "Extra Columns": len(summary["tables"][table]["extra_columns"]),
                    "Duplicate Rows": summary["tables"][table]["duplicate_rows"],
                }
                for table in SOURCE_TABLE_ORDER
            ]
        )
        st.dataframe(
            table_summary,
            use_container_width=True,
            hide_index=True,
            height=245,
        )

        validation_df = _validation_rows(summary)
        status_filter = st.multiselect(
            "Show validation statuses",
            ["FAIL", "WARN", "PASS", "INFO"],
            default=["FAIL", "WARN"],
        )
        if status_filter:
            shown = validation_df[
                validation_df["Status"].isin(status_filter)
            ].copy()
        else:
            shown = validation_df
        st.dataframe(
            shown,
            use_container_width=True,
            hide_index=True,
            height=360,
        )

        cols = st.columns(4)
        with cols[0]:
            kpi_card(
                "Bundle Status",
                summary["status"],
                "Source-contract readiness",
                "green" if summary["ready_to_stage"] else "red",
            )
        with cols[1]:
            kpi_card(
                "Total Rows",
                f"{summary['total_rows']:,}",
                "Across all uploaded sources",
                "blue",
            )
        with cols[2]:
            kpi_card(
                "Warning Tables",
                str(summary["warning_tables"]),
                "Review before promotion",
                "amber",
            )
        with cols[3]:
            kpi_card(
                "Ready to Stage",
                "YES" if summary["ready_to_stage"] else "NO",
                "No source files overwritten",
                "green" if summary["ready_to_stage"] else "red",
            )

        if summary["ready_to_stage"]:
            st.success(
                "The bundle is structurally ready. Staging will copy these files "
                "into a new isolated upload run; it will not run the churn pipeline yet."
            )
        else:
            st.error(
                "Resolve the FAIL checks before staging. Missing files or required "
                "columns are the most common blockers."
            )

        stage_clicked = st.button(
            "Stage Validated Bundle",
            type="primary",
            use_container_width=True,
            disabled=not summary["ready_to_stage"],
        )
        if stage_clicked:
            with st.spinner("Creating isolated staged upload run..."):
                staged = stage_upload_bundle(root, uploads, summary)
            st.session_state["last_staged_upload"] = staged
            st.success(
                f"Staged successfully as {staged['run_id']}. "
                "The current production-like local source folder is unchanged."
            )
            meta_strip(
                [
                    ("Run ID", staged["run_id"]),
                    ("Status", staged["status"]),
                    ("Ready to Promote", str(staged["ready_to_promote"])),
                    ("Source Folder", staged["source_dir"]),
                ]
            )

    staged_runs = list_staged_uploads(root)
    section_title(
        "3. Recent staged uploads",
        "These are validated input bundles waiting for the future Promote / Run Pipeline step.",
    )
    if staged_runs:
        st.dataframe(
            pd.DataFrame(staged_runs),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No staged upload runs yet.")
