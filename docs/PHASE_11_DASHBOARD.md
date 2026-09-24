# Phase 11 — DuckDB + Streamlit Dashboard

Phase 11 is the final analytics layer of the fully offline DexKo churn replica.

It does not retrain or rescore the model. It consumes the validated artifacts produced by Phases 8–10 and materializes them into a local DuckDB database for interactive Streamlit analytics.

## Architecture

```text
Phase 8 scoring + SHAP
        +
Phase 9 monitoring/history
        +
Phase 10 business outputs
        ↓
build_dashboard_db.py
        ↓
data/dashboard/dexko_churn.duckdb
        ↓
Streamlit
```

## Build the dashboard database

```powershell
python scripts\build_dashboard_db.py
```

The build creates:

`data/dashboard/dexko_churn.duckdb`

Core DuckDB tables:

- `category_scores`
- `customer_rollup`
- `reason_long`
- `scoring_history`
- `drift_metrics`
- `drift_summary`
- `retraining_decisions`
- `cdata_output`
- `customer_attrs`
- `peak_months`
- `top_categories`
- `reject_log`
- `dashboard_metadata`
- `pipeline_stage_status`

`performance_history` is optional until mature labels become available.

Convenience views:

- `latest_category_scores`
- `latest_customer_rollup`
- `latest_drift_metrics`
- `latest_drift_summary`

## Validate the dashboard data contract

```powershell
python scripts\validate_dashboard.py
```

Validation checks include:

- required DuckDB tables and views exist
- dashboard score rows reconcile to the Phase 10 call list
- customer rollup count reconciles
- reject-log count reconciles
- probabilities remain in [0,1]
- risk tiers are valid
- Customer × Business Unit × ZIP × Category × Snapshot grain is unique
- every scored row has explainability rows
- metadata snapshot matches the Phase 10 output
- Streamlit application exists

Risk Movement is considered ready only when at least two distinct weekly scoring snapshots exist.

## Launch the dashboard

Either run:

```powershell
python scripts\run_dashboard.py
```

or:

```powershell
python -m streamlit run dashboard\app.py
```

Streamlit will print the local browser URL.

## Dashboard pages

### Executive Overview

Shows:

- scored category count
- customer-location count
- annual category spend
- category revenue at risk
- mean churn risk
- risk-tier distribution
- action distribution
- highest-risk customer locations

### Customer Workbench

Interactive latest-snapshot filters for:

- Customer Group
- Business Unit
- Risk Tier
- Action
- Category

The result table includes identity, location, sales rep, category risk, spend, category revenue at risk, and customer rollup action.

### Customer 360

For one Customer + Business Unit + ZIP:

- customer risk score
- risk tier
- action
- annual spend
- revenue at risk
- Customer ID
- customer group
- branch
- sales rep
- credit limit
- peak months
- top categories
- per-category risk table
- detailed SHAP reasons

### Category Intelligence

Provides category-level:

- scored rows
- high-risk rows
- average risk
- annual spend
- revenue at risk
- top customer-category risks

### Risk Movement

Compares the latest two distinct snapshots in `scoring_history`.

Classifications:

- NEW
- DROPPED
- INCREASED
- DECREASED
- NO CHANGE

The first local run may contain only one snapshot. In that case the dashboard displays a valid waiting state rather than inventing movement.

### Model Health

Shows:

- current drift status
- maximum PSI
- WATCH / ACT counts
- per-feature and score PSI
- retraining recommendation
- automatic-retraining state
- matured-label performance history when available

Performance may remain in `BLOCKED_WAITING_FOR_LABELS` until the 194-day H14 label maturity window is satisfied.

### Pipeline Health

Shows:

- dashboard metadata
- pipeline stages
- reject reasons
- business-output workbook paths

## Offline boundaries

This dashboard uses only synthetic/local pipeline data.

The production SharePoint Account Owner mapping is not fabricated. The Phase 10 workbook and audit outputs continue to identify that external mapping as unavailable in the offline replica.

## Final Phase 11 commands

```powershell
python -m pip install -r requirements.txt
python scripts\build_dashboard_db.py
python scripts\validate_dashboard.py
pytest -q
python scripts\run_dashboard.py
```
