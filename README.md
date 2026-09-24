# DexKo Customer Churn — Fully Offline Local Replica

This repository is a fully offline, synthetic-data replica of the DexKo Customer Churn solution. It requires no Databricks connection and contains no production customer data.

## Project status

All planned implementation phases are complete:

- Phase 1 — source-contract extraction: **complete**
- Phase 2 — synthetic EDW + behavioral generator: **complete**
- Phase 3 — Monday weekly snapshot layer: **complete**
- Phase 4 — certified load-data layer: **complete**
- Phase 5 — feature engineering: **complete**
- Phase 6 — H14 labels and eligibility: **complete**
- Phase 7 — XGBoost + local MLflow training: **complete**
- Phase 8 — weekly scoring + SHAP explainability: **complete**
- Phase 9 — PSI drift + delayed performance monitoring: **complete**
- Phase 10 — Excel / audit / CData business outputs: **complete**
- Phase 11 — DuckDB + Streamlit dashboard: **complete**

## End-to-end architecture

```text
Synthetic EDW
    ↓
Weekly snapshots + registry
    ↓
Certified load-data pipeline
    ↓
Feature engineering
    ↓
H14 labels + eligibility
    ↓
XGBoost training + MLflow
    ↓
Weekly scoring + SHAP
    ↓
Customer risk rollup + reasons
    ↓
PSI / performance monitoring
    ↓
Excel + audit + CData output
    ↓
DuckDB
    ↓
Streamlit dashboard
```

## Core model contract

- target grain: Customer × Business Unit × ZIP × Category × Snapshot
- churn horizon: 14 days
- observation / confirmation window: 180 days
- label maturity: 194 days
- live scoring gate: `score_eligible_cat_14`
- category model features: 44
- risk tiers:
  - High: `p_14 >= 0.40`
  - Medium: `0.20 <= p_14 < 0.40`
  - Low: `p_14 < 0.20`

The Customer × Business Unit × ZIP rollup uses trailing category spend as weights.

## Environment setup

```powershell
git clone https://github.com/sekhar234/Dexko-Churn-Project.git
cd Dexko-Churn-Project

py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Full local pipeline

Generate synthetic data:

```powershell
python scripts\generate_synthetic_data.py --scale small
```

Run and validate the weekly data layer:

```powershell
python scripts\run_monday.py
python scripts\validate_monday.py
```

Run feature engineering:

```powershell
python scripts\run_feature_engineering.py
```

Run labels and eligibility:

```powershell
python scripts\run_labels_eligibility.py
```

Train the H14 model:

```powershell
python scripts\run_model_training.py
python scripts\validate_model_training.py
```

Run weekly scoring:

```powershell
python scripts\run_weekly_scoring.py
python scripts\validate_weekly_scoring.py
```

Run monitoring:

```powershell
python scripts\run_monitoring.py
python scripts\validate_monitoring.py
```

Generate business outputs:

```powershell
python scripts\run_business_output.py
python scripts\validate_business_output.py
```

Build and validate the analytics database:

```powershell
python scripts\build_dashboard_db.py
python scripts\validate_dashboard.py
```

Run all lightweight tests:

```powershell
pytest -q
```

Launch the dashboard:

```powershell
python scripts\run_dashboard.py
```

## Business outputs

Phase 10 generates:

```text
outputs/
├── Weekly Call List MM-DD-YY.xlsx
└── Weekly Call List MM-DD-YY_audit.xlsx
```

It also creates:

`data/output/dex_v2_cdata_output_snapshot.parquet`

The main call list carries Customer ID, Customer, Customer Group, Ship-To, Business Unit, Branch, Sales Rep, Category, Risk Score, Risk Tier, Action, Annual Spend, Revenue at Risk, profile context, SHAP signals, and scoring lineage.

## Monitoring

PSI classification:

- `< 0.10` → STABLE
- `0.10–<0.25` → WATCH
- `>= 0.25` → ACT

Critical drift features:

- `recency_days`
- `median_cycle_days`
- `spend_0_30_g`

A retraining recommendation requires two distinct consecutive ACT snapshots on a critical feature. Automatic retraining remains disabled.

Actual performance monitoring waits until labels have matured for 194 days.

## Dashboard

The Streamlit application provides:

- Executive Overview
- Customer Workbench
- Customer 360
- Category Intelligence
- Risk Movement
- Model Health
- Pipeline Health

The local DuckDB database is generated at:

`data/dashboard/dexko_churn.duckdb`

See `docs/PHASE_11_DASHBOARD.md` for details.

## Important offline boundary

All customer and transaction data in this repository is synthetic.

External production-only integrations such as the SharePoint Account Owner mapping are not fabricated. The audit output explicitly identifies unavailable external mappings.
