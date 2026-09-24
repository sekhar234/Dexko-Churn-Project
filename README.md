# DexKo Customer Churn — Fully Offline Local Replica

This repository is a **fully offline, synthetic-data replica** of the current DexKo churn implementation. It requires no Databricks connection and contains no production customer data.

## Current status

- Phase 1 — source-contract extraction: **complete**
- Phase 2 — synthetic EDW + behavioral generator: **complete**
- Phase 3 — local Monday weekly snapshot layer: **complete**
- Phase 4 — local `02_load_data` certified-data layer: **complete**
- Phase 5 — feature engineering: **implemented on `feature/phase5-feature-engineering`**
- Phase 6 — eligibility / H14 labels: next
- Phase 7 — XGBoost / MLflow: pending
- Phase 8 — scoring / SHAP: pending
- Phase 9 — monitoring: pending
- Phase 10 — Excel/CData outputs: pending
- Phase 11 — Streamlit dashboard: pending

## Architecture

```text
Synthetic EDW
    ↓
Weekly snapshots + snapshot_registry
    ↓
master_weekly
    ↓
Certified load-data pipeline
    ↓
dex_v2_base + dex_v2_cust_attrs
    ↓
Feature engineering
    ↓
Eligibility + H14 labels
    ↓
XGBoost / local MLflow
    ↓
Scoring + SHAP
    ↓
Monitoring + outputs
    ↓
Streamlit dashboard
```

## Quick start

```powershell
git clone https://github.com/sekhar234/Dexko-Churn-Project.git
cd Dexko-Churn-Project
git checkout feature/phase5-feature-engineering

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For a fast complete local feature-engineering run, start with the new `tiny` profile:

```powershell
python scripts\generate_synthetic_data.py --scale tiny
python scripts\run_monday.py
python scripts\validate_monday.py
python scripts\run_feature_engineering.py
pytest -q
```

## Current generated outputs

```text
data/
├── source/edw/
├── edw_cache/
├── certified/
│   ├── dex_v2_base
│   └── dex_v2_cust_attrs
└── features/
    ├── dex_v2_checkpoint_billto_panel_zipcode
    ├── dex_v2_checkpoint_category_panel_zipcode
    ├── dex_v2_customer_peak_months_zipcode
    └── dex_v2_customer_top_categories_zipcode
```

## Feature contracts

The latest production configuration defines:

- **29 bill-to model features**
- **44 category model features**

The offline implementation mirrors those explicit contracts in `src/features/contracts.py`.

It also reproduces the Monday point-in-time visibility rule, running purchase-cycle history, rolling spend/frequency windows, momentum, price, lead-time, tenure, category breadth, AOV, volatility, category share/mix, peak-month, and Pareto-category logic.

## Local validation

A complete tiny end-to-end run with seed 42 produced:

- source invoice rows: **7,232**
- certified `dex_v2_base`: **4,872**
- certified `dex_v2_cust_attrs`: **28**
- bill-to panel rows: **20,320**
- category panel rows: **87,999**
- peak-month rows: **28**
- top-category rows: **28**
- feature counts: **29 bill-to / 44 category**
- grain assertions: **PASS**
- lightweight automated suite: **5 passed**

The original small-profile Monday reference remains approximately:

- source invoice rows: **36,277**
- certified `dex_v2_base`: **28,250**
- certified `dex_v2_cust_attrs`: **175**
- ZIP anomalies: **534**

These are synthetic engineering reference numbers, not production DexKo metrics.

## Performance guidance

Use `tiny` while developing the full historical feature pipeline. A multi-year weekly Customer × ZIP × Category scaffold expands quickly; the `small` and `medium` source profiles can generate hundreds of thousands or millions of feature-panel rows.

## Documented production-parity note

The current shared production exclusive frequency/spend-band wrapper omits `zipcode` from its grouping keys, while the latest feature-engineering notebook explicitly documents ZIP-grain fixes for other rolling features and joins.

The offline implementation includes `zipcode` for those band features so it preserves the stated Customer × ZIP grain. This is explicitly documented in `docs/PHASE_5_FEATURE_ENGINEERING.md`; no production code has been modified.

## Next phase

Phase 6 will add:

- historical label maturity
- H14 category-loss labels
- scoring eligibility
- `score_eligible_cat_14`
- `y_cat_activeacct_14`
- validation against the hidden synthetic behavioral ground truth

See:

- `docs/PHASE_3_4_IMPLEMENTATION.md`
- `docs/PHASE_5_FEATURE_ENGINEERING.md`
