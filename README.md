# DexKo Customer Churn — Fully Offline Local Replica

This repository is a **fully offline, synthetic-data replica** of the current DexKo churn implementation. It requires no Databricks connection and contains no production customer data.

## Current status

- Phase 1 — source-contract extraction: **complete**
- Phase 2 — synthetic EDW + behavioral generator: **complete**
- Phase 3 — local Monday weekly snapshot layer: **complete**
- Phase 4 — local `02_load_data` certified-data layer: **complete**
- Phase 5 — feature engineering: **next**
- Phase 6 — eligibility / H14 labels: pending
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

## Synthetic source layer

The generator creates local equivalents of the five EDW tables used by the current churn pipeline:

- `dimcustomer`
- `factsalesinvoice`
- `dimproduct`
- `dimwarehouselocation`
- `dimcustomershipto`

It also generates behavioral scenarios including stable, growing, declining, category churn, account churn, seasonal, sporadic, new customer, price shock, lead-time problem, migration, and false alarm.

Generated datasets are intentionally **not committed to Git**. They are deterministic and rebuilt locally from seed 42 by default.

## Windows quick start

```powershell
git clone https://github.com/sekhar234/Dexko-Churn-Project.git
cd Dexko-Churn-Project
git checkout feature/offline-local-replica

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Generate a small synthetic dataset:

```powershell
python scripts\generate_synthetic_data.py --scale small
```

Run the local Monday pipeline:

```powershell
python scripts\run_monday.py
```

Validate it:

```powershell
python scripts\validate_monday.py
```

Run automated tests:

```powershell
pytest -q
```

## Current outputs

After the Monday pipeline, local generated data includes:

```text
data/
├── source/edw/
│   ├── dimcustomer
│   ├── factsalesinvoice
│   ├── dimproduct
│   ├── dimwarehouselocation
│   └── dimcustomershipto
├── edw_cache/
│   ├── jdbc_customer_weekly
│   ├── jdbc_invoice_weekly
│   ├── jdbc_product_weekly
│   ├── jdbc_warehouse_weekly
│   ├── jdbc_shiptokey_weekly
│   ├── master_weekly
│   ├── snapshot_registry
│   └── zipcode_anomaly_eda
└── certified/
    ├── dex_v2_base
    └── dex_v2_cust_attrs
```

## Deterministic small-run reference

Using seed 42, the verified local reference run produced approximately:

- source invoice rows: **36,277**
- master rows: **36,277**
- master join loss: **0.0%**
- ZIP anomalies routed: **534**
- certified `dex_v2_base`: **28,250** rows
- certified `dex_v2_cust_attrs`: **175** Customer × ZIP rows
- test suite: **4 passing tests**

These are engineering reference numbers for the synthetic dataset, not production DexKo metrics.

## Next phase

Phase 5 will consume only:

```text
data/certified/dex_v2_base
data/certified/dex_v2_cust_attrs
```

and create local equivalents of:

- `dex_v2_checkpoint_category_panel_zipcode`
- `dex_v2_checkpoint_billto_panel_zipcode`
- `dex_v2_customer_peak_months_zipcode`
- `dex_v2_customer_top_categories_zipcode`

The goal is to reproduce the Customer × ZIP × Category weekly grain and the rolling behavioral feature contracts from the latest production churn code.

See `docs/PHASE_3_4_IMPLEMENTATION.md` for the current production-to-local mapping.
