# Phase 5 — Offline Feature Engineering

This phase ports the current `03_feature_engineering.ipynb` behavior into a fully offline pandas implementation.

## Inputs

- `data/certified/dex_v2_base`
- `data/certified/dex_v2_cust_attrs`

## Outputs

- `dex_v2_checkpoint_billto_panel_zipcode`
- `dex_v2_checkpoint_category_panel_zipcode`
- `dex_v2_customer_peak_months_zipcode`
- `dex_v2_customer_top_categories_zipcode`

All generated outputs remain under `data/features/` and are excluded from Git.

## Production behavior reproduced

- Customer × ZIP × BU weekly bill-to scaffold
- Customer × ZIP × BU × Category weekly category scaffold
- Monday point-in-time visibility rule:
  - Monday purchase → same Monday
  - Tuesday-Sunday purchase → following Monday
- expanding purchase-gap history
- running observed median purchase cycle
- category-median / 30-day fallback for sparse category history
- 90-day, 180-day, 365-day rolling windows
- previous-period windows for 3m/6m momentum
- recency-to-cycle ratios
- weighted price-per-unit features
- order-to-invoice timing
- tenure
- category breadth
- average order value
- six-month spend volatility
- account/category share and mix-shift features
- top three buying months for recent/prior 12-month blocks
- trailing-365 category Pareto output
- OEM + Stocking Dealer panel scope

## Authoritative model feature contracts

The current source defines:

- **29 bill-to model features**
- **44 category model features**

These are defined explicitly in `src/features/contracts.py`.

## Validation completed

A local tiny end-to-end validation was run using 20 synthetic customers.

Results:

- synthetic invoice rows: 7,232
- certified `dex_v2_base`: 4,872
- certified `dex_v2_cust_attrs`: 28
- bill-to panel rows: 20,320
- category panel rows: 87,999
- peak-month rows: 28
- top-category rows: 28
- required bill-to feature count: 29
- required category feature count: 44
- feature-panel grain assertions: PASS
- unit suite after Phase 5 contract addition: 5 passed

The tiny scale is intended for full local feature-engineering development. Larger synthetic populations can create multi-million-row historical panels and should be used only when the machine has sufficient memory.

## Important parity note: exclusive frequency/spend bands

The current shared production helper wrappers call the exclusive band helper with:

- bill-to IDs: customer + BU
- category IDs: customer + BU + category

They do **not** include `zipcode`, while the latest feature-engineering notebook explicitly documents ZIP-grain fixes for the other rolling features and panel joins.

The offline port deliberately includes `zipcode` in the exclusive-band grouping so those features remain at the stated Customer × ZIP grain.

This is a documented local parity difference, not a silent production-code change. The production implementation should be separately reviewed before deciding whether its shared helper should also include ZIP.

## Run

```powershell
python scripts\generate_synthetic_data.py --scale tiny
python scripts\run_monday.py
python scripts\run_feature_engineering.py
pytest -q
```

## Next phase

Phase 6 will add:

- scoring eligibility
- historical training eligibility
- H14 category-loss label
- 194-day maturity rule
- `score_eligible_cat_14`
- `y_cat_activeacct_14`
- hidden-ground-truth validation against the synthetic behavioral scenarios
