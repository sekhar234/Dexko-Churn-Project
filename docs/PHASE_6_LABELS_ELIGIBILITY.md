# Phase 6 — Labels and Eligibility

Phase 6 is deliberately separated from feature engineering.

## Inputs

- `data/features/dex_v2_checkpoint_billto_panel_zipcode`
- `data/features/dex_v2_checkpoint_category_panel_zipcode`
- `data/certified/dex_v2_base`

The synthetic ground-truth table is **not** used to construct labels or gates.

## Horizon contract

- Prediction horizon: **14 days**
- Observation/confirmation window: **180 days**
- Label maturity: **194 days**

For a snapshot at time `T`, the terminal outcome is evaluated from transactions after
`T + 14 days` through `T + 194 days`.

This preserves the intended "terminal within horizon" meaning: purchases may occur during
the first 14 days, but a category is considered retained only if it purchases again after
the horizon during the confirmation window.

## Category target

`y_cat_activeacct_14` is resolved only when the row is training-eligible and the whole
account remains active after the horizon.

- account retained after H, category retained after H -> **0**
- account retained after H, category not retained after H -> **1**
- account not retained after H -> **NULL**
- training-ineligible / immature row -> **NULL**

The future lookup is bounded to the 180-day confirmation window. It does not use an
unbounded global maximum purchase date.

## Training eligibility

`train_eligible_14` / compatibility alias `eligible_cat_14` requires:

1. full 194-day outcome maturity
2. prior category purchase
3. observed category cycle (not just fallback cycle)
4. `n_gaps_running >= 2`
5. `active_months_12_g >= 2`

`eligible_cat_14` must never be used as the live scoring gate.

## Scoring eligibility

`score_eligible_cat_14` is independent of label maturity and requires:

1. prior category purchase
2. `recency_days < 90`
3. `n_gaps_running >= 2`
4. `active_months_12_g >= 2`

This permits the latest unresolved snapshot to be scored.

## Outputs

- `data/labels/dex_v2_category_labeled_panel_zipcode`
- `data/labels/dex_v2_billto_labeled_panel_zipcode`
- `data/labels/training_gate_stage2`
- `data/qa/labels_eligibility_qa.json`

## Local commands

```powershell
python scripts\run_labels_eligibility.py
python scripts\validate_labels_eligibility.py
pytest -q
```

A successful validation must show `"status": "PASS"`.
