# Phase 8 — Weekly Scoring, Risk Rollup and Reasons

Phase 8 scores the latest weekly category panel using the locally persisted Phase 7 model.

## Scoring population

1. Read `data/labels/dex_v2_category_labeled_panel_zipcode`.
2. Select the latest `snapshot_dt`.
3. Score only rows where `score_eligible_cat_14 == 1`.
4. Persist all other latest-snapshot rows to the reject log with their Phase 6 rejection reason.

Training eligibility and label maturity are not used as live-scoring gates.

## Model scoring

Phase 8 loads:

- `artifacts/models/category_churn_h14/model.json`
- `artifacts/models/category_churn_h14/encoder.json`
- `artifacts/models/category_churn_h14/metadata.json`

The saved training encoder is applied in the original encoded feature order and native XGBoost produces raw `p_14` probabilities.

Current verified production scoring also uses raw XGBoost probability; probability calibration is not active.

## Risk tiers

- High: `p_14 >= 0.40`
- Medium: `0.20 <= p_14 < 0.40`
- Low: `p_14 < 0.20`

The same boundaries are applied to customer×ZIP weighted risk.

## Customer × ZIP weighted risk

For each scored customer×BU×ZIP:

```
weighted_risk =
    SUM(p_14 * spend_365_category)
    / SUM(spend_365_category)

risk_score = round(weighted_risk * 100)

revenue_at_risk =
    spend_365_all * weighted_risk
```

Only score-eligible categories participate in the weighted-risk calculation.

## Value bands

The current verified production method uses relative terciles across the scored customer×ZIP population.

The local replica therefore ranks `spend_365_all` and creates three near-equal bands:

- Low
- Mid
- High

The proposed fixed $50K / $250K thresholds are not used in this phase because they are not the verified current Excel implementation.

## Action grid

| Value | Low Risk | Medium Risk | High Risk |
|---|---|---|---|
| High | Monitor | Protect | Urgent Save |
| Mid | Maintain | Engage | Retain |
| Low | Automate | Watch | Evaluate |

Both customer-level and category-level actions are persisted.

## Explainability

No separate SHAP package is required.

Native XGBoost `pred_contribs=True` produces SHAP contribution values using the same fitted Booster.

Encoded categorical contributions are aggregated back to their raw feature. Static categorical inputs are excluded from business reason ranking.

Outputs include:

- top 3 readable reasons per scored category
- complete long-form feature contribution table
- raw feature name
- raw feature value
- SHAP value
- absolute SHAP magnitude
- risk-up / risk-down direction

## Outputs

Generated under `data/scoring/`:

- `dex_v2_category_loss_scores_wide_zipcode`
- `dex_v2_category_loss_viz_latest_zipcode`
- `dex_v2_category_reason_latest_zipcode`
- `dex_v2_category_reason_long_zipcode`
- `dex_v2_customer_zip_risk_rollup`
- `dex_v2_weekly_reject_log`

## Local commands

```powershell
python scripts\run_weekly_scoring.py
python scripts\validate_weekly_scoring.py
pytest -q
```

A successful validator must show `"status": "PASS"`.
