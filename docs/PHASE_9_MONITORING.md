# Phase 9 — Drift, Performance and Retraining Monitoring

Phase 9 adds weekly model-health monitoring to the offline DexKo churn replica.

## Drift monitoring

The local monitor compares the latest scored population against a frozen training/reference baseline using Population Stability Index (PSI).

PSI interpretation:

- `PSI < 0.10` → STABLE
- `0.10 <= PSI < 0.25` → WATCH
- `PSI >= 0.25` → ACT

PSI uses quantile bins derived from the baseline for numeric variables and baseline category distributions for categorical variables. Missing values are monitored explicitly.

## Monitored inputs

The verified production documentation confirms 14 numeric and 2 categorical monitored inputs and identifies these critical features:

- `recency_days`
- `median_cycle_days`
- `spend_0_30_g`

The surfaced documentation does not enumerate every member of the 14-name numeric list. The offline replica therefore freezes a 14-feature monitoring contract drawn from the approved 44 model features while preserving all confirmed critical fields.

Categorical drift is monitored for:

- `customergroupname_current`
- `mgrl1`

Prediction-score drift is monitored separately as `__SCORE_P14__`.

## Baseline

The local baseline is created once and then frozen.

Feature distributions use Phase 7's training reference period: train-eligible, resolved labels prior to the final validation period.

The score baseline uses the untouched Phase 7 test predictions.

This provides a deterministic local equivalent of the approved model reference distribution.

## Retraining recommendation

Automatic retraining is disabled.

A retraining recommendation is raised only when a critical feature has ACT-level PSI for two distinct consecutive monitored scoring snapshots.

A rerun of the same snapshot is idempotent and does not count as a second confirmation run.

## Performance monitoring

Weekly probabilities cannot be evaluated immediately.

The target requires:

- 14-day prediction horizon
- 180-day confirmation window
- total label maturity = 194 days

Phase 9 stores every scoring snapshot in a persistent local scoring history.

For each monitoring run:

1. Find scored snapshots at least 194 days old.
2. Join them to the now-resolved Phase 6 labels.
3. Require at least 100 scored + labeled rows.
4. Compute:
   - ROC AUC
   - Average Precision
   - Precision @ 0.40
   - Recall @ 0.40
   - Precision @ 0.20
   - Recall @ 0.20
   - Lift @ 10%
   - positive rate
5. Persist the result once per scoring snapshot.

If no labels are mature yet, the monitor returns:

`BLOCKED_WAITING_FOR_LABELS`

This is a valid non-failing state.

## Outputs

Generated under `data/monitoring/`:

- `dex_v2_drift_baseline`
- `dex_v2_drift_metrics`
- `dex_v2_drift_summary`
- `dex_v2_monitoring_feature_distributions`
- `dex_v2_monitoring_score_distributions`
- `dex_v2_retraining_decisions`
- `dex_v2_scoring_history`
- `dex_v2_performance_history` once mature labels exist

## Local commands

```powershell
python scripts\run_monitoring.py
python scripts\validate_monitoring.py
pytest -q
```

A healthy first monitoring run may show:

- overall monitoring status: PASS
- drift: STABLE, WATCH, or ACT depending on synthetic-data shift
- retraining: not recommended until confirmation criteria are met
- performance: BLOCKED_WAITING_FOR_LABELS
