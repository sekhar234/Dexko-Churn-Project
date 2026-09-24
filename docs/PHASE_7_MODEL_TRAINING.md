# Phase 7 — Local XGBoost Model Training

Phase 7 trains the H14 category-loss model entirely offline from the Phase 6 labeled panel.

## Model contract

- Target: `y_cat_activeacct_14`
- Raw feature count: **44**
- Categorical features: `customergroupname_current`, `mgrl1`
- Remaining 42 inputs are numeric
- Training API: native `xgboost.train`
- Objective: `binary:logistic`
- Evaluation metric: `aucpr`
- Tree method: `hist`
- Random seed: 42
- Maximum H14 boosting rounds: 1000
- Early stopping: 50 rounds

The local deterministic parameter set uses midpoint values from the production H14 search space.

## Why scikit-learn is not required

The offline implementation intentionally avoids scikit-learn compiled extensions because some
Windows App Control policies block native scikit-learn `.pyd` files inside user-managed virtual
environments.

Preprocessing is implemented with pandas/NumPy and persisted as JSON. The model is trained through
native XGBoost, so the model contract is preserved without depending on scikit-learn.

## Population

Training reads:

`data/labels/dex_v2_category_labeled_panel_zipcode`

and keeps only rows where:

- `train_eligible_14 == 1`
- `y_cat_activeacct_14` is resolved
- `snapshot_dt >= 2024-01-01`

## Temporal split

The final labeled period is split chronologically:

1. Train: all eligible history before the final 180 days
2. Validation: preceding 90 calendar days
3. Test: most recent 90 calendar days

The test partition is not supplied to XGBoost during fitting or early stopping.

## Encoding

The encoder is fitted only on the training partition:

- numeric features: training median imputation
- categorical features: training mode for missing values
- categorical encoding: deterministic one-hot columns learned from training only
- unseen validation/test/scoring categories: all-zero category block

The encoder contract is persisted to `encoder.json`.

## Metrics

Each partition records:

- ROC AUC
- Average Precision (AP)
- Lift@10%
- positive rate
- row count

Current governance gates mirrored locally:

- validation AUC >= 0.70
- validation AP >= 0.25
- AUC train-valid gap <= 0.20

The untouched test split is evaluated separately.

## MLflow

Tracking backend:

`mlflow.db`

Experiment:

`dexko-churn-local`

Run name:

`category_churn_h14_phase7`

Model artifacts are stored under:

`artifacts/models/category_churn_h14/`

including:

- `model.json`
- `encoder.json`
- `metrics.json`
- `metadata.json`
- `test_predictions.parquet`

Generated artifacts are intentionally excluded from Git.

## Local commands

```powershell
python -m pip install -r requirements.txt
python scripts\run_model_training.py
python scripts\validate_model_training.py
pytest -q
```

Optional MLflow UI:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
