# Phase 7 — Local XGBoost Model Training

Phase 7 trains the H14 category-loss model entirely offline from the Phase 6 labeled panel.

## Model contract

- Target: `y_cat_activeacct_14`
- Raw feature count: **44**
- Categorical features: `customergroupname_current`, `mgrl1`
- Remaining 42 inputs are numeric
- Algorithm: XGBoost classifier
- Objective: `binary:logistic`
- Evaluation metric: `aucpr`
- Tree method: `hist`
- Random seed: 42
- Maximum H14 estimators: 1000
- Early stopping: 50 rounds

The local deterministic parameter set uses midpoint values from the production H14 search space.
This keeps laptop runs reproducible while preserving the production model family and training behavior.

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

The encoder is fitted **only on the training partition**:

- numeric features: median imputation
- categorical features: most-frequent imputation + OneHotEncoder
- unknown categories: ignored safely at validation/test/scoring time

The fitted encoder is persisted with the model.

## Metrics

Each partition records:

- ROC AUC
- Average Precision (AP)
- Lift@10%
- positive rate
- row count

Overfit gaps are also recorded.

Current governance gates mirrored locally:

- validation AUC >= 0.70
- validation AP >= 0.25
- AUC train-valid gap <= 0.20

The training output also evaluates the untouched test split against the same minimum AUC/AP thresholds.

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
- `encoder.joblib`
- `metrics.json`
- `metadata.json`
- `test_predictions.parquet`

These generated artifacts are intentionally excluded from Git.

## Local commands

After pulling the Phase 7 branch:

```powershell
pip install -r requirements.txt
python scripts\run_model_training.py
python scripts\validate_model_training.py
pytest -q
```

Optional MLflow UI from the repository root:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open the local URL printed by MLflow.
