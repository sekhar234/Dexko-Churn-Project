"""Phase 7 model-training contracts for the offline replica."""

RANDOM_STATE = 42
VALID_DAYS = 90
TEST_DAYS = 90
TRAINING_FLOOR = "2024-01-01"

TARGET_COL = "y_cat_activeacct_14"
TRAIN_ELIGIBLE_COL = "train_eligible_14"

CATEGORICAL_FEATURES = ["customergroupname_current", "mgrl1"]

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "random_state": RANDOM_STATE,
    "tree_method": "hist",
    "n_jobs": -1,
    "n_estimators": 1000,
    "early_stopping_rounds": 50,
    # Production search-space midpoint defaults for deterministic local training.
    "max_depth": 4,
    "min_child_weight": 6.0,
    "reg_lambda": 2.0,
    "reg_alpha": 1.0,
    "colsample_bytree": 0.65,
    "subsample": 0.75,
    "gamma": 0.10,
    "learning_rate": 0.08,
}

PROMOTION_GATES = {
    "min_auc": 0.70,
    "min_ap": 0.25,
    "max_overfit_gap": 0.20,
}
