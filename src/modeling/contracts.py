"""Phase 7 model-training contracts for the offline replica."""

RANDOM_STATE = 42
VALID_DAYS = 90
TEST_DAYS = 90
TRAINING_FLOOR = "2024-01-01"

TARGET_COL = "y_cat_activeacct_14"
TRAIN_ELIGIBLE_COL = "train_eligible_14"

CATEGORICAL_FEATURES = ["customergroupname_current", "mgrl1"]

XGB_TRAIN_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "seed": RANDOM_STATE,
    "tree_method": "hist",
    "nthread": -1,
    # Production H14 search-space midpoint defaults for deterministic local training.
    "max_depth": 4,
    "min_child_weight": 6.0,
    "lambda": 2.0,
    "alpha": 1.0,
    "colsample_bytree": 0.65,
    "subsample": 0.75,
    "gamma": 0.10,
    "eta": 0.08,
}

NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50

PROMOTION_GATES = {
    "min_auc": 0.70,
    "min_ap": 0.25,
    "max_overfit_gap": 0.20,
}
