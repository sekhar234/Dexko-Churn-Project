from __future__ import annotations

from pathlib import Path
import json
import math
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.features.contracts import CATEGORY_FEATURE_COLS, CATEGORY_KEYS
from src.local_pipeline.io import read_df, resolve_table, update_manifest
from .contracts import (
    CATEGORICAL_FEATURES,
    PROMOTION_GATES,
    RANDOM_STATE,
    TARGET_COL,
    TEST_DAYS,
    TRAIN_ELIGIBLE_COL,
    TRAINING_FLOOR,
    VALID_DAYS,
    XGB_PARAMS,
)


def lift_at_fraction(y_true, y_prob, fraction: float = 0.10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    if len(y) == 0:
        return float("nan")
    base = float(np.mean(y))
    if base <= 0:
        return float("nan")
    n_top = max(1, int(math.ceil(len(y) * fraction)))
    order = np.argsort(-p)
    top_rate = float(np.mean(y[order[:n_top]]))
    return top_rate / base


def binary_metrics(y_true, y_prob) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    if len(np.unique(y)) < 2:
        auc = float("nan")
        ap = float("nan")
    else:
        auc = float(roc_auc_score(y, p))
        ap = float(average_precision_score(y, p))
    return {
        "auc": auc,
        "ap": ap,
        "lift_at_10pct": float(lift_at_fraction(y, p, 0.10)),
        "positive_rate": float(np.mean(y)) if len(y) else float("nan"),
        "rows": int(len(y)),
    }


def temporal_train_valid_test_split(
    df: pd.DataFrame,
    valid_days: int = VALID_DAYS,
    test_days: int = TEST_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    x = df.copy()
    x["snapshot_dt"] = pd.to_datetime(x["snapshot_dt"])
    max_dt = x["snapshot_dt"].max().normalize()

    test_start = max_dt - pd.Timedelta(days=test_days - 1)
    valid_end = test_start - pd.Timedelta(days=1)
    valid_start = valid_end - pd.Timedelta(days=valid_days - 1)

    train = x[x["snapshot_dt"] < valid_start].copy()
    valid = x[(x["snapshot_dt"] >= valid_start) & (x["snapshot_dt"] < test_start)].copy()
    test = x[x["snapshot_dt"] >= test_start].copy()

    for name, part in [("train", train), ("valid", valid), ("test", test)]:
        if part.empty:
            raise ValueError(f"Temporal split produced empty {name} partition")
        if part[TARGET_COL].nunique(dropna=True) < 2:
            raise ValueError(f"Temporal split {name} has only one target class")

    boundaries = {
        "train_min": str(train["snapshot_dt"].min().date()),
        "train_max": str(train["snapshot_dt"].max().date()),
        "valid_min": str(valid["snapshot_dt"].min().date()),
        "valid_max": str(valid["snapshot_dt"].max().date()),
        "test_min": str(test["snapshot_dt"].min().date()),
        "test_max": str(test["snapshot_dt"].max().date()),
    }
    return train, valid, test, boundaries


def _make_preprocessor() -> tuple[ColumnTransformer, list[str], list[str]]:
    categorical = list(CATEGORICAL_FEATURES)
    numeric = [c for c in CATEGORY_FEATURE_COLS if c not in categorical]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )
    return preprocessor, numeric, categorical


def _promotion_checks(valid_metrics: dict, test_metrics: dict, train_metrics: dict) -> dict:
    auc_gap = (
        train_metrics["auc"] - valid_metrics["auc"]
        if np.isfinite(train_metrics["auc"]) and np.isfinite(valid_metrics["auc"])
        else float("nan")
    )
    ap_gap = (
        train_metrics["ap"] - valid_metrics["ap"]
        if np.isfinite(train_metrics["ap"]) and np.isfinite(valid_metrics["ap"])
        else float("nan")
    )

    checks = {
        "valid_auc_gate": bool(valid_metrics["auc"] >= PROMOTION_GATES["min_auc"]),
        "valid_ap_gate": bool(valid_metrics["ap"] >= PROMOTION_GATES["min_ap"]),
        "test_auc_gate": bool(test_metrics["auc"] >= PROMOTION_GATES["min_auc"]),
        "test_ap_gate": bool(test_metrics["ap"] >= PROMOTION_GATES["min_ap"]),
        "auc_overfit_gate": bool(auc_gap <= PROMOTION_GATES["max_overfit_gap"]),
        "metrics_finite": bool(
            all(
                np.isfinite(v)
                for m in [train_metrics, valid_metrics, test_metrics]
                for k, v in m.items()
                if k in {"auc", "ap", "lift_at_10pct", "positive_rate"}
            )
        ),
    }
    return {
        "checks": checks,
        "pass": bool(all(checks.values())),
        "auc_overfit_gap": float(auc_gap),
        "ap_overfit_gap": float(ap_gap),
    }


def train_category_model(root: str | Path) -> dict:
    root = Path(root)
    panel = read_df(
        resolve_table(root / "data" / "labels" / "dex_v2_category_labeled_panel_zipcode")
    )

    missing = [c for c in CATEGORY_FEATURE_COLS + [TARGET_COL, TRAIN_ELIGIBLE_COL, "snapshot_dt"] if c not in panel.columns]
    if missing:
        raise KeyError(f"Missing training columns: {missing}")

    panel["snapshot_dt"] = pd.to_datetime(panel["snapshot_dt"])
    training = panel[
        (panel[TRAIN_ELIGIBLE_COL] == 1)
        & panel[TARGET_COL].notna()
        & (panel["snapshot_dt"] >= pd.Timestamp(TRAINING_FLOOR))
    ].copy()
    training[TARGET_COL] = training[TARGET_COL].astype(int)

    if len(CATEGORY_FEATURE_COLS) != 44:
        raise AssertionError(f"Expected 44 raw model features, got {len(CATEGORY_FEATURE_COLS)}")
    if training.empty:
        raise ValueError("No training-eligible labeled rows found")

    train, valid, test, boundaries = temporal_train_valid_test_split(training)

    preprocessor, numeric_cols, categorical_cols = _make_preprocessor()
    X_train = preprocessor.fit_transform(train[CATEGORY_FEATURE_COLS])
    X_valid = preprocessor.transform(valid[CATEGORY_FEATURE_COLS])
    X_test = preprocessor.transform(test[CATEGORY_FEATURE_COLS])

    y_train = train[TARGET_COL].to_numpy()
    y_valid = valid[TARGET_COL].to_numpy()
    y_test = test[TARGET_COL].to_numpy()

    model = XGBClassifier(**XGB_PARAMS)
    started = time.time()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )
    train_seconds = float(time.time() - started)

    pred_train = model.predict_proba(X_train)[:, 1]
    pred_valid = model.predict_proba(X_valid)[:, 1]
    pred_test = model.predict_proba(X_test)[:, 1]

    metrics = {
        "train": binary_metrics(y_train, pred_train),
        "valid": binary_metrics(y_valid, pred_valid),
        "test": binary_metrics(y_test, pred_test),
    }
    promotion = _promotion_checks(metrics["valid"], metrics["test"], metrics["train"])

    feature_names = [str(x) for x in preprocessor.get_feature_names_out()]
    encoded_feature_count = len(feature_names)

    artifact_dir = root / "artifacts" / "models" / "category_churn_h14"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    encoder_path = artifact_dir / "encoder.joblib"
    model_path = artifact_dir / "model.json"
    metrics_path = artifact_dir / "metrics.json"
    metadata_path = artifact_dir / "metadata.json"
    test_predictions_path = artifact_dir / "test_predictions.parquet"

    joblib.dump(preprocessor, encoder_path)
    model.save_model(model_path)

    pred_cols = [c for c in CATEGORY_KEYS if c in test.columns]
    pred_df = test[pred_cols + [TARGET_COL]].copy()
    pred_df["p_14"] = pred_test
    pred_df.to_parquet(test_predictions_path, index=False)

    best_iteration = getattr(model, "best_iteration", None)
    metadata = {
        "model_type": "XGBClassifier",
        "target": TARGET_COL,
        "random_state": RANDOM_STATE,
        "raw_feature_count": len(CATEGORY_FEATURE_COLS),
        "encoded_feature_count": encoded_feature_count,
        "numeric_feature_count": len(numeric_cols),
        "categorical_feature_count": len(categorical_cols),
        "categorical_features": categorical_cols,
        "feature_names_encoded": feature_names,
        "split_boundaries": boundaries,
        "best_iteration": None if best_iteration is None else int(best_iteration),
        "training_seconds": train_seconds,
        "xgb_params": XGB_PARAMS,
    }

    metrics_payload = {
        "metrics": metrics,
        "promotion": promotion,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    import mlflow

    db_path = (root / "mlflow.db").as_posix()
    tracking_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment("dexko-churn-local")

    with mlflow.start_run(run_name="category_churn_h14_phase7") as run:
        mlflow.log_params({
            "target": TARGET_COL,
            "raw_feature_count": len(CATEGORY_FEATURE_COLS),
            "encoded_feature_count": encoded_feature_count,
            "valid_days": VALID_DAYS,
            "test_days": TEST_DAYS,
            "training_floor": TRAINING_FLOOR,
            **{f"xgb_{k}": v for k, v in XGB_PARAMS.items() if k != "n_jobs"},
        })
        for split_name, split_metrics in metrics.items():
            for metric_name, value in split_metrics.items():
                if metric_name != "rows" and np.isfinite(value):
                    mlflow.log_metric(f"{split_name}_{metric_name}", float(value))
            mlflow.log_metric(f"{split_name}_rows", int(split_metrics["rows"]))
        mlflow.log_metric("auc_overfit_gap", promotion["auc_overfit_gap"])
        mlflow.log_metric("ap_overfit_gap", promotion["ap_overfit_gap"])
        mlflow.log_metric("promotion_pass", int(promotion["pass"]))
        if best_iteration is not None:
            mlflow.log_metric("best_iteration", int(best_iteration))
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(encoder_path))
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(metadata_path))
        mlflow.log_artifact(str(test_predictions_path))
        run_id = run.info.run_id

    result = {
        "status": "PASS",
        "run_id": run_id,
        "tracking_uri": tracking_uri,
        "rows": {
            "eligible_labeled": int(len(training)),
            "train": int(len(train)),
            "valid": int(len(valid)),
            "test": int(len(test)),
        },
        "features": {
            "raw": len(CATEGORY_FEATURE_COLS),
            "encoded": encoded_feature_count,
            "numeric": len(numeric_cols),
            "categorical": len(categorical_cols),
        },
        "split_boundaries": boundaries,
        "metrics": metrics,
        "promotion": promotion,
        "best_iteration": None if best_iteration is None else int(best_iteration),
        "paths": {
            "model": str(model_path),
            "encoder": str(encoder_path),
            "metrics": str(metrics_path),
            "metadata": str(metadata_path),
            "test_predictions": str(test_predictions_path),
        },
    }
    update_manifest(root, "model_training", result)
    return result
