from __future__ import annotations

from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from src.features.contracts import CATEGORY_FEATURE_COLS, CATEGORY_KEYS
from src.local_pipeline.io import read_df, resolve_table, update_manifest
from .contracts import (
    CATEGORICAL_FEATURES,
    EARLY_STOPPING_ROUNDS,
    NUM_BOOST_ROUND,
    PROMOTION_GATES,
    RANDOM_STATE,
    TARGET_COL,
    TEST_DAYS,
    TRAIN_ELIGIBLE_COL,
    TRAINING_FLOOR,
    VALID_DAYS,
    XGB_TRAIN_PARAMS,
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
    order = np.argsort(-p, kind="mergesort")
    top_rate = float(np.mean(y[order[:n_top]]))
    return top_rate / base


def roc_auc(y_true, y_prob) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(p).rank(method="average").to_numpy()
    pos_rank_sum = float(ranks[y == 1].sum())
    return (
        pos_rank_sum - (n_pos * (n_pos + 1) / 2.0)
    ) / float(n_pos * n_neg)


def average_precision(y_true, y_prob) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    cum_pos = np.cumsum(ys)
    ranks = np.arange(1, len(ys) + 1)
    precision = cum_pos / ranks
    return float(precision[ys == 1].sum() / n_pos)


def binary_metrics(y_true, y_prob) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    return {
        "auc": float(roc_auc(y, p)),
        "ap": float(average_precision(y, p)),
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
    valid = x[
        (x["snapshot_dt"] >= valid_start)
        & (x["snapshot_dt"] < test_start)
    ].copy()
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


def _fit_encoder(train: pd.DataFrame) -> dict:
    categorical = list(CATEGORICAL_FEATURES)
    numeric = [c for c in CATEGORY_FEATURE_COLS if c not in categorical]

    medians = {}
    for col in numeric:
        s = pd.to_numeric(train[col], errors="coerce")
        value = s.median()
        medians[col] = float(value) if pd.notna(value) else 0.0

    categories = {}
    fill_values = {}
    for col in categorical:
        s = train[col].astype("string")
        mode = s.dropna().mode()
        fill = str(mode.iloc[0]) if len(mode) else "__MISSING__"
        fill_values[col] = fill
        vals = s.fillna(fill).astype(str)
        categories[col] = sorted(vals.unique().tolist())

    feature_names = list(numeric)
    category_feature_map = {}
    for col in categorical:
        names = [f"{col}__{i}" for i in range(len(categories[col]))]
        category_feature_map[col] = names
        feature_names.extend(names)

    return {
        "raw_features": list(CATEGORY_FEATURE_COLS),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "numeric_medians": medians,
        "categorical_fill_values": fill_values,
        "categorical_values": categories,
        "categorical_encoded_names": category_feature_map,
        "encoded_feature_names": feature_names,
    }


def _transform(df: pd.DataFrame, encoder: dict) -> np.ndarray:
    numeric_arrays = []
    for col in encoder["numeric_features"]:
        s = pd.to_numeric(df[col], errors="coerce").fillna(
            encoder["numeric_medians"][col]
        )
        numeric_arrays.append(s.to_numpy(dtype=np.float32, copy=False))

    if numeric_arrays:
        numeric_matrix = np.column_stack(numeric_arrays).astype(np.float32, copy=False)
    else:
        numeric_matrix = np.empty((len(df), 0), dtype=np.float32)

    blocks = [numeric_matrix]
    for col in encoder["categorical_features"]:
        fill = encoder["categorical_fill_values"][col]
        values = df[col].astype("string").fillna(fill).astype(str).to_numpy()
        known = np.asarray(encoder["categorical_values"][col], dtype=object)
        one_hot = (values[:, None] == known[None, :]).astype(np.float32)
        blocks.append(one_hot)

    return np.concatenate(blocks, axis=1)


def _promotion_checks(
    valid_metrics: dict,
    test_metrics: dict,
    train_metrics: dict,
) -> dict:
    auc_gap = (
        train_metrics["auc"] - valid_metrics["auc"]
        if np.isfinite(train_metrics["auc"])
        and np.isfinite(valid_metrics["auc"])
        else float("nan")
    )
    ap_gap = (
        train_metrics["ap"] - valid_metrics["ap"]
        if np.isfinite(train_metrics["ap"])
        and np.isfinite(valid_metrics["ap"])
        else float("nan")
    )

    checks = {
        "valid_auc_gate": bool(
            valid_metrics["auc"] >= PROMOTION_GATES["min_auc"]
        ),
        "valid_ap_gate": bool(
            valid_metrics["ap"] >= PROMOTION_GATES["min_ap"]
        ),
        "test_auc_gate": bool(
            test_metrics["auc"] >= PROMOTION_GATES["min_auc"]
        ),
        "test_ap_gate": bool(
            test_metrics["ap"] >= PROMOTION_GATES["min_ap"]
        ),
        "auc_overfit_gate": bool(
            auc_gap <= PROMOTION_GATES["max_overfit_gap"]
        ),
        "metrics_finite": bool(
            all(
                np.isfinite(v)
                for m in [train_metrics, valid_metrics, test_metrics]
                for k, v in m.items()
                if k
                in {"auc", "ap", "lift_at_10pct", "positive_rate"}
            )
        ),
    }
    return {
        "checks": checks,
        "pass": bool(all(checks.values())),
        "auc_overfit_gap": float(auc_gap),
        "ap_overfit_gap": float(ap_gap),
    }


def _log_mlflow(
    root: Path,
    metrics: dict,
    promotion: dict,
    encoded_feature_count: int,
    best_iteration: int | None,
    artifact_paths: list[Path],
) -> tuple[str | None, str, str | None]:
    try:
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
                "num_boost_round": NUM_BOOST_ROUND,
                "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
                **{f"xgb_{k}": v for k, v in XGB_TRAIN_PARAMS.items()
                   if k != "nthread"},
            })
            for split_name, split_metrics in metrics.items():
                for metric_name, value in split_metrics.items():
                    if metric_name != "rows" and np.isfinite(value):
                        mlflow.log_metric(
                            f"{split_name}_{metric_name}",
                            float(value),
                        )
                mlflow.log_metric(
                    f"{split_name}_rows",
                    int(split_metrics["rows"]),
                )
            mlflow.log_metric(
                "auc_overfit_gap",
                promotion["auc_overfit_gap"],
            )
            mlflow.log_metric(
                "ap_overfit_gap",
                promotion["ap_overfit_gap"],
            )
            mlflow.log_metric(
                "promotion_pass",
                int(promotion["pass"]),
            )
            if best_iteration is not None:
                mlflow.log_metric("best_iteration", int(best_iteration))
            for path in artifact_paths:
                mlflow.log_artifact(str(path))
            return run.info.run_id, tracking_uri, None
    except Exception as exc:
        return None, "NOT_LOGGED", f"{type(exc).__name__}: {exc}"


def train_category_model(root: str | Path) -> dict:
    root = Path(root)
    panel = read_df(
        resolve_table(
            root
            / "data"
            / "labels"
            / "dex_v2_category_labeled_panel_zipcode"
        )
    )

    required = CATEGORY_FEATURE_COLS + [
        TARGET_COL,
        TRAIN_ELIGIBLE_COL,
        "snapshot_dt",
    ]
    missing = [c for c in required if c not in panel.columns]
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
        raise AssertionError(
            f"Expected 44 raw model features, got {len(CATEGORY_FEATURE_COLS)}"
        )
    if training.empty:
        raise ValueError("No training-eligible labeled rows found")

    train, valid, test, boundaries = temporal_train_valid_test_split(training)

    encoder = _fit_encoder(train)
    X_train = _transform(train[CATEGORY_FEATURE_COLS], encoder)
    X_valid = _transform(valid[CATEGORY_FEATURE_COLS], encoder)
    X_test = _transform(test[CATEGORY_FEATURE_COLS], encoder)

    y_train = train[TARGET_COL].to_numpy(dtype=np.float32)
    y_valid = valid[TARGET_COL].to_numpy(dtype=np.float32)
    y_test = test[TARGET_COL].to_numpy(dtype=np.float32)

    feature_names = encoder["encoded_feature_names"]
    dtrain = xgb.DMatrix(
        X_train,
        label=y_train,
        feature_names=feature_names,
    )
    dvalid = xgb.DMatrix(
        X_valid,
        label=y_valid,
        feature_names=feature_names,
    )
    dtest = xgb.DMatrix(
        X_test,
        label=y_test,
        feature_names=feature_names,
    )

    started = time.time()
    booster = xgb.train(
        params=XGB_TRAIN_PARAMS,
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    train_seconds = float(time.time() - started)

    best_iteration = getattr(booster, "best_iteration", None)
    iteration_range = (
        (0, int(best_iteration) + 1)
        if best_iteration is not None
        else (0, 0)
    )

    pred_train = booster.predict(dtrain, iteration_range=iteration_range)
    pred_valid = booster.predict(dvalid, iteration_range=iteration_range)
    pred_test = booster.predict(dtest, iteration_range=iteration_range)

    metrics = {
        "train": binary_metrics(y_train, pred_train),
        "valid": binary_metrics(y_valid, pred_valid),
        "test": binary_metrics(y_test, pred_test),
    }
    promotion = _promotion_checks(
        metrics["valid"],
        metrics["test"],
        metrics["train"],
    )

    encoded_feature_count = len(feature_names)
    artifact_dir = root / "artifacts" / "models" / "category_churn_h14"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    encoder_path = artifact_dir / "encoder.json"
    model_path = artifact_dir / "model.json"
    metrics_path = artifact_dir / "metrics.json"
    metadata_path = artifact_dir / "metadata.json"
    test_predictions_path = artifact_dir / "test_predictions.parquet"

    encoder_path.write_text(
        json.dumps(encoder, indent=2),
        encoding="utf-8",
    )
    booster.save_model(model_path)

    pred_cols = [c for c in CATEGORY_KEYS if c in test.columns]
    pred_df = test[pred_cols + [TARGET_COL]].copy()
    pred_df["p_14"] = pred_test
    pred_df.to_parquet(test_predictions_path, index=False)

    metadata = {
        "model_type": "xgboost.Booster",
        "training_api": "xgboost.train",
        "target": TARGET_COL,
        "random_state": RANDOM_STATE,
        "raw_feature_count": len(CATEGORY_FEATURE_COLS),
        "encoded_feature_count": encoded_feature_count,
        "numeric_feature_count": len(encoder["numeric_features"]),
        "categorical_feature_count": len(
            encoder["categorical_features"]
        ),
        "categorical_features": encoder["categorical_features"],
        "feature_names_encoded": feature_names,
        "split_boundaries": boundaries,
        "best_iteration": (
            None if best_iteration is None else int(best_iteration)
        ),
        "training_seconds": train_seconds,
        "xgb_params": XGB_TRAIN_PARAMS,
        "num_boost_round": NUM_BOOST_ROUND,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
    }

    metrics_payload = {
        "metrics": metrics,
        "promotion": promotion,
    }
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    artifact_paths = [
        model_path,
        encoder_path,
        metrics_path,
        metadata_path,
        test_predictions_path,
    ]
    run_id, tracking_uri, mlflow_error = _log_mlflow(
        root,
        metrics,
        promotion,
        encoded_feature_count,
        None if best_iteration is None else int(best_iteration),
        artifact_paths,
    )

    result = {
        "status": "PASS" if mlflow_error is None else "MODEL_PASS_MLFLOW_PENDING",
        "run_id": run_id,
        "tracking_uri": tracking_uri,
        "mlflow_error": mlflow_error,
        "rows": {
            "eligible_labeled": int(len(training)),
            "train": int(len(train)),
            "valid": int(len(valid)),
            "test": int(len(test)),
        },
        "features": {
            "raw": len(CATEGORY_FEATURE_COLS),
            "encoded": encoded_feature_count,
            "numeric": len(encoder["numeric_features"]),
            "categorical": len(encoder["categorical_features"]),
        },
        "split_boundaries": boundaries,
        "metrics": metrics,
        "promotion": promotion,
        "best_iteration": (
            None if best_iteration is None else int(best_iteration)
        ),
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
