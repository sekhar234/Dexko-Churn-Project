from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

from src.features.contracts import CATEGORY_KEYS
from src.local_pipeline.io import read_df, resolve_table, write_df, update_manifest
from src.modeling.pipeline import average_precision, lift_at_fraction, roc_auc
from .contracts import (
    CRITICAL_DRIFT_FEATURES,
    DRIFT_CONFIRMATION_RUNS,
    LABEL_MATURITY_DAYS,
    MIN_PERFORMANCE_ROWS,
    MONITORED_CATEGORICAL_FEATURES,
    MONITORED_NUMERIC_FEATURES,
    PSI_EPSILON,
    PSI_NUMERIC_BINS,
    PSI_STABLE_MAX,
    PSI_WATCH_MAX,
)


def _status(psi: float) -> str:
    if psi < PSI_STABLE_MAX:
        return "STABLE"
    if psi < PSI_WATCH_MAX:
        return "WATCH"
    return "ACT"


def _psi(base_props: np.ndarray, current_props: np.ndarray) -> float:
    b = np.clip(np.asarray(base_props, dtype=float), PSI_EPSILON, None)
    c = np.clip(np.asarray(current_props, dtype=float), PSI_EPSILON, None)
    return float(np.sum((c - b) * np.log(c / b)))


def _numeric_edges(series: pd.Series) -> list[float]:
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return [-float("inf"), float("inf")]
    q = np.linspace(0.0, 1.0, PSI_NUMERIC_BINS + 1)
    vals = np.unique(np.quantile(x, q))
    internal = vals[1:-1].tolist() if len(vals) > 2 else []
    return [-float("inf"), *[float(v) for v in internal], float("inf")]


def _numeric_distribution(
    series: pd.Series,
    edges: list[float],
    feature_name: str,
    source: str,
) -> pd.DataFrame:
    x = pd.to_numeric(series, errors="coerce")
    nonnull = x.dropna()
    bins = pd.cut(
        nonnull,
        bins=edges,
        right=False,
        include_lowest=True,
        duplicates="drop",
    )
    counts = bins.value_counts(sort=False)
    total = max(1, len(x))

    rows = []
    for i, (interval, count) in enumerate(counts.items()):
        rows.append({
            "feature_name": feature_name,
            "feature_type": "numeric",
            "bin_index": i,
            "bin_label": str(interval),
            "lower_bound": float(interval.left),
            "upper_bound": float(interval.right),
            "category_value": None,
            "count": int(count),
            "proportion": float(count / total),
            "source": source,
        })

    missing = int(x.isna().sum())
    rows.append({
        "feature_name": feature_name,
        "feature_type": "numeric",
        "bin_index": len(rows),
        "bin_label": "__MISSING__",
        "lower_bound": None,
        "upper_bound": None,
        "category_value": "__MISSING__",
        "count": missing,
        "proportion": float(missing / total),
        "source": source,
    })
    return pd.DataFrame(rows)


def _categorical_distribution(
    series: pd.Series,
    categories: list[str],
    feature_name: str,
    source: str,
) -> pd.DataFrame:
    raw = series.astype("string")
    values = raw.fillna("__MISSING__").astype(str)
    known = set(categories)
    normalized = values.where(values.isin(known), "__OTHER__")
    labels = [*categories, "__OTHER__", "__MISSING__"]
    labels = list(dict.fromkeys(labels))
    total = max(1, len(values))
    counts = normalized.value_counts()

    rows = []
    for i, label in enumerate(labels):
        count = int(counts.get(label, 0))
        rows.append({
            "feature_name": feature_name,
            "feature_type": "categorical",
            "bin_index": i,
            "bin_label": label,
            "lower_bound": None,
            "upper_bound": None,
            "category_value": label,
            "count": count,
            "proportion": float(count / total),
            "source": source,
        })
    return pd.DataFrame(rows)


def _make_baseline(
    reference: pd.DataFrame,
    score_reference: pd.Series,
    baseline_id: str,
) -> pd.DataFrame:
    frames = []

    for feature in MONITORED_NUMERIC_FEATURES:
        edges = _numeric_edges(reference[feature])
        d = _numeric_distribution(
            reference[feature],
            edges,
            feature,
            "training_reference",
        )
        d["edges_json"] = json.dumps(edges)
        frames.append(d)

    for feature in MONITORED_CATEGORICAL_FEATURES:
        vals = (
            reference[feature]
            .astype("string")
            .dropna()
            .astype(str)
            .value_counts()
            .index
            .tolist()
        )
        d = _categorical_distribution(
            reference[feature],
            vals,
            feature,
            "training_reference",
        )
        d["edges_json"] = json.dumps(vals)
        frames.append(d)

    score_edges = _numeric_edges(score_reference)
    score_dist = _numeric_distribution(
        score_reference,
        score_edges,
        "__SCORE_P14__",
        "phase7_test_reference",
    )
    score_dist["edges_json"] = json.dumps(score_edges)
    frames.append(score_dist)

    out = pd.concat(frames, ignore_index=True)
    out["baseline_id"] = baseline_id
    out["created_at"] = datetime.now().isoformat(timespec="seconds")
    return out


def _current_distributions(
    scores: pd.DataFrame,
    baseline: pd.DataFrame,
    snapshot_dt: pd.Timestamp,
    monitoring_run_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_frames = []
    score_frames = []
    metric_rows = []

    for feature in [
        *MONITORED_NUMERIC_FEATURES,
        *MONITORED_CATEGORICAL_FEATURES,
        "__SCORE_P14__",
    ]:
        b = baseline[baseline["feature_name"] == feature].sort_values("bin_index")
        if b.empty:
            raise KeyError(f"Missing baseline distribution for {feature}")

        ftype = str(b["feature_type"].iloc[0])
        if feature == "__SCORE_P14__":
            series = scores["p_14"]
        else:
            series = scores[feature]

        if ftype == "numeric":
            edges = json.loads(str(b["edges_json"].iloc[0]))
            c = _numeric_distribution(series, edges, feature, "current_scoring")
        else:
            categories = json.loads(str(b["edges_json"].iloc[0]))
            c = _categorical_distribution(
                series,
                categories,
                feature,
                "current_scoring",
            )

        c["snapshot_dt"] = snapshot_dt
        c["monitoring_run_id"] = monitoring_run_id
        c["baseline_id"] = str(b["baseline_id"].iloc[0])

        expected_bins = b["bin_label"].astype(str).tolist()
        c = c.set_index("bin_label").reindex(expected_bins, fill_value=0).reset_index()
        for col in [
            "feature_name",
            "feature_type",
            "source",
            "snapshot_dt",
            "monitoring_run_id",
            "baseline_id",
        ]:
            if col in c.columns:
                c[col] = c[col].replace(0, np.nan).ffill().bfill()

        base_props = b["proportion"].to_numpy(dtype=float)
        current_props = c["proportion"].to_numpy(dtype=float)
        value = _psi(base_props, current_props)
        metric_rows.append({
            "monitoring_run_id": monitoring_run_id,
            "scoring_snapshot": snapshot_dt,
            "baseline_id": str(b["baseline_id"].iloc[0]),
            "metric_type": "PSI",
            "feature_name": feature,
            "feature_type": ftype,
            "metric_value": value,
            "threshold_watch": PSI_STABLE_MAX,
            "threshold_action": PSI_WATCH_MAX,
            "status": _status(value),
            "is_critical": feature in CRITICAL_DRIFT_FEATURES,
            "created_timestamp": datetime.now().isoformat(timespec="seconds"),
        })

        if feature == "__SCORE_P14__":
            score_frames.append(c)
        else:
            feature_frames.append(c)

    return (
        pd.concat(feature_frames, ignore_index=True),
        pd.concat(score_frames, ignore_index=True),
        pd.DataFrame(metric_rows),
    )


def _upsert_history(
    base: Path,
    new_rows: pd.DataFrame,
    key_cols: list[str],
    preferred_format: str,
) -> Path:
    try:
        old = read_df(resolve_table(base))
        combined = pd.concat([old, new_rows], ignore_index=True)
    except FileNotFoundError:
        combined = new_rows.copy()

    combined = combined.drop_duplicates(key_cols, keep="last")
    return write_df(combined, base, preferred_format)


def _classification_metrics(y_true, prob) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)

    def threshold_metrics(t: float) -> tuple[float, float]:
        pred = p >= t
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        return float(precision), float(recall)

    precision_40, recall_40 = threshold_metrics(0.40)
    precision_20, recall_20 = threshold_metrics(0.20)

    return {
        "auc_roc": float(roc_auc(y, p)),
        "avg_precision": float(average_precision(y, p)),
        "precision_at_40": precision_40,
        "recall_at_40": recall_40,
        "precision_at_20": precision_20,
        "recall_at_20": recall_20,
        "lift_at_10pct": float(lift_at_fraction(y, p, 0.10)),
        "positive_rate": float(y.mean()) if len(y) else float("nan"),
        "rows": int(len(y)),
    }


def _performance_monitoring(
    scoring_history: pd.DataFrame,
    panel: pd.DataFrame,
    existing_history: pd.DataFrame | None,
) -> tuple[dict, pd.DataFrame | None]:
    today = pd.Timestamp(date.today())
    scoring_history = scoring_history.copy()
    scoring_history["snapshot_dt"] = pd.to_datetime(scoring_history["snapshot_dt"])

    mature_cutoff = today - pd.Timedelta(days=LABEL_MATURITY_DAYS)
    candidates = sorted(
        scoring_history.loc[
            scoring_history["snapshot_dt"] <= mature_cutoff,
            "snapshot_dt",
        ].drop_duplicates()
    )

    evaluated = set()
    if existing_history is not None and not existing_history.empty:
        evaluated = set(pd.to_datetime(existing_history["scoring_snapshot_dt"]))

    pending = [x for x in candidates if x not in evaluated]
    if not pending:
        next_maturity = (
            scoring_history["snapshot_dt"].min()
            + pd.Timedelta(days=LABEL_MATURITY_DAYS)
        )
        return {
            "status": "BLOCKED_WAITING_FOR_LABELS",
            "label_maturity_days": LABEL_MATURITY_DAYS,
            "mature_cutoff": str(mature_cutoff.date()),
            "next_maturity_date": str(next_maturity.date()),
        }, None

    snapshot = pd.Timestamp(pending[-1])
    s = scoring_history[scoring_history["snapshot_dt"] == snapshot].copy()
    labels = panel[pd.to_datetime(panel["snapshot_dt"]) == snapshot][
        CATEGORY_KEYS + ["y_cat_activeacct_14"]
    ].copy()
    joined = s.merge(labels, on=CATEGORY_KEYS, how="inner", validate="one_to_one")
    joined = joined[joined["y_cat_activeacct_14"].notna()].copy()

    if len(joined) < MIN_PERFORMANCE_ROWS:
        return {
            "status": "BLOCKED_INSUFFICIENT_MATURE_ROWS",
            "scoring_snapshot_dt": str(snapshot.date()),
            "rows": int(len(joined)),
            "minimum_rows": MIN_PERFORMANCE_ROWS,
        }, None

    metrics = _classification_metrics(
        joined["y_cat_activeacct_14"].astype(int),
        joined["p_14"],
    )
    row = pd.DataFrame([{
        "scoring_snapshot_dt": snapshot,
        "evaluation_dt": today,
        **metrics,
        "created_timestamp": datetime.now().isoformat(timespec="seconds"),
    }])
    return {
        "status": "EVALUATED",
        "scoring_snapshot_dt": str(snapshot.date()),
        **metrics,
    }, row


def run_monitoring(
    root: str | Path,
    preferred_format: str = "parquet",
) -> dict:
    root = Path(root)
    monitoring_dir = root / "data" / "monitoring"
    monitoring_dir.mkdir(parents=True, exist_ok=True)

    scores = read_df(
        resolve_table(
            root / "data" / "scoring" / "dex_v2_category_loss_scores_wide_zipcode"
        )
    )
    panel = read_df(
        resolve_table(
            root / "data" / "labels" / "dex_v2_category_labeled_panel_zipcode"
        )
    )
    scores["snapshot_dt"] = pd.to_datetime(scores["snapshot_dt"])
    panel["snapshot_dt"] = pd.to_datetime(panel["snapshot_dt"])
    snapshot_dt = scores["snapshot_dt"].max()
    scores = scores[scores["snapshot_dt"] == snapshot_dt].copy()

    metadata_path = root / "artifacts" / "models" / "category_churn_h14" / "metadata.json"
    test_predictions_path = (
        root / "artifacts" / "models" / "category_churn_h14" / "test_predictions.parquet"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    valid_min = pd.Timestamp(metadata["split_boundaries"]["valid_min"])
    reference = panel[
        (panel["train_eligible_14"] == 1)
        & panel["y_cat_activeacct_14"].notna()
        & (panel["snapshot_dt"] < valid_min)
    ].copy()
    if reference.empty:
        raise ValueError("No frozen training reference rows available for drift baseline")

    score_reference = read_df(test_predictions_path)["p_14"]
    baseline_id = (
        "H14_TRAIN_REFERENCE_"
        + metadata["split_boundaries"]["train_max"].replace("-", "")
    )
    baseline_base = monitoring_dir / "dex_v2_drift_baseline"

    try:
        baseline = read_df(resolve_table(baseline_base))
    except FileNotFoundError:
        baseline = _make_baseline(reference, score_reference, baseline_id)
        write_df(baseline, baseline_base, preferred_format)

    monitoring_run_id = (
        "MON_"
        + snapshot_dt.strftime("%Y%m%d")
        + "_"
        + datetime.now().strftime("%H%M%S")
        + "_LOCAL"
    )

    feature_dist, score_dist, metrics = _current_distributions(
        scores,
        baseline,
        snapshot_dt,
        monitoring_run_id,
    )

    _upsert_history(
        monitoring_dir / "dex_v2_monitoring_feature_distributions",
        feature_dist,
        ["snapshot_dt", "feature_name", "bin_label"],
        preferred_format,
    )
    _upsert_history(
        monitoring_dir / "dex_v2_monitoring_score_distributions",
        score_dist,
        ["snapshot_dt", "feature_name", "bin_label"],
        preferred_format,
    )
    drift_metrics_path = _upsert_history(
        monitoring_dir / "dex_v2_drift_metrics",
        metrics,
        ["scoring_snapshot", "feature_name", "metric_type"],
        preferred_format,
    )

    overall_status = "STABLE"
    if (metrics["status"] == "ACT").any():
        overall_status = "ACT"
    elif (metrics["status"] == "WATCH").any():
        overall_status = "WATCH"

    critical_act = metrics.loc[
        metrics["is_critical"] & (metrics["status"] == "ACT"),
        "feature_name",
    ].tolist()

    drift_history = read_df(drift_metrics_path)
    drift_history["scoring_snapshot"] = pd.to_datetime(
        drift_history["scoring_snapshot"]
    )

    confirmed = []
    for feature in CRITICAL_DRIFT_FEATURES:
        h = drift_history[
            drift_history["feature_name"].eq(feature)
        ].sort_values("scoring_snapshot")
        h = h.drop_duplicates("scoring_snapshot", keep="last")
        tail = h.tail(DRIFT_CONFIRMATION_RUNS)
        if (
            len(tail) >= DRIFT_CONFIRMATION_RUNS
            and tail["status"].eq("ACT").all()
        ):
            confirmed.append(feature)

    retrain_recommended = bool(confirmed)
    retrain_row = pd.DataFrame([{
        "monitoring_run_id": monitoring_run_id,
        "scoring_snapshot": snapshot_dt,
        "decision": (
            "RETRAIN_RECOMMENDED"
            if retrain_recommended
            else "NO_RETRAIN"
        ),
        "trigger_features": "|".join(confirmed),
        "current_critical_act_features": "|".join(critical_act),
        "confirmation_runs_required": DRIFT_CONFIRMATION_RUNS,
        "automatic_retraining_enabled": False,
        "created_timestamp": datetime.now().isoformat(timespec="seconds"),
    }])
    retraining_path = _upsert_history(
        monitoring_dir / "dex_v2_retraining_decisions",
        retrain_row,
        ["scoring_snapshot"],
        preferred_format,
    )

    summary_row = pd.DataFrame([{
        "monitoring_run_id": monitoring_run_id,
        "scoring_snapshot": snapshot_dt,
        "baseline_id": baseline_id,
        "overall_status": overall_status,
        "max_psi": float(metrics["metric_value"].max()),
        "act_count": int((metrics["status"] == "ACT").sum()),
        "watch_count": int((metrics["status"] == "WATCH").sum()),
        "stable_count": int((metrics["status"] == "STABLE").sum()),
        "critical_act_features": "|".join(critical_act),
        "confirmed_critical_act_features": "|".join(confirmed),
        "retrain_recommended": retrain_recommended,
        "model_changed_flag": False,
        "created_timestamp": datetime.now().isoformat(timespec="seconds"),
    }])
    drift_summary_path = _upsert_history(
        monitoring_dir / "dex_v2_drift_summary",
        summary_row,
        ["scoring_snapshot"],
        preferred_format,
    )

    scoring_history_path = _upsert_history(
        monitoring_dir / "dex_v2_scoring_history",
        scores,
        CATEGORY_KEYS,
        preferred_format,
    )
    scoring_history = read_df(scoring_history_path)

    performance_base = monitoring_dir / "dex_v2_performance_history"
    try:
        performance_history = read_df(resolve_table(performance_base))
    except FileNotFoundError:
        performance_history = None

    performance, performance_row = _performance_monitoring(
        scoring_history,
        panel,
        performance_history,
    )
    performance_path = None
    if performance_row is not None:
        performance_path = str(_upsert_history(
            performance_base,
            performance_row,
            ["scoring_snapshot_dt"],
            preferred_format,
        ))

    result = {
        "status": "PASS",
        "monitoring_run_id": monitoring_run_id,
        "snapshot_dt": str(snapshot_dt.date()),
        "baseline_id": baseline_id,
        "drift": {
            "overall_status": overall_status,
            "max_psi": float(metrics["metric_value"].max()),
            "stable": int((metrics["status"] == "STABLE").sum()),
            "watch": int((metrics["status"] == "WATCH").sum()),
            "act": int((metrics["status"] == "ACT").sum()),
            "critical_act_features": critical_act,
            "confirmed_critical_act_features": confirmed,
        },
        "retraining": {
            "recommended": retrain_recommended,
            "automatic_retraining_enabled": False,
            "confirmation_runs_required": DRIFT_CONFIRMATION_RUNS,
        },
        "performance": performance,
        "paths": {
            "baseline": str(resolve_table(baseline_base)),
            "drift_metrics": str(drift_metrics_path),
            "drift_summary": str(drift_summary_path),
            "retraining_decisions": str(retraining_path),
            "scoring_history": str(scoring_history_path),
            "performance_history": performance_path,
        },
    }
    update_manifest(root, "monitoring", result)
    return result
