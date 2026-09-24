from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from src.features.contracts import CATEGORY_FEATURE_COLS, CATEGORY_KEYS, CUSTOMER_COLUMN
from src.local_pipeline.io import read_df, resolve_table, write_df, update_manifest
from src.modeling.pipeline import _transform
from .contracts import (
    ACTION_GRID,
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
    NON_ACTIONABLE_REASON_FEATURES,
    TOP_REASON_COUNT,
)


def _risk_tier(p: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [p >= HIGH_RISK_THRESHOLD, p >= MEDIUM_RISK_THRESHOLD],
            ["High", "Medium"],
            default="Low",
        ),
        index=p.index,
        dtype="string",
    )


def _value_bands(account_spend: pd.Series) -> pd.Series:
    """Mirror current production tercile-style business-value segmentation."""
    if len(account_spend) == 0:
        return pd.Series(dtype="string")
    ranked = account_spend.rank(method="first", ascending=True)
    bands = pd.qcut(ranked, q=3, labels=["Low", "Mid", "High"])
    return bands.astype("string")


def _action(value_band: pd.Series, risk_tier: pd.Series) -> pd.Series:
    return pd.Series(
        [ACTION_GRID[(str(v), str(r))] for v, r in zip(value_band, risk_tier)],
        index=value_band.index,
        dtype="string",
    )


def _feature_value_text(feature: str, value) -> str:
    if pd.isna(value):
        return "missing"

    v = float(value) if isinstance(value, (int, float, np.number)) else value

    if feature in {"recency_days", "recency_days_all"}:
        return f"{int(round(float(v)))} days since last purchase"
    if feature in {"ratio_recency_to_cycle", "ratio_recency_to_cycle_all"}:
        return f"{float(v):.1f}x typical buying cycle"
    if feature in {"active_months_12_g", "active_months_12_all"}:
        return f"active in {int(round(float(v)))} of last 12 months"
    if feature in {"spend_share_365", "spend_share_3m", "mix_shift_share_3m"}:
        return f"{float(v) * 100:.1f} percentage points/share"
    if feature.startswith("spend_logratio_"):
        pct = (np.exp(float(v)) - 1.0) * 100.0
        direction = "up" if pct >= 0 else "down"
        return f"spend {direction} {abs(pct):.0f}% versus comparison period"
    if feature.startswith("freq_logratio_"):
        pct = (np.exp(float(v)) - 1.0) * 100.0
        direction = "up" if pct >= 0 else "down"
        return f"order frequency {direction} {abs(pct):.0f}% versus comparison period"
    if "net_price_per_unit_ratio" in feature:
        pct = (float(v) - 1.0) * 100.0
        direction = "up" if pct >= 0 else "down"
        return f"unit price {direction} {abs(pct):.0f}% versus prior period"
    if feature.startswith("order_to_invoice_days"):
        return f"{float(v):.1f} average order-to-invoice days"
    if feature == "avg_order_value_365_all":
        return "annual average order value $" + f"{float(v):,.0f}"
    if feature in {"spend_0_30_g", "spend_181_365_g", "spend_0_30_all", "spend_181_365_all"}:
        return "$" + f"{float(v):,.0f}"
    if feature in {"freq_0_30_g", "freq_181_365_g", "freq_0_30_all", "freq_181_365_all"}:
        return f"{int(round(float(v)))} orders"
    if feature in {"median_cycle_days", "median_cycle_days_all"}:
        return f"{float(v):.0f}-day typical buying cycle"
    if feature == "recency_delta":
        return f"{float(v):.0f}-day category/account recency gap"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _reason_text(feature: str, value, shap_value: float) -> str:
    readable = {
        "recency_days": "Category recency",
        "recency_days_all": "Account recency",
        "ratio_recency_to_cycle": "Category overdue ratio",
        "ratio_recency_to_cycle_all": "Account overdue ratio",
        "active_months_12_g": "Category activity",
        "active_months_12_all": "Account activity",
        "spend_logratio_3m_g": "Category spend trend",
        "spend_logratio_3m_all": "Account spend trend",
        "freq_logratio_3m_g": "Category order trend",
        "freq_logratio_3m_all": "Account order trend",
        "net_price_per_unit_ratio_3m_g": "Category price change",
        "net_price_per_unit_ratio_3m_all": "Account price change",
        "order_to_invoice_days_3m_g": "Category fulfilment timing",
        "order_to_invoice_days_3m_all": "Account fulfilment timing",
        "avg_order_value_365_all": "Average order value",
        "spend_share_365": "Category annual spend share",
        "mix_shift_share_3m": "Category mix shift",
        "recency_delta": "Category versus account recency",
    }.get(feature, feature.replace("_", " ").title())

    direction = "raises risk" if shap_value > 0 else "lowers risk"
    return f"{readable}: {_feature_value_text(feature, value)} ({direction})"


def _raw_shap_matrix(
    contribs: np.ndarray,
    encoder: dict,
) -> tuple[np.ndarray, list[str]]:
    encoded_names = encoder["encoded_feature_names"]
    raw_names = list(CATEGORY_FEATURE_COLS)
    raw_index = {name: i for i, name in enumerate(raw_names)}
    out = np.zeros((contribs.shape[0], len(raw_names)), dtype=np.float32)

    for j, encoded_name in enumerate(encoded_names):
        if encoded_name in raw_index:
            raw_name = encoded_name
        else:
            raw_name = None
            for cat in encoder["categorical_features"]:
                if encoded_name.startswith(f"{cat}__"):
                    raw_name = cat
                    break
            if raw_name is None:
                continue
        out[:, raw_index[raw_name]] += contribs[:, j]

    return out, raw_names


def _build_reason_outputs(
    scored: pd.DataFrame,
    raw_shap: np.ndarray,
    raw_features: list[str],
    scoring_run_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows = []
    latest_rows = []

    actionable_idx = [
        i for i, f in enumerate(raw_features)
        if f not in NON_ACTIONABLE_REASON_FEATURES
    ]

    for pos, (_, row) in enumerate(scored.iterrows()):
        contributions = raw_shap[pos]
        order = sorted(
            actionable_idx,
            key=lambda i: abs(float(contributions[i])),
            reverse=True,
        )

        base = {c: row[c] for c in CATEGORY_KEYS}
        top_texts = []
        for rank, feature_idx in enumerate(order, start=1):
            feature = raw_features[feature_idx]
            shap_value = float(contributions[feature_idx])
            feature_value = row.get(feature)
            reason_text = _reason_text(feature, feature_value, shap_value)

            long_rows.append({
                **base,
                "scoring_run_id": scoring_run_id,
                "reason_rank": rank,
                "feature_name": feature,
                "feature_value": feature_value,
                "shap_value": shap_value,
                "abs_shap_value": abs(shap_value),
                "direction": "risk_up" if shap_value > 0 else "risk_down",
                "reason_text": reason_text,
            })

            if rank <= TOP_REASON_COUNT:
                top_texts.append(reason_text)

        latest = {**base, "scoring_run_id": scoring_run_id}
        for i in range(TOP_REASON_COUNT):
            latest[f"reason_{i + 1}"] = top_texts[i] if i < len(top_texts) else None
        latest["reason_summary"] = " | ".join(top_texts)
        latest_rows.append(latest)

    return pd.DataFrame(latest_rows), pd.DataFrame(long_rows)


def _customer_rollup(scores: pd.DataFrame) -> pd.DataFrame:
    account_keys = [CUSTOMER_COLUMN, "bulevel1", "zipcode"]

    rows = []
    for key, g in scores.groupby(account_keys, dropna=False, sort=False):
        spend = pd.to_numeric(g["spend_365"], errors="coerce").fillna(0.0)
        total_cat_spend = float(spend.sum())
        if total_cat_spend > 0:
            weighted_risk = float(np.average(g["p_14"], weights=spend))
        else:
            weighted_risk = float(g["p_14"].mean())

        spend_all = float(pd.to_numeric(g["spend_365_all"], errors="coerce").max())
        rows.append({
            CUSTOMER_COLUMN: key[0],
            "bulevel1": key[1],
            "zipcode": key[2],
            "snapshot_dt": g["snapshot_dt"].max(),
            "customergroupname_current": (
                g["customergroupname_current"].dropna().iloc[0]
                if g["customergroupname_current"].notna().any()
                else None
            ),
            "category_count_scored": int(len(g)),
            "max_category_risk": float(g["p_14"].max()),
            "weighted_risk": weighted_risk,
            "risk_score": int(round(weighted_risk * 100)),
            "spend_365_all": spend_all,
            "sum_scored_category_spend_365": total_cat_spend,
            "revenue_at_risk": spend_all * weighted_risk,
        })

    roll = pd.DataFrame(rows)
    roll["risk_tier"] = _risk_tier(roll["weighted_risk"])
    roll["value_band"] = _value_bands(roll["spend_365_all"])
    roll["action"] = _action(roll["value_band"], roll["risk_tier"])
    return roll


def run_weekly_scoring(
    root: str | Path,
    preferred_format: str = "parquet",
) -> dict:
    root = Path(root)

    panel = read_df(
        resolve_table(
            root / "data" / "labels" / "dex_v2_category_labeled_panel_zipcode"
        )
    )
    panel["snapshot_dt"] = pd.to_datetime(panel["snapshot_dt"])
    latest_snapshot = panel["snapshot_dt"].max()
    latest = panel[panel["snapshot_dt"] == latest_snapshot].copy()

    eligible = latest[latest["score_eligible_cat_14"] == 1].copy()
    rejected = latest[latest["score_eligible_cat_14"] != 1].copy()

    if eligible.empty:
        raise ValueError("No score-eligible rows in latest snapshot")
    if eligible.duplicated(CATEGORY_KEYS).any():
        raise AssertionError("Latest score-eligible category keys are not unique")

    artifact_dir = root / "artifacts" / "models" / "category_churn_h14"
    model_path = artifact_dir / "model.json"
    encoder_path = artifact_dir / "encoder.json"
    metadata_path = artifact_dir / "metadata.json"

    if not model_path.exists() or not encoder_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Phase 7 model artifacts are missing. Run scripts/run_model_training.py first."
        )

    encoder = json.loads(encoder_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    missing_features = [c for c in CATEGORY_FEATURE_COLS if c not in eligible.columns]
    if missing_features:
        raise KeyError(f"Latest panel is missing scoring features: {missing_features}")

    X = _transform(eligible[CATEGORY_FEATURE_COLS], encoder)
    dscore = xgb.DMatrix(X, feature_names=encoder["encoded_feature_names"])

    booster = xgb.Booster()
    booster.load_model(model_path)

    best_iteration = metadata.get("best_iteration")
    iteration_range = (
        (0, int(best_iteration) + 1)
        if best_iteration is not None
        else (0, 0)
    )

    probs = booster.predict(dscore, iteration_range=iteration_range)
    contribs = booster.predict(
        dscore,
        pred_contribs=True,
        iteration_range=iteration_range,
    )
    if contribs.shape[1] != len(encoder["encoded_feature_names"]) + 1:
        raise AssertionError("Unexpected SHAP contribution width")

    raw_shap, raw_features = _raw_shap_matrix(contribs[:, :-1], encoder)

    scoring_run_id = (
        f"SCORE_{latest_snapshot.strftime('%Y%m%d')}_"
        f"{datetime.now().strftime('%H%M%S')}_LOCAL"
    )

    scores = eligible.copy()
    scores["p_14"] = probs.astype(float)
    scores["risk_tier"] = _risk_tier(scores["p_14"])
    scores["scoring_run_id"] = scoring_run_id
    scores["model_best_iteration"] = best_iteration
    scores["model_artifact"] = str(model_path)

    rollup = _customer_rollup(scores)
    account_keys = [CUSTOMER_COLUMN, "bulevel1", "zipcode"]
    rollup_for_scores = rollup[
        account_keys + [
            "weighted_risk",
            "risk_score",
            "value_band",
            "action",
            "revenue_at_risk",
        ]
    ].rename(columns={
        "risk_score": "customer_risk_score",
        "action": "customer_action",
    })
    scores = scores.merge(
        rollup_for_scores,
        on=account_keys,
        how="left",
        validate="many_to_one",
    )
    scores["category_action"] = _action(
        scores["value_band"],
        scores["risk_tier"],
    )

    reason_latest, reason_long = _build_reason_outputs(
        scores,
        raw_shap,
        raw_features,
        scoring_run_id,
    )

    viz_cols = CATEGORY_KEYS + [
        "customergroupname_current",
        "p_14",
        "risk_tier",
        "value_band",
        "category_action",
        "weighted_risk",
        "customer_risk_score",
        "customer_action",
        "spend_365",
        "spend_365_all",
        "revenue_at_risk",
        "scoring_run_id",
    ]
    viz = scores[viz_cols].merge(
        reason_latest,
        on=CATEGORY_KEYS + ["scoring_run_id"],
        how="left",
        validate="one_to_one",
    )

    rejected = rejected[
        CATEGORY_KEYS + [
            "score_ineligible_reason_14",
            "customergroupname_current",
            "recency_days",
            "n_gaps_running",
            "active_months_12_g",
        ]
    ].copy()
    rejected["scoring_run_id"] = scoring_run_id

    out = root / "data" / "scoring"
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "scores_wide": str(write_df(
            scores,
            out / "dex_v2_category_loss_scores_wide_zipcode",
            preferred_format,
        )),
        "viz_latest": str(write_df(
            viz,
            out / "dex_v2_category_loss_viz_latest_zipcode",
            preferred_format,
        )),
        "reason_latest": str(write_df(
            reason_latest,
            out / "dex_v2_category_reason_latest_zipcode",
            preferred_format,
        )),
        "reason_long": str(write_df(
            reason_long,
            out / "dex_v2_category_reason_long_zipcode",
            preferred_format,
        )),
        "customer_rollup": str(write_df(
            rollup,
            out / "dex_v2_customer_zip_risk_rollup",
            preferred_format,
        )),
        "reject_log": str(write_df(
            rejected,
            out / "dex_v2_weekly_reject_log",
            preferred_format,
        )),
    }

    risk_counts = {
        str(k): int(v)
        for k, v in scores["risk_tier"].value_counts().sort_index().items()
    }
    value_counts = {
        str(k): int(v)
        for k, v in rollup["value_band"].value_counts().sort_index().items()
    }

    result = {
        "status": "PASS",
        "scoring_run_id": scoring_run_id,
        "snapshot_dt": str(latest_snapshot.date()),
        "rows": {
            "latest_panel": int(len(latest)),
            "score_eligible": int(len(scores)),
            "rejected": int(len(rejected)),
            "customer_zip_rollup": int(len(rollup)),
            "reason_latest": int(len(reason_latest)),
            "reason_long": int(len(reason_long)),
        },
        "risk_counts": risk_counts,
        "value_band_counts_customer_zip": value_counts,
        "probability": {
            "min": float(scores["p_14"].min()),
            "mean": float(scores["p_14"].mean()),
            "max": float(scores["p_14"].max()),
        },
        "paths": paths,
    }
    update_manifest(root, "weekly_scoring", result)
    return result
