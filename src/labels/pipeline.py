from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from src.local_pipeline.io import read_df, resolve_table, write_df, update_manifest
from src.features.contracts import (
    CUSTOMER_COLUMN,
    CATEGORY_COLUMN,
    BILLTO_KEYS,
    CATEGORY_KEYS,
)
from .contracts import (
    HORIZON_DAYS,
    OBS_WINDOW_DAYS,
    LABEL_MATURITY_DAYS,
    MIN_GAPS,
    MIN_ACTIVE_MONTHS_12,
    SCORING_RECENCY_MAX_DAYS,
)


def _reason_string(row: pd.Series, ordered_rules: list[tuple[str, str]]) -> str:
    failed = [reason for col, reason in ordered_rules if int(row[col]) == 0]
    return "|".join(failed)


def _future_post_horizon_counts(
    panel: pd.DataFrame,
    tx: pd.DataFrame,
    keys: list[str],
    horizon_days: int,
    obs_window_days: int,
) -> np.ndarray:
    """Count purchases in (snapshot+horizon, snapshot+horizon+obs_window]."""
    out = np.zeros(len(panel), dtype=np.int32)
    tx_dates = (
        tx[keys + ["purchase_dt"]]
        .dropna(subset=["purchase_dt"])
        .drop_duplicates()
        .sort_values(keys + ["purchase_dt"])
    )
    grouped = {
        k if isinstance(k, tuple) else (k,): pd.to_datetime(g["purchase_dt"]).to_numpy(dtype="datetime64[D]")
        for k, g in tx_dates.groupby(keys, dropna=False, sort=False)
    }

    for key, idx in panel.groupby(keys, dropna=False, sort=False).groups.items():
        kt = key if isinstance(key, tuple) else (key,)
        dates = grouped.get(kt)
        if dates is None or len(dates) == 0:
            continue

        snaps = pd.to_datetime(panel.loc[idx, "snapshot_dt"]).to_numpy(dtype="datetime64[D]")
        left_boundary = snaps + np.timedelta64(horizon_days, "D")
        right_boundary = snaps + np.timedelta64(horizon_days + obs_window_days, "D")

        left = np.searchsorted(dates, left_boundary, side="right")
        right = np.searchsorted(dates, right_boundary, side="right")
        out[np.asarray(idx)] = (right - left).astype(np.int32)

    return out


def _apply_billto_labels(
    bill: pd.DataFrame,
    base: pd.DataFrame,
    panel_end: pd.Timestamp,
) -> pd.DataFrame:
    x = bill.copy()
    x["snapshot_dt"] = pd.to_datetime(x["snapshot_dt"])
    x["last_purchase_dt_all"] = pd.to_datetime(x["last_purchase_dt_all"], errors="coerce")

    x["gate_billto_label_observable_14"] = (
        x["snapshot_dt"] + pd.Timedelta(days=LABEL_MATURITY_DAYS) <= panel_end
    ).astype("int8")
    x["gate_billto_has_last_purchase_14"] = x["last_purchase_dt_all"].notna().astype("int8")
    x["gate_billto_observed_cycle_14"] = x["median_cycle_days_obs_all"].notna().astype("int8")
    x["gate_billto_min_gaps_14"] = (pd.to_numeric(x["n_gaps_running_all"], errors="coerce").fillna(0) >= MIN_GAPS).astype("int8")
    x["gate_billto_min_active_months_14"] = (
        pd.to_numeric(x["active_months_12_all"], errors="coerce").fillna(0) >= MIN_ACTIVE_MONTHS_12
    ).astype("int8")

    train_gates = [
        "gate_billto_label_observable_14",
        "gate_billto_has_last_purchase_14",
        "gate_billto_observed_cycle_14",
        "gate_billto_min_gaps_14",
        "gate_billto_min_active_months_14",
    ]
    x["train_eligible_billto_14"] = x[train_gates].all(axis=1).astype("int8")
    x["eligible_billto_14"] = x["train_eligible_billto_14"]

    account_keys = [CUSTOMER_COLUMN, "bulevel1", "zipcode"]
    x["account_post_horizon_purchase_count_14"] = _future_post_horizon_counts(
        x, base, account_keys, HORIZON_DAYS, OBS_WINDOW_DAYS
    )
    x["account_retained_post_horizon_14"] = (x["account_post_horizon_purchase_count_14"] > 0).astype("int8")

    x["y_billto_14"] = np.where(
        x["train_eligible_billto_14"] == 1,
        np.where(x["account_retained_post_horizon_14"] == 1, 0.0, 1.0),
        np.nan,
    )

    x["score_gate_billto_has_last_purchase_14"] = x["last_purchase_dt_all"].notna().astype("int8")
    x["score_gate_billto_recency_lt_90_14"] = (
        pd.to_numeric(x["recency_days_all"], errors="coerce") < SCORING_RECENCY_MAX_DAYS
    ).fillna(False).astype("int8")
    x["score_gate_billto_min_gaps_14"] = (
        pd.to_numeric(x["n_gaps_running_all"], errors="coerce").fillna(0) >= MIN_GAPS
    ).astype("int8")
    x["score_gate_billto_min_active_months_14"] = (
        pd.to_numeric(x["active_months_12_all"], errors="coerce").fillna(0) >= MIN_ACTIVE_MONTHS_12
    ).astype("int8")
    score_gates = [
        "score_gate_billto_has_last_purchase_14",
        "score_gate_billto_recency_lt_90_14",
        "score_gate_billto_min_gaps_14",
        "score_gate_billto_min_active_months_14",
    ]
    x["score_eligible_billto_14"] = x[score_gates].all(axis=1).astype("int8")

    train_reason_rules = [
        ("gate_billto_label_observable_14", "LABEL_NOT_MATURE"),
        ("gate_billto_has_last_purchase_14", "NO_PRIOR_PURCHASE"),
        ("gate_billto_observed_cycle_14", "NO_OBSERVED_CYCLE"),
        ("gate_billto_min_gaps_14", "INSUFFICIENT_GAPS"),
        ("gate_billto_min_active_months_14", "INSUFFICIENT_ACTIVE_MONTHS"),
    ]
    score_reason_rules = [
        ("score_gate_billto_has_last_purchase_14", "NO_PRIOR_PURCHASE"),
        ("score_gate_billto_recency_lt_90_14", "RECENCY_GE_90"),
        ("score_gate_billto_min_gaps_14", "INSUFFICIENT_GAPS"),
        ("score_gate_billto_min_active_months_14", "INSUFFICIENT_ACTIVE_MONTHS"),
    ]
    x["train_ineligible_reason_billto_14"] = x.apply(
        lambda r: _reason_string(r, train_reason_rules), axis=1
    )
    x["score_ineligible_reason_billto_14"] = x.apply(
        lambda r: _reason_string(r, score_reason_rules), axis=1
    )
    return x


def _apply_category_labels(
    cat: pd.DataFrame,
    base: pd.DataFrame,
    panel_end: pd.Timestamp,
) -> pd.DataFrame:
    x = cat.copy()
    x["snapshot_dt"] = pd.to_datetime(x["snapshot_dt"])
    x["last_purchase_dt"] = pd.to_datetime(x["last_purchase_dt"], errors="coerce")

    x["gate_label_observable_14"] = (
        x["snapshot_dt"] + pd.Timedelta(days=LABEL_MATURITY_DAYS) <= panel_end
    ).astype("int8")
    x["gate_has_last_purchase_14"] = x["last_purchase_dt"].notna().astype("int8")
    x["gate_observed_cycle_14"] = x["median_cycle_days_obs_g"].notna().astype("int8")
    x["gate_min_gaps_14"] = (
        pd.to_numeric(x["n_gaps_running"], errors="coerce").fillna(0) >= MIN_GAPS
    ).astype("int8")
    x["gate_min_active_months_14"] = (
        pd.to_numeric(x["active_months_12_g"], errors="coerce").fillna(0) >= MIN_ACTIVE_MONTHS_12
    ).astype("int8")

    train_gates = [
        "gate_label_observable_14",
        "gate_has_last_purchase_14",
        "gate_observed_cycle_14",
        "gate_min_gaps_14",
        "gate_min_active_months_14",
    ]
    x["train_eligible_14"] = x[train_gates].all(axis=1).astype("int8")
    # Compatibility alias: this is training/label eligibility, never the live scoring gate.
    x["eligible_cat_14"] = x["train_eligible_14"]

    account_keys = [CUSTOMER_COLUMN, "bulevel1", "zipcode"]
    category_keys = [CUSTOMER_COLUMN, "bulevel1", CATEGORY_COLUMN, "zipcode"]

    x["account_post_horizon_purchase_count_14"] = _future_post_horizon_counts(
        x, base, account_keys, HORIZON_DAYS, OBS_WINDOW_DAYS
    )
    x["category_post_horizon_purchase_count_14"] = _future_post_horizon_counts(
        x, base, category_keys, HORIZON_DAYS, OBS_WINDOW_DAYS
    )
    x["account_retained_post_horizon_14"] = (
        x["account_post_horizon_purchase_count_14"] > 0
    ).astype("int8")
    x["category_retained_post_horizon_14"] = (
        x["category_post_horizon_purchase_count_14"] > 0
    ).astype("int8")

    # Target semantics:
    # - unresolved/ineligible training row -> NULL
    # - whole account did not remain active after H -> NULL
    # - category repurchased after H -> 0
    # - account repurchased after H but category did not -> 1
    x["y_cat_activeacct_14"] = np.select(
        [
            x["train_eligible_14"] == 0,
            x["account_retained_post_horizon_14"] == 0,
            x["category_retained_post_horizon_14"] == 1,
        ],
        [np.nan, np.nan, 0.0],
        default=1.0,
    )

    x["label_status_14"] = np.select(
        [
            x["train_eligible_14"] == 0,
            x["account_retained_post_horizon_14"] == 0,
            x["category_retained_post_horizon_14"] == 1,
        ],
        [
            "INELIGIBLE_OR_UNMATURE",
            "WHOLE_ACCOUNT_NOT_RETAINED",
            "CATEGORY_RETAINED",
        ],
        default="CATEGORY_LOSS_ACTIVE_ACCOUNT",
    )

    # Current live-scoring eligibility is intentionally independent of label maturity.
    x["score_gate_has_last_purchase_14"] = x["last_purchase_dt"].notna().astype("int8")
    x["score_gate_recency_lt_90_14"] = (
        pd.to_numeric(x["recency_days"], errors="coerce") < SCORING_RECENCY_MAX_DAYS
    ).fillna(False).astype("int8")
    x["score_gate_min_gaps_14"] = (
        pd.to_numeric(x["n_gaps_running"], errors="coerce").fillna(0) >= MIN_GAPS
    ).astype("int8")
    x["score_gate_min_active_months_14"] = (
        pd.to_numeric(x["active_months_12_g"], errors="coerce").fillna(0) >= MIN_ACTIVE_MONTHS_12
    ).astype("int8")

    score_gates = [
        "score_gate_has_last_purchase_14",
        "score_gate_recency_lt_90_14",
        "score_gate_min_gaps_14",
        "score_gate_min_active_months_14",
    ]
    x["score_eligible_cat_14"] = x[score_gates].all(axis=1).astype("int8")

    train_reason_rules = [
        ("gate_label_observable_14", "LABEL_NOT_MATURE"),
        ("gate_has_last_purchase_14", "NO_PRIOR_PURCHASE"),
        ("gate_observed_cycle_14", "NO_OBSERVED_CYCLE"),
        ("gate_min_gaps_14", "INSUFFICIENT_GAPS"),
        ("gate_min_active_months_14", "INSUFFICIENT_ACTIVE_MONTHS"),
    ]
    score_reason_rules = [
        ("score_gate_has_last_purchase_14", "NO_PRIOR_PURCHASE"),
        ("score_gate_recency_lt_90_14", "RECENCY_GE_90"),
        ("score_gate_min_gaps_14", "INSUFFICIENT_GAPS"),
        ("score_gate_min_active_months_14", "INSUFFICIENT_ACTIVE_MONTHS"),
    ]
    x["train_ineligible_reason_14"] = x.apply(
        lambda r: _reason_string(r, train_reason_rules), axis=1
    )
    x["score_ineligible_reason_14"] = x.apply(
        lambda r: _reason_string(r, score_reason_rules), axis=1
    )
    return x


def _gate_funnel(cat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(cat)
    rows.append({"stage": "category_panel_rows", "rows": total})

    current = pd.Series(True, index=cat.index)
    stages = [
        ("label_observable", "gate_label_observable_14"),
        ("has_last_purchase", "gate_has_last_purchase_14"),
        ("observed_cycle", "gate_observed_cycle_14"),
        ("min_gaps", "gate_min_gaps_14"),
        ("min_active_months", "gate_min_active_months_14"),
    ]
    for label, col in stages:
        current &= cat[col].eq(1)
        rows.append({"stage": label, "rows": int(current.sum())})

    labelable = cat["y_cat_activeacct_14"].notna()
    rows.append({"stage": "category_training_labels", "rows": int(labelable.sum())})

    latest = cat["snapshot_dt"].max()
    latest_rows = cat[cat["snapshot_dt"] == latest]
    rows.append({"stage": "latest_snapshot_rows", "rows": int(len(latest_rows))})
    rows.append({
        "stage": "latest_score_eligible",
        "rows": int((latest_rows["score_eligible_cat_14"] == 1).sum()),
    })
    return pd.DataFrame(rows)


def build_labels_eligibility(root: str | Path, preferred_format: str = "parquet") -> dict:
    root = Path(root)

    cat = read_df(resolve_table(root / "data" / "features" / "dex_v2_checkpoint_category_panel_zipcode"))
    bill = read_df(resolve_table(root / "data" / "features" / "dex_v2_checkpoint_billto_panel_zipcode"))
    base = read_df(resolve_table(root / "data" / "certified" / "dex_v2_base"))

    cfg = yaml.safe_load((root / "config" / "synthetic.yml").read_text(encoding="utf-8"))
    panel_end = pd.Timestamp(cfg["dates"]["end"]).normalize()

    base = base.copy()
    base["purchase_dt"] = pd.to_datetime(base["purchase_dt"], errors="coerce")
    base["netamountextended"] = pd.to_numeric(base["netamountextended"], errors="coerce")
    base["quantity"] = pd.to_numeric(base["quantity"], errors="coerce")
    base = base[
        base["purchase_dt"].notna()
        & (base["netamountextended"] > 0)
        & (base["quantity"] > 0)
    ].copy()

    cat_labeled = _apply_category_labels(cat, base, panel_end)
    bill_labeled = _apply_billto_labels(bill, base, panel_end)

    if cat_labeled.duplicated(CATEGORY_KEYS).any():
        raise AssertionError("Category labeled panel key is not unique")
    if bill_labeled.duplicated(BILLTO_KEYS).any():
        raise AssertionError("Bill-to labeled panel key is not unique")

    funnel = _gate_funnel(cat_labeled)

    out = root / "data" / "labels"
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "category_labeled_panel": str(write_df(
            cat_labeled,
            out / "dex_v2_category_labeled_panel_zipcode",
            preferred_format,
        )),
        "billto_labeled_panel": str(write_df(
            bill_labeled,
            out / "dex_v2_billto_labeled_panel_zipcode",
            preferred_format,
        )),
        "gate_funnel": str(write_df(
            funnel,
            out / "training_gate_stage2",
            preferred_format,
        )),
    }

    latest = pd.to_datetime(cat_labeled["snapshot_dt"]).max()
    latest_rows = cat_labeled[pd.to_datetime(cat_labeled["snapshot_dt"]) == latest]
    labeled = cat_labeled[cat_labeled["y_cat_activeacct_14"].notna()]
    positives = int((labeled["y_cat_activeacct_14"] == 1).sum())
    negatives = int((labeled["y_cat_activeacct_14"] == 0).sum())

    result = {
        "horizon_days": HORIZON_DAYS,
        "obs_window_days": OBS_WINDOW_DAYS,
        "label_maturity_days": LABEL_MATURITY_DAYS,
        "panel_end_dt": str(panel_end.date()),
        "rows": {
            "category_panel": int(len(cat_labeled)),
            "billto_panel": int(len(bill_labeled)),
            "category_labels_resolved": int(len(labeled)),
            "category_positive_labels": positives,
            "category_negative_labels": negatives,
            "latest_score_eligible": int((latest_rows["score_eligible_cat_14"] == 1).sum()),
        },
        "paths": paths,
    }
    update_manifest(root, "labels_eligibility", result)
    return result
