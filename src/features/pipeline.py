from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from src.local_pipeline.io import read_df, resolve_table, write_df, update_manifest
from .contracts import (
    CUSTOMER_COLUMN, CATEGORY_COLUMN, BILLTO_FEATURE_COLS, CATEGORY_FEATURE_COLS,
    FREQ_SPEND_BANDS, DEFAULT_GROUP_MEDIAN_DAYS,
)


def _visible_monday(s: pd.Series) -> pd.Series:
    """Production PIT rule: Monday purchases visible Monday; Tue-Sun next Monday."""
    s = pd.to_datetime(s)
    delta = (7 - s.dt.weekday) % 7
    return (s + pd.to_timedelta(delta, unit="D")).dt.normalize()


def _business_days(start: pd.Series, end: pd.Series) -> pd.Series:
    out = np.full(len(start), np.nan, dtype=float)
    sv = pd.to_datetime(start, errors="coerce")
    ev = pd.to_datetime(end, errors="coerce")
    mask = sv.notna() & ev.notna() & (ev >= sv)
    if mask.any():
        ss = sv[mask].values.astype("datetime64[D]")
        ee = ev[mask].values.astype("datetime64[D]")
        out[np.where(mask)[0]] = np.busday_count(ss, ee).astype(float)
    return pd.Series(out, index=start.index)


def _rolling_unique_months(dates: pd.Series, activity: pd.Series, days: int = 365) -> np.ndarray:
    d = pd.to_datetime(dates).to_numpy(dtype="datetime64[D]")
    months = pd.to_datetime(dates).dt.to_period("M").astype(str).to_numpy()
    a = activity.to_numpy(dtype=bool)
    out = np.zeros(len(d), dtype=int)
    for i, dt in enumerate(d):
        lo = dt - np.timedelta64(days - 1, "D")
        j = np.searchsorted(d, lo, side="left")
        vals = months[j:i+1][a[j:i+1]]
        out[i] = len(set(vals.tolist())) if len(vals) else 0
    return out


def _rolling_apply_group(g: pd.DataFrame, account: bool) -> pd.DataFrame:
    g = g.sort_values("snapshot_dt").copy()
    idx = pd.DatetimeIndex(g["snapshot_dt"])
    if account:
        inv_col, net_col = "inv_w_all0", "net_w_all0"
        nprice_col, qtyprice_col = "net_price_num_w_all0", "qty_price_w_all0"
        otn_col, otd_col = "order_to_invoice_num_w_all0", "order_to_invoice_den_w_all0"
        rec_col, med_col = "recency_days_all", "median_cycle_days_all"
        act_col = "has_activity_w_all"
    else:
        inv_col, net_col = "invoice_cnt_w0", "net_w0"
        nprice_col, qtyprice_col = "net_price_num_w0", "qty_price_w0"
        otn_col, otd_col = "order_to_invoice_num_w0", "order_to_invoice_den_w0"
        rec_col, med_col = "recency_days", "median_cycle_days"
        act_col = "has_activity_w"

    def roll(col: str, window: str, minp: int = 1):
        s = pd.Series(pd.to_numeric(g[col], errors="coerce").fillna(0).to_numpy(), index=idx)
        return s.rolling(window, min_periods=minp).sum().to_numpy()

    r90_inv, r180_inv, r360_inv, r365_inv = roll(inv_col, "90D"), roll(inv_col, "180D"), roll(inv_col, "360D"), roll(inv_col, "365D")
    r90_net, r180_net, r360_net, r365_net = roll(net_col, "90D"), roll(net_col, "180D"), roll(net_col, "360D"), roll(net_col, "365D")

    if account:
        g["freq_90_all"] = r90_inv
        g["freq_180_all"] = r180_inv
        g["freq_365_all"] = r365_inv
        g["spend_90_all"] = r90_net
        g["spend_180_all"] = r180_net
        g["spend_365_all"] = r365_net
        g["active_days_365_all"] = pd.Series((g[inv_col] > 0).astype(float).to_numpy(), index=idx).rolling("365D", min_periods=1).sum().to_numpy()
        g["active_months_12_all"] = _rolling_unique_months(g["snapshot_dt"], g[act_col].astype(bool), 365)
        g["spend_3m_all"] = r90_net
        g["spend_prev3m_all"] = r180_net - r90_net
        g["spend_6m_all"] = r180_net
        g["spend_prev6m_all"] = r360_net - r180_net
        g["freq_3m_all"] = r90_inv
        g["freq_prev3m_all"] = r180_inv - r90_inv
        g["freq_6m_all"] = r180_inv
        g["freq_prev6m_all"] = r360_inv - r180_inv
        pfx = "all"
    else:
        g["freq_90"] = r90_inv
        g["freq_365"] = r365_inv
        g["spend_90"] = r90_net
        g["spend_365"] = r365_net
        g["active_months_12_g"] = _rolling_unique_months(g["snapshot_dt"], g[act_col].astype(bool), 365)
        g["spend_3m_g"] = r90_net
        g["spend_prev3m_g"] = r180_net - r90_net
        g["spend_6m_g"] = r180_net
        g["spend_prev6m_g"] = r360_net - r180_net
        g["freq_3m_g"] = r90_inv
        g["freq_prev3m_g"] = r180_inv - r90_inv
        g["freq_6m_g"] = r180_inv
        g["freq_prev6m_g"] = r360_inv - r180_inv
        pfx = "g"

    for stem in ["spend", "freq"]:
        cur3 = g[f"{stem}_3m_{pfx}"]
        prev3 = g[f"{stem}_prev3m_{pfx}"]
        cur6 = g[f"{stem}_6m_{pfx}"]
        prev6 = g[f"{stem}_prev6m_{pfx}"]
        g[f"{stem}_logratio_3m_{pfx}"] = np.log((cur3 + 1) / (prev3 + 1))
        g[f"{stem}_logratio_6m_{pfx}"] = np.log((cur6 + 1) / (prev6 + 1))

    n90, n180 = roll(nprice_col, "90D"), roll(nprice_col, "180D")
    q90, q180 = roll(qtyprice_col, "90D"), roll(qtyprice_col, "180D")
    nprev, qprev = n180 - n90, q180 - q90
    cur_price = np.divide(n90, q90, out=np.full(len(g), np.nan), where=q90 > 0)
    prev_price = np.divide(nprev, qprev, out=np.full(len(g), np.nan), where=qprev > 0)
    ratio = np.divide(cur_price, prev_price, out=np.full(len(g), np.nan), where=(cur_price > 0) & (prev_price > 0))

    o90, o180 = roll(otn_col, "90D"), roll(otn_col, "180D")
    d90, d180 = roll(otd_col, "90D"), roll(otd_col, "180D")
    oprev, dprev = o180 - o90, d180 - d90
    cur_ot = np.divide(o90, d90, out=np.full(len(g), np.nan), where=d90 > 0)
    prev_ot = np.divide(oprev, dprev, out=np.full(len(g), np.nan), where=dprev > 0)

    if account:
        g["net_price_per_unit_3m_all"] = cur_price
        g["net_price_per_unit_prev3m_all"] = prev_price
        g["net_price_per_unit_ratio_3m_all"] = ratio
        g["net_price_per_unit_logratio_3m_all"] = np.where((cur_price > 0) & (prev_price > 0), np.log(cur_price / prev_price), np.nan)
        g["order_to_invoice_days_3m_all"] = cur_ot
        g["order_to_invoice_days_prev3m_all"] = prev_ot
        g["order_to_invoice_days_delta_3m_all"] = cur_ot - prev_ot
    else:
        g["net_price_per_unit_3m_g"] = cur_price
        g["net_price_per_unit_prev3m_g"] = prev_price
        g["net_price_per_unit_ratio_3m_g"] = ratio
        g["net_price_per_unit_logratio_3m_g"] = np.where((cur_price > 0) & (prev_price > 0), np.log(cur_price / prev_price), np.nan)
        g["order_to_invoice_days_3m_g"] = cur_ot
        g["order_to_invoice_days_prev3m_g"] = prev_ot
        g["order_to_invoice_days_delta_3m_g"] = cur_ot - prev_ot

    med = pd.to_numeric(g[med_col], errors="coerce").to_numpy()
    ratio_rec = np.divide(pd.to_numeric(g[rec_col], errors="coerce"), med, out=np.full(len(g), np.nan), where=med > 0)
    capped = np.minimum(ratio_rec, 3.0)
    if account:
        g["ratio_recency_to_cycle_all"] = ratio_rec
        g["ratio_capped_all"] = capped
        g["log_ratio_all"] = np.log1p(capped)
    else:
        g["ratio_recency_to_cycle"] = ratio_rec
        g["ratio_capped"] = capped
        g["log_ratio"] = np.log1p(capped)
    return g


def _expanding_purchase_stats(day: pd.DataFrame, keys: list[str], suffix: str) -> pd.DataFrame:
    parts = []
    for _, g in day.groupby(keys, sort=False, dropna=False):
        g = g.sort_values("purchase_dt").copy()
        gaps = g["purchase_dt"].diff().dt.days
        g["gap_days" + suffix] = gaps
        g["n_gaps_running" + suffix] = gaps.notna().cumsum().astype(int)
        g["median_gap_running_obs" + suffix] = gaps.expanding(min_periods=1).median()
        g["next_purchase_dt" + suffix] = g["purchase_dt"].shift(-1)
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else day.copy()


def _scaffold(stats: pd.DataFrame, keys: list[str], first_col: str, panel_end_week: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for r in stats.itertuples(index=False):
        data = r._asdict()
        start = pd.Timestamp(data[first_col]) - pd.to_timedelta(pd.Timestamp(data[first_col]).weekday(), unit="D")
        for dt in pd.date_range(start, panel_end_week, freq="7D"):
            out = {k: data[k] for k in keys}
            out["snapshot_dt"] = dt
            out["snapshot_asof_dt"] = dt
            rows.append(out)
    return pd.DataFrame(rows)


def _add_bands(panel: pd.DataFrame, tx: pd.DataFrame, id_cols: list[str], suffix: str) -> pd.DataFrame:
    # Current production helper wrappers omit ZIP; local replica includes ZIP to preserve
    # the explicit Customer × ZIP grain applied throughout the latest feature notebook.
    panel = panel.copy()
    frames = []
    tx_inv = tx.groupby(id_cols + ["purchase_dt", "invoiceid"], dropna=False, as_index=False)["netamountextended"].sum()
    grouped = {k if isinstance(k, tuple) else (k,): g.sort_values("purchase_dt") for k, g in tx_inv.groupby(id_cols, dropna=False, sort=False)}
    for key, pg in panel.groupby(id_cols, dropna=False, sort=False):
        kt = key if isinstance(key, tuple) else (key,)
        t = grouped.get(kt)
        out = pg[[*id_cols, "snapshot_asof_dt"]].copy()
        if t is None or t.empty:
            for band, _, _ in FREQ_SPEND_BANDS:
                out[f"freq_{band}{suffix}"] = 0.0
                out[f"spend_{band}{suffix}"] = 0.0
        else:
            dates = pd.to_datetime(t["purchase_dt"]).to_numpy(dtype="datetime64[D]")
            spends = t["netamountextended"].to_numpy(float)
            prefix = np.r_[0.0, np.cumsum(spends)]
            snaps = pd.to_datetime(pg["snapshot_asof_dt"]).to_numpy(dtype="datetime64[D]")
            for band, lo, hi in FREQ_SPEND_BANDS:
                left = np.searchsorted(dates, snaps - np.timedelta64(hi, "D"), side="left")
                right = np.searchsorted(dates, snaps - np.timedelta64(lo, "D"), side="right")
                out[f"freq_{band}{suffix}"] = (right - left).astype(float)
                out[f"spend_{band}{suffix}"] = prefix[right] - prefix[left]
        out[f"freq_ratio_recent_vs_prior{suffix}"] = (out[f"freq_0_30{suffix}"] + 1) / (out[f"freq_31_90{suffix}"] + 1)
        out[f"freq_ratio_mid_vs_older{suffix}"] = (out[f"freq_31_90{suffix}"] + 1) / (out[f"freq_91_180{suffix}"] + 1)
        out[f"spend_ratio_recent_vs_prior{suffix}"] = (out[f"spend_0_30{suffix}"] + 1) / (out[f"spend_31_90{suffix}"] + 1)
        den = out[f"freq_0_30{suffix}"] + out[f"freq_31_90{suffix}"]
        out[f"freq_share_0_30_in_90{suffix}"] = np.where(den > 0, out[f"freq_0_30{suffix}"] / den, np.nan)
        frames.append(out)
    bands = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return panel.merge(bands, on=id_cols + ["snapshot_asof_dt"], how="left", validate="one_to_one")


def _peak_months(df: pd.DataFrame, panel_end: pd.Timestamp) -> pd.DataFrame:
    recent_start = panel_end - pd.DateOffset(months=12)
    prior_start = panel_end - pd.DateOffset(months=24)
    x = df[(df.netamountextended > 0) & df.purchase_dt.notna()].copy()
    x["window"] = np.select(
        [(x.purchase_dt >= recent_start) & (x.purchase_dt < panel_end), (x.purchase_dt >= prior_start) & (x.purchase_dt < recent_start)],
        ["recent", "prior"], default=None,
    )
    x = x[x.window.notna()].copy()
    x["month_label"] = x.purchase_dt.dt.strftime("%b-%y")
    x["ym"] = x.purchase_dt.dt.to_period("M")
    m = x.groupby([CUSTOMER_COLUMN, "zipcode", "window", "ym", "month_label"], as_index=False)["netamountextended"].sum()
    m = m.sort_values([CUSTOMER_COLUMN, "zipcode", "window", "netamountextended", "month_label"], ascending=[True, True, True, False, True])
    m["rank"] = m.groupby([CUSTOMER_COLUMN, "zipcode", "window"]).cumcount() + 1
    m = m[m["rank"] <= 3]
    piv = m.pivot_table(index=[CUSTOMER_COLUMN, "zipcode"], columns=["window", "rank"], values="month_label", aggfunc="first")
    piv.columns = [f"{w}_peak_{int(rank)}" for w, rank in piv.columns]
    return piv.reset_index()


def _top_categories(df: pd.DataFrame) -> pd.DataFrame:
    maxdt = df.purchase_dt.max()
    x = df[(df.netamountextended > 0) & (df.purchase_dt >= maxdt - pd.Timedelta(days=365))]
    cs = x.groupby([CUSTOMER_COLUMN, "zipcode", CATEGORY_COLUMN], as_index=False)["netamountextended"].sum().rename(columns={"netamountextended": "cat_spend"})
    rows = []
    for (cust, z), g in cs.groupby([CUSTOMER_COLUMN, "zipcode"], sort=False):
        g = g.sort_values(["cat_spend", CATEGORY_COLUMN], ascending=[False, True]).reset_index(drop=True)
        total = g.cat_spend.sum()
        cum_excl = g.cat_spend.cumsum() - g.cat_spend
        keep = g[cum_excl / total < 0.80] if total > 0 else g.iloc[:0]
        rows.append({
            CUSTOMER_COLUMN: cust,
            "zipcode": z,
            "top_3_categories": ", ".join(g[CATEGORY_COLUMN].head(3)),
            "top_80pct_categories": ", ".join(keep[CATEGORY_COLUMN]),
            "n_categories_80pct": len(keep),
        })
    return pd.DataFrame(rows)


def build_features(root: str | Path, preferred_format: str = "parquet") -> dict:
    root = Path(root)
    base = read_df(resolve_table(root / "data" / "certified" / "dex_v2_base"))
    attrs = read_df(resolve_table(root / "data" / "certified" / "dex_v2_cust_attrs"))
    cfg = yaml.safe_load((root / "config" / "synthetic.yml").read_text())
    panel_end = pd.Timestamp(cfg["dates"]["end"]).normalize()
    panel_end_week = panel_end - pd.Timedelta(days=panel_end.weekday())

    df = base.copy()
    df["purchase_dt"] = pd.to_datetime(df["purchase_dt"], errors="coerce")
    df["salesorder_dt"] = pd.to_datetime(df["salesorder_dt"], errors="coerce")
    df["netamountextended"] = pd.to_numeric(df["netamountextended"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df = df[df.purchase_dt.notna() & (df.netamountextended > 0) & ~df[CATEGORY_COLUMN].astype(str).str.upper().isin(["UNASSIGNED", "N/A"])].copy()
    df["salesrep_raw"] = df["salesrep"].astype("string").str.strip().replace("", pd.NA)
    df["creditmax_num"] = pd.to_numeric(df["creditmax"], errors="coerce")
    df["order_to_invoice_days_raw"] = _business_days(df.salesorder_dt, df.purchase_dt)
    df["purchase_week_start"] = _visible_monday(df.purchase_dt)
    df["net_price_num_raw"] = np.where(df.quantity > 0, df.netamountextended, 0.0)
    df["qty_price_raw"] = np.where(df.quantity > 0, df.quantity, 0.0)

    cat_keys = [CUSTOMER_COLUMN, "bulevel1", CATEGORY_COLUMN, "zipcode"]
    acct_keys = [CUSTOMER_COLUMN, "bulevel1", "zipcode"]

    g_day = df.groupby(cat_keys + ["purchase_dt"], as_index=False, dropna=False).agg(
        net_day=("netamountextended", "sum"), qty_day=("quantity", "sum"),
        invoice_cnt_day=("invoiceid", "nunique"), net_price_num_day=("net_price_num_raw", "sum"),
        qty_price_day=("qty_price_raw", "sum"),
    )
    g_hist = _expanding_purchase_stats(g_day, cat_keys, "")
    g_last = g_hist[cat_keys + ["purchase_dt", "next_purchase_dt", "median_gap_running_obs", "n_gaps_running"]].rename(columns={"purchase_dt": "last_purchase_dt", "median_gap_running_obs": "median_cycle_days_obs_g"})
    max_g = g_day.groupby(cat_keys, as_index=False)["purchase_dt"].max().rename(columns={"purchase_dt": "max_purchase_dt_g"})
    cat_median = g_hist.groupby(CATEGORY_COLUMN)["gap_days"].median()

    c_day = df.groupby(acct_keys + ["purchase_dt"], as_index=False, dropna=False).agg(
        net_day_all=("netamountextended", "sum"), qty_day_all=("quantity", "sum"),
        invoice_cnt_day_all=("invoiceid", "nunique"), net_price_num_day_all=("net_price_num_raw", "sum"),
        qty_price_day_all=("qty_price_raw", "sum"),
    )
    c_hist = _expanding_purchase_stats(c_day, acct_keys, "_all")
    c_last = c_hist[acct_keys + ["purchase_dt", "next_purchase_dt_all", "median_gap_running_obs_all", "n_gaps_running_all"]].rename(columns={"purchase_dt": "last_purchase_dt_all", "median_gap_running_obs_all": "median_cycle_days_obs_all"})
    max_c = c_day.groupby(acct_keys, as_index=False)["purchase_dt"].max().rename(columns={"purchase_dt": "max_purchase_dt_all"})

    g_stats = g_day.groupby(cat_keys, as_index=False).purchase_dt.min().rename(columns={"purchase_dt": "first_purchase_dt"})
    c_stats = c_day.groupby(acct_keys, as_index=False).purchase_dt.min().rename(columns={"purchase_dt": "first_purchase_dt_all"})
    g_snap = _scaffold(g_stats, cat_keys, "first_purchase_dt", panel_end_week)
    c_snap = _scaffold(c_stats, acct_keys, "first_purchase_dt_all", panel_end_week)

    c_week = df.groupby(acct_keys + ["purchase_week_start"], as_index=False, dropna=False).agg(
        purchase_dt_in_week_all=("purchase_dt", "max"), net_w_all=("netamountextended", "sum"),
        qty_w_all=("quantity", "sum"), invoice_cnt_w_all=("invoiceid", "nunique"),
        net_price_num_w_all=("net_price_num_raw", "sum"), qty_price_w_all=("qty_price_raw", "sum"),
        order_to_invoice_num_w_all=("order_to_invoice_days_raw", "sum"), order_to_invoice_den_w_all=("order_to_invoice_days_raw", "count"),
        creditmax_obs_w=("creditmax_num", "max"), salesrep_obs_w=("salesrep_raw", "last"),
    ).rename(columns={"purchase_week_start": "snapshot_dt"})
    c = c_snap.merge(c_week, on=acct_keys + ["snapshot_dt"], how="left", validate="one_to_one").sort_values(acct_keys + ["snapshot_dt"])
    for src, dst in [
        ("net_w_all", "net_w_all0"), ("qty_w_all", "qty_w_all0"), ("invoice_cnt_w_all", "inv_w_all0"),
        ("net_price_num_w_all", "net_price_num_w_all0"), ("qty_price_w_all", "qty_price_w_all0"),
        ("order_to_invoice_num_w_all", "order_to_invoice_num_w_all0"), ("order_to_invoice_den_w_all", "order_to_invoice_den_w_all0"),
    ]:
        c[dst] = pd.to_numeric(c[src], errors="coerce").fillna(0)
    c["last_purchase_dt_all"] = c.groupby(acct_keys)["purchase_dt_in_week_all"].ffill()
    c["creditmax_current"] = c.groupby(acct_keys)["creditmax_obs_w"].ffill()
    c["salesrep_current"] = c.groupby(acct_keys)["salesrep_obs_w"].ffill()
    c = c.merge(c_last, on=acct_keys + ["last_purchase_dt_all"], how="left", validate="many_to_one").merge(max_c, on=acct_keys, how="left", validate="many_to_one")
    c["median_cycle_days_all"] = c["median_cycle_days_obs_all"]
    c["recency_days_all"] = (c.snapshot_asof_dt - c.last_purchase_dt_all).dt.days
    c["has_activity_w_all"] = (c.inv_w_all0 > 0).astype(int)
    c["snapshot_month_idx"] = (c.snapshot_asof_dt.dt.year - 2000) * 12 + c.snapshot_asof_dt.dt.month - 1
    c = pd.concat([_rolling_apply_group(g, True) for _, g in c.groupby(acct_keys, dropna=False, sort=False)], ignore_index=True)
    c = c.merge(c_stats[acct_keys + ["first_purchase_dt_all"]], on=acct_keys, how="left")
    c["tenure_days_all"] = (c.snapshot_asof_dt - c.first_purchase_dt_all).dt.days
    c["avg_order_value_365_all"] = np.where(c.freq_365_all > 0, c.spend_365_all / c.freq_365_all, np.nan)
    c["avg_order_value_90_all"] = np.where(c.freq_90_all > 0, c.spend_90_all / c.freq_90_all, np.nan)
    c["spend_volatility_6m_all"] = pd.concat([
        pd.Series(pd.to_numeric(g.net_w_all0, errors="coerce").to_numpy(), index=pd.DatetimeIndex(g.snapshot_dt)).rolling("180D", min_periods=2).std().reset_index(drop=True)
        for _, g in c.groupby(acct_keys, dropna=False, sort=False)
    ], ignore_index=True)

    g_week = df.groupby(cat_keys + ["purchase_week_start"], as_index=False, dropna=False).agg(
        purchase_dt_in_week=("purchase_dt", "max"), net_w=("netamountextended", "sum"),
        qty_w=("quantity", "sum"), invoice_cnt_w=("invoiceid", "nunique"),
        net_price_num_w=("net_price_num_raw", "sum"), qty_price_w=("qty_price_raw", "sum"),
        order_to_invoice_num_w=("order_to_invoice_days_raw", "sum"), order_to_invoice_den_w=("order_to_invoice_days_raw", "count"),
    ).rename(columns={"purchase_week_start": "snapshot_dt"})
    p = g_snap.merge(g_week, on=cat_keys + ["snapshot_dt"], how="left", validate="one_to_one").sort_values(cat_keys + ["snapshot_dt"])
    for src, dst in [
        ("net_w", "net_w0"), ("qty_w", "qty_w0"), ("invoice_cnt_w", "invoice_cnt_w0"),
        ("net_price_num_w", "net_price_num_w0"), ("qty_price_w", "qty_price_w0"),
        ("order_to_invoice_num_w", "order_to_invoice_num_w0"), ("order_to_invoice_den_w", "order_to_invoice_den_w0"),
    ]:
        p[dst] = pd.to_numeric(p[src], errors="coerce").fillna(0)
    p["last_purchase_dt"] = p.groupby(cat_keys)["purchase_dt_in_week"].ffill()
    p = p.merge(g_last, on=cat_keys + ["last_purchase_dt"], how="left", validate="many_to_one").merge(max_g, on=cat_keys, how="left", validate="many_to_one")
    p["category_median_gap"] = p[CATEGORY_COLUMN].map(cat_median)
    p["median_cycle_days"] = p["median_cycle_days_obs_g"].fillna(p.category_median_gap).fillna(DEFAULT_GROUP_MEDIAN_DAYS)
    p["recency_days"] = (p.snapshot_asof_dt - p.last_purchase_dt).dt.days
    p["has_activity_w"] = (p.invoice_cnt_w0 > 0).astype(int)
    p["snapshot_month_idx"] = (p.snapshot_asof_dt.dt.year - 2000) * 12 + p.snapshot_asof_dt.dt.month - 1
    p = pd.concat([_rolling_apply_group(g, False) for _, g in p.groupby(cat_keys, dropna=False, sort=False)], ignore_index=True)

    breadth = []
    vis = df[[*acct_keys, CATEGORY_COLUMN, "purchase_week_start"]].drop_duplicates()
    for key, cg in c.groupby(acct_keys, dropna=False, sort=False):
        kt = key if isinstance(key, tuple) else (key,)
        mask = np.ones(len(vis), dtype=bool)
        for col, val in zip(acct_keys, kt):
            mask &= vis[col].eq(val).to_numpy()
        tx = vis[mask].sort_values("purchase_week_start")
        dates = tx.purchase_week_start.to_numpy(dtype="datetime64[D]")
        cats = tx[CATEGORY_COLUMN].astype(str).to_numpy()
        vals = []
        for dt in cg.snapshot_dt.to_numpy(dtype="datetime64[D]"):
            lo = dt - np.timedelta64(364, "D")
            left = np.searchsorted(dates, lo, "left")
            right = np.searchsorted(dates, dt, "right")
            vals.append(len(set(cats[left:right].tolist())))
        breadth.extend(vals)
    c["distinct_mgrl1_365_all"] = breadth

    attr = attrs[[CUSTOMER_COLUMN, "zipcode", "customergroupname"]].drop_duplicates([CUSTOMER_COLUMN, "zipcode"]).rename(columns={"customergroupname": "customergroupname_current"})
    c = c.merge(attr, on=[CUSTOMER_COLUMN, "zipcode"], how="left", validate="many_to_one")

    c_ctx_cols = acct_keys + [
        "snapshot_dt", "last_purchase_dt_all", "next_purchase_dt_all", "max_purchase_dt_all",
        "median_cycle_days_obs_all", "median_cycle_days_all", "n_gaps_running_all",
        "active_months_12_all", "recency_days_all", "ratio_recency_to_cycle_all", "log_ratio_all",
        "freq_90_all", "freq_180_all", "freq_365_all", "spend_90_all", "spend_180_all", "spend_365_all",
        "spend_3m_all", "spend_prev3m_all", "spend_6m_all", "spend_prev6m_all",
        "freq_3m_all", "freq_prev3m_all", "freq_6m_all", "freq_prev6m_all",
        "spend_logratio_3m_all", "spend_logratio_6m_all", "freq_logratio_3m_all", "freq_logratio_6m_all",
        "creditmax_current", "salesrep_current",
        "order_to_invoice_days_3m_all", "order_to_invoice_days_prev3m_all", "order_to_invoice_days_delta_3m_all",
        "net_price_per_unit_3m_all", "net_price_per_unit_prev3m_all", "net_price_per_unit_ratio_3m_all", "net_price_per_unit_logratio_3m_all",
        "tenure_days_all", "distinct_mgrl1_365_all", "avg_order_value_365_all", "avg_order_value_90_all",
        "spend_volatility_6m_all", "customergroupname_current",
    ]
    p = p.merge(c[c_ctx_cols], on=acct_keys + ["snapshot_dt"], how="left", validate="many_to_one")
    p["spend_share_365"] = np.where(p.spend_365_all > 0, p.spend_365 / p.spend_365_all, 0.0)
    p["freq_share_365"] = np.where(p.freq_365_all > 0, p.freq_365 / p.freq_365_all, 0.0)
    p["recency_delta"] = p.recency_days - p.recency_days_all
    p["spend_share_3m"] = np.where(p.spend_3m_all > 0, p.spend_3m_g / p.spend_3m_all, 0.0)
    p["spend_share_prev3m"] = np.where(p.spend_prev3m_all > 0, p.spend_prev3m_g / p.spend_prev3m_all, 0.0)
    p["mix_shift_share_3m"] = p.spend_share_3m - p.spend_share_prev3m
    mx = p.groupby(acct_keys + ["snapshot_dt"], as_index=False).spend_share_365.max().rename(columns={"spend_share_365": "max_mgrl1_share_365_all"})
    p = p.merge(mx, on=acct_keys + ["snapshot_dt"], how="left", validate="many_to_one")

    c = _add_bands(c, df, acct_keys, "_all")
    p = _add_bands(p, df, cat_keys, "_g")

    c = c[c.customergroupname_current.isin(["OEM", "Stocking Dealer"])].copy()
    p = p[p.customergroupname_current.isin(["OEM", "Stocking Dealer"])].copy()

    assert not c.duplicated(acct_keys + ["snapshot_dt"]).any()
    assert not p.duplicated(cat_keys + ["snapshot_dt"]).any()
    missing_b = [x for x in BILLTO_FEATURE_COLS if x not in c.columns]
    missing_p = [x for x in CATEGORY_FEATURE_COLS if x not in p.columns]
    if missing_b or missing_p:
        raise AssertionError(f"Missing model features: billto={missing_b}, category={missing_p}")

    peak = _peak_months(df, panel_end)
    top = _top_categories(df)
    out = root / "data" / "features"
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "billto_panel": str(write_df(c, out / "dex_v2_checkpoint_billto_panel_zipcode", preferred_format)),
        "category_panel": str(write_df(p, out / "dex_v2_checkpoint_category_panel_zipcode", preferred_format)),
        "peak_months": str(write_df(peak, out / "dex_v2_customer_peak_months_zipcode", preferred_format)),
        "top_categories": str(write_df(top, out / "dex_v2_customer_top_categories_zipcode", preferred_format)),
    }
    result = {
        "panel_end_dt": str(panel_end.date()),
        "rows": {"billto_panel": len(c), "category_panel": len(p), "peak_months": len(peak), "top_categories": len(top)},
        "feature_counts": {"billto": len(BILLTO_FEATURE_COLS), "category": len(CATEGORY_FEATURE_COLS)},
        "paths": paths,
    }
    update_manifest(root, "feature_engineering", result)
    return result
