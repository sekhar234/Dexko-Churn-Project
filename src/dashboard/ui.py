from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


COLORS = {
    "ink": "#0F172A",
    "muted": "#64748B",
    "blue": "#2563EB",
    "cyan": "#0EA5E9",
    "teal": "#14B8A6",
    "green": "#10B981",
    "amber": "#F59E0B",
    "red": "#EF4444",
    "purple": "#8B5CF6",
    "slate": "#94A3B8",
    "line": "#E2E8F0",
}

RISK_COLORS = {
    "High": COLORS["red"],
    "Medium": COLORS["amber"],
    "Low": COLORS["green"],
}

MOVEMENT_COLORS = {
    "INCREASED": COLORS["red"],
    "DECREASED": COLORS["green"],
    "NEW": COLORS["blue"],
    "DROPPED": COLORS["slate"],
    "NO CHANGE": "#CBD5E1",
}


def inject_css(css_path: str | Path) -> None:
    css = Path(css_path).read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str, eyebrow: str = "DexKo Churn Intelligence") -> None:
    st.markdown(
        f"""
        <div class="dashboard-hero">
          <div class="eyebrow">{escape(eyebrow)}</div>
          <h1>{escape(title)}</h1>
          <p>{escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def meta_strip(items: list[tuple[str, str]]) -> None:
    pills = "".join(
        f'<span class="meta-pill"><strong>{escape(str(k))}</strong>&nbsp;{escape(str(v))}</span>'
        for k, v in items
    )
    st.markdown(f'<div class="meta-strip">{pills}</div>', unsafe_allow_html=True)


def section_title(title: str, subtitle: str | None = None) -> None:
    sub = f"<p>{escape(subtitle)}</p>" if subtitle else ""
    st.markdown(
        f'<div class="section-title"><h3>{escape(title)}</h3>{sub}</div>',
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, note: str = "", accent: str = "blue") -> None:
    color = COLORS.get(accent, COLORS["blue"])
    st.markdown(
        f"""
        <div class="kpi-card" style="--accent:{color};">
          <div class="kpi-label">{escape(label)}</div>
          <div class="kpi-value">{escape(value)}</div>
          <div class="kpi-note">{escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_class(value: str) -> str:
    text = str(value).upper()
    if text in {"PASS", "STABLE", "LOW", "NO_RETRAIN", "EVALUATED"}:
        return "status-green"
    if text in {"WATCH", "MEDIUM", "BLOCKED_WAITING_FOR_LABELS", "BLOCKED_INSUFFICIENT_MATURE_ROWS"}:
        return "status-amber"
    if text in {"FAIL", "ACT", "HIGH", "RETRAIN_RECOMMENDED"}:
        return "status-red"
    if text in {"RUNNING", "NEW"}:
        return "status-blue"
    return "status-slate"


def status_badge(value: str) -> str:
    return (
        f'<span class="status-badge {status_class(value)}">'
        f"{escape(str(value))}</span>"
    )


def info_card(items: list[tuple[str, object]]) -> None:
    body = "".join(
        f"""
        <div>
          <div class="info-label">{escape(str(label))}</div>
          <div class="info-value">{escape('' if pd.isna(value) else str(value))}</div>
        </div>
        """
        for label, value in items
    )
    st.markdown(
        f'<div class="info-card"><div class="info-grid">{body}</div></div>',
        unsafe_allow_html=True,
    )


def _base_layout(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=35, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["ink"]),
        hoverlabel=dict(bgcolor="#FFFFFF", font_color=COLORS["ink"]),
        showlegend=False,
    )
    return fig


def donut_chart(df: pd.DataFrame, label_col: str, value_col: str, color_map: dict[str, str]) -> go.Figure:
    labels = df[label_col].astype(str).tolist()
    values = df[value_col].astype(float).tolist()
    colors = [color_map.get(x, COLORS["slate"]) for x in labels]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=.64,
            marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Rows: %{value:,}<br>%{percent}<extra></extra>",
        )
    )
    _base_layout(fig, 330)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-.05, x=.5, xanchor="center"))
    return fig


def horizontal_bar(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    color: str = "#2563EB",
    color_col: str | None = None,
    color_map: dict[str, str] | None = None,
    height: int = 360,
    currency: bool = False,
) -> go.Figure:
    x = df[value_col].astype(float)
    y = df[category_col].astype(str)
    colors = color
    if color_col and color_map:
        colors = [color_map.get(str(v), COLORS["slate"]) for v in df[color_col]]
    hover = "$%{x:,.0f}" if currency else "%{x:,.0f}"
    fig = go.Figure(
        go.Bar(
            x=x,
            y=y,
            orientation="h",
            marker=dict(color=colors),
            hovertemplate=f"<b>%{{y}}</b><br>{hover}<extra></extra>",
        )
    )
    _base_layout(fig, height)
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(gridcolor="#EEF2F7", zeroline=False)
    return fig


def risk_gauge(score: float) -> go.Figure:
    score = max(0.0, min(float(score), 100.0))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"font": {"size": 42, "color": COLORS["ink"]}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": COLORS["muted"]},
                "bar": {"color": COLORS["blue"], "thickness": .22},
                "bgcolor": "#FFFFFF",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 20], "color": "#DCFCE7"},
                    {"range": [20, 40], "color": "#FEF3C7"},
                    {"range": [40, 100], "color": "#FEE2E2"},
                ],
                "threshold": {
                    "line": {"color": COLORS["red"], "width": 4},
                    "thickness": .72,
                    "value": 40,
                },
            },
            title={"text": "Customer Risk Score", "font": {"size": 16, "color": COLORS["muted"]}},
        )
    )
    _base_layout(fig, 265)
    return fig


def psi_chart(df: pd.DataFrame) -> go.Figure:
    colors = [
        COLORS["red"] if str(s) == "ACT"
        else COLORS["amber"] if str(s) == "WATCH"
        else COLORS["green"]
        for s in df["Status"]
    ]
    fig = go.Figure(
        go.Bar(
            x=df["PSI"].astype(float),
            y=df["Feature"].astype(str),
            orientation="h",
            marker=dict(color=colors),
            customdata=df[["Status", "Critical"]].astype(str).to_numpy(),
            hovertemplate=(
                "<b>%{y}</b><br>PSI: %{x:.3f}<br>Status: %{customdata[0]}"
                "<br>Critical: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    _base_layout(fig, max(380, 26 * len(df)))
    fig.add_vline(x=.10, line_dash="dash", line_color=COLORS["amber"], annotation_text="WATCH 0.10")
    fig.add_vline(x=.25, line_dash="dash", line_color=COLORS["red"], annotation_text="ACT 0.25")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(gridcolor="#EEF2F7", zeroline=False, title="PSI")
    return fig
