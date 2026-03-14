# app/utils/plot_utils.py
"""
Plotly 绘图工具函数

提供统一风格的交互式图表生成函数，供各 Streamlit 页面使用。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """将十六进制颜色转换为 rgba 字符串"""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


# =========================================================
# 颜色主题
# =========================================================
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def plot_equity_curves(
    curves: Dict[str, pd.DataFrame],
    title: str = "策略净值走势对比",
    height: int = 500,
) -> go.Figure:
    """
    绘制多策略净值曲线

    Parameters
    ----------
    curves : dict
        key=策略名, value=DataFrame(date, equity)
    """
    fig = go.Figure()
    for i, (name, df) in enumerate(curves.items()):
        if df.empty or "equity" not in df.columns:
            continue
        date_col = "date" if "date" in df.columns else df.index
        equity = df["equity"]
        # 归一化为净值（初始=1）
        nav = equity / equity.iloc[0]
        fig.add_trace(
            go.Scatter(
                x=df[date_col] if "date" in df.columns else df.index,
                y=nav,
                name=name,
                line=dict(color=COLORS[i % len(COLORS)], width=2),
                hovertemplate=f"<b>{name}</b><br>日期: %{{x|%Y-%m-%d}}<br>净值: %{{y:.4f}}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="日期",
        yaxis_title="净值",
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_drawdown(
    curves: Dict[str, pd.DataFrame],
    title: str = "回撤曲线",
    height: int = 350,
) -> go.Figure:
    """绘制多策略最大回撤曲线"""
    fig = go.Figure()
    for i, (name, df) in enumerate(curves.items()):
        if df.empty or "equity" not in df.columns:
            continue
        equity = np.array(df["equity"], dtype=float)
        peak = np.maximum.accumulate(equity)
        dd = (equity / np.maximum(peak, 1e-12)) - 1.0
        date_col = df["date"] if "date" in df.columns else df.index

        fig.add_trace(
            go.Scatter(
                x=date_col,
                y=dd * 100,
                name=name,
                line=dict(color=COLORS[i % len(COLORS)], width=1.5),
                fill="tozeroy",
                fillcolor=_hex_to_rgba(COLORS[i % len(COLORS)], alpha=0.1),
                hovertemplate=f"<b>{name}</b><br>日期: %{{x|%Y-%m-%d}}<br>回撤: %{{y:.2f}}%<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="日期",
        yaxis_title="回撤 (%)",
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_metrics_bar(
    comparison_df: pd.DataFrame,
    metric: str,
    title: str = "",
    height: int = 400,
    pct: bool = False,
) -> go.Figure:
    """
    绘制策略指标对比柱状图

    Parameters
    ----------
    comparison_df : pd.DataFrame
        含 strategy 列和 metric 列
    metric : str
        要展示的指标列名
    pct : bool
        是否转为百分比显示
    """
    if comparison_df.empty or metric not in comparison_df.columns:
        return go.Figure()

    df = comparison_df.copy()
    if pct:
        df[metric] = df[metric] * 100

    fig = px.bar(
        df,
        x="strategy",
        y=metric,
        title=title or metric,
        color="strategy",
        color_discrete_sequence=COLORS,
        text=df[metric].round(2),
        height=height,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, xaxis_title="策略", yaxis_title=metric)
    return fig


def plot_metrics_radar(
    comparison_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    title: str = "策略综合雷达图",
    height: int = 500,
) -> go.Figure:
    """绘制策略综合雷达图"""
    if comparison_df.empty:
        return go.Figure()

    if metrics is None:
        metrics = [c for c in ["total_return", "sharpe", "calmar", "win_rate"] if c in comparison_df.columns]
    if not metrics:
        return go.Figure()

    fig = go.Figure()
    for i, row in comparison_df.iterrows():
        values = []
        for m in metrics:
            val = float(row.get(m, 0.0))
            values.append(val)
        values.append(values[0])  # 闭合雷达图

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill="toself",
                name=str(row.get("strategy", f"Strategy {i}")),
                line=dict(color=COLORS[i % len(COLORS)]),
                fillcolor=_hex_to_rgba(COLORS[i % len(COLORS)], alpha=0.15),
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        title=title,
        height=height,
        showlegend=True,
    )
    return fig


def plot_ic_bar(
    ic_df: pd.DataFrame,
    metric: str = "rank_ic",
    top_n: int = 15,
    title: str = "因子 IC 排行",
    height: int = 500,
) -> go.Figure:
    """绘制因子 IC 柱状图"""
    if ic_df.empty or metric not in ic_df.columns:
        return go.Figure()

    df = ic_df.head(top_n).copy()
    colors = ["#d62728" if v < 0 else "#1f77b4" for v in df[metric]]

    fig = go.Figure(
        go.Bar(
            x=df["factor"],
            y=df[metric],
            marker_color=colors,
            text=df[metric].round(4),
            textposition="outside",
        )
    )
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(
        title=title,
        xaxis_title="因子",
        yaxis_title=metric.upper(),
        height=height,
        xaxis_tickangle=-45,
    )
    return fig


def plot_ic_decay(
    decay_df: pd.DataFrame,
    title: str = "因子 IC 衰减",
    height: int = 400,
) -> go.Figure:
    """绘制因子 IC 衰减曲线"""
    if decay_df.empty:
        return go.Figure()

    fig = go.Figure()
    factors = decay_df["factor"].unique() if "factor" in decay_df.columns else []
    for i, factor in enumerate(factors):
        sub = decay_df[decay_df["factor"] == factor]
        fig.add_trace(
            go.Scatter(
                x=sub["period"],
                y=sub["rank_ic"],
                name=factor,
                mode="lines+markers",
                line=dict(color=COLORS[i % len(COLORS)], width=2),
                marker=dict(size=6),
            )
        )
    fig.add_hline(y=0, line_color="gray", line_dash="dash")
    fig.update_layout(
        title=title,
        xaxis_title="持有期（天）",
        yaxis_title="Rank IC",
        height=height,
        hovermode="x unified",
    )
    return fig


def plot_factor_quantile_returns(
    quantile_df: pd.DataFrame,
    factor_name: str = "",
    height: int = 400,
) -> go.Figure:
    """绘制因子分层平均收益"""
    if quantile_df.empty:
        return go.Figure()

    colors = [COLORS[3] if v < 0 else COLORS[0] for v in quantile_df["avg_return"]]
    fig = go.Figure(
        go.Bar(
            x=[f"Q{int(q)}" for q in quantile_df["quantile"]],
            y=quantile_df["avg_return"] * 100,
            marker_color=colors,
            text=(quantile_df["avg_return"] * 100).round(3),
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"{factor_name} 分层平均收益" if factor_name else "分层平均收益",
        xaxis_title="因子分层",
        yaxis_title="平均日收益 (%)",
        height=height,
    )
    return fig
