# app/pages/03_strategy_comparison.py
"""
策略对比页面 - 详细的策略绩效指标对比
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd

from app.utils.data_loader import load_comparison_table
from app.utils.plot_utils import plot_metrics_bar, plot_metrics_radar

st.set_page_config(page_title="策略对比", page_icon="🔄", layout="wide")
st.title("🔄 策略对比")
st.markdown("详细的胜率、最大回撤、夏普比率等多维度对比。")
st.markdown("---")

# ===================== 加载数据 =====================
comparison = load_comparison_table()

if comparison.empty:
    st.warning(
        "⚠️ 未找到回测结果。请先运行：\n"
        "```bash\n"
        "python -m multi_agent_quant.experiments.run_backtest\n"
        "```"
    )
    st.stop()

# ===================== 策略选择 =====================
strategy_names = comparison["strategy"].tolist() if "strategy" in comparison.columns else []
selected = st.multiselect("选择要对比的策略", options=strategy_names, default=strategy_names)
if not selected:
    st.info("请至少选择一个策略。")
    st.stop()

cmp = comparison[comparison["strategy"].isin(selected)].reset_index(drop=True)

# ===================== 综合雷达图 =====================
st.subheader("🕸️ 综合雷达图")

# 找到可用的雷达图指标（归一化到 0-1）
radar_cols = [c for c in ["total_return", "sharpe", "calmar", "win_rate"] if c in cmp.columns]
if radar_cols:
    # 简单归一化
    radar_df = cmp.copy()
    for col in radar_cols:
        col_min = radar_df[col].min()
        col_max = radar_df[col].max()
        if col_max != col_min:
            radar_df[col] = (radar_df[col] - col_min) / (col_max - col_min)
        else:
            radar_df[col] = 0.5

    fig_radar = plot_metrics_radar(radar_df, radar_cols, title="策略综合雷达图（归一化）", height=450)
    st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")

# ===================== 逐项指标对比 =====================
st.subheader("📊 逐项指标对比")

metric_labels = {
    "total_return": ("总收益率", True),
    "annual_return": ("年化收益率", True),
    "max_drawdown": ("最大回撤", True),
    "sharpe": ("夏普比率", False),
    "calmar": ("Calmar 比率", False),
    "win_rate": ("胜率", True),
    "profit_loss_ratio": ("盈亏比", False),
    "num_trades": ("交易次数", False),
}

available_metrics = [(k, v) for k, v in metric_labels.items() if k in cmp.columns]

# 每行两列
for row_idx in range(0, len(available_metrics), 2):
    cols = st.columns(2)
    for col_idx, (metric, (label, pct)) in enumerate(available_metrics[row_idx: row_idx + 2]):
        with cols[col_idx]:
            fig = plot_metrics_bar(cmp, metric, title=label, pct=pct, height=320)
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ===================== 完整对比表格 =====================
st.subheader("📋 完整对比数据")

display_df = cmp.copy()
for col in ["total_return", "annual_return", "max_drawdown", "win_rate"]:
    if col in display_df.columns:
        display_df[col] = display_df[col].apply(lambda x: f"{x * 100:.2f}%")
for col in ["sharpe", "calmar", "profit_loss_ratio"]:
    if col in display_df.columns:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")

st.dataframe(display_df, use_container_width=True)

csv_bytes = cmp.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
st.download_button(
    "⬇️ 下载对比数据 CSV",
    data=csv_bytes,
    file_name="strategy_comparison.csv",
    mime="text/csv",
)
