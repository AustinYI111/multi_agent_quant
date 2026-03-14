# app/pages/01_dashboard.py
"""
仪表板页面 - 总体回测指标展示
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd

from app.utils.data_loader import load_comparison_table, load_equity_curves
from app.utils.plot_utils import plot_equity_curves, plot_metrics_bar

st.set_page_config(page_title="仪表板", page_icon="📊", layout="wide")
st.title("📊 仪表板")
st.markdown("总体回测指标、关键绩效展示。")
st.markdown("---")

# ===================== 加载数据 =====================
comparison = load_comparison_table()
curves = load_equity_curves()

if comparison.empty:
    st.warning(
        "⚠️ 未找到回测结果。请先运行：\n"
        "```bash\n"
        "python -m multi_agent_quant.experiments.run_backtest\n"
        "```"
    )
    st.stop()

# ===================== 关键指标 KPI =====================
st.subheader("📌 关键指标总览")

# 默认展示最佳策略
best_idx = comparison["sharpe"].idxmax() if "sharpe" in comparison.columns else 0
best = comparison.iloc[best_idx]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric(
    "最佳策略",
    str(best.get("strategy", "N/A")),
)
col2.metric(
    "总收益率",
    f"{best.get('total_return', 0) * 100:.2f}%",
)
col3.metric(
    "年化收益率",
    f"{best.get('annual_return', 0) * 100:.2f}%",
)
col4.metric(
    "最大回撤",
    f"{best.get('max_drawdown', 0) * 100:.2f}%",
)
col5.metric(
    "夏普比率",
    f"{best.get('sharpe', 0):.3f}",
)

st.markdown("---")

# ===================== 策略对比表格 =====================
st.subheader("📋 策略对比表格")

# 格式化展示
display_df = comparison.copy()
pct_cols = ["total_return", "annual_return", "max_drawdown", "win_rate"]
for col in pct_cols:
    if col in display_df.columns:
        display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%")

float_cols = ["sharpe", "calmar", "profit_loss_ratio"]
for col in float_cols:
    if col in display_df.columns:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")

st.dataframe(display_df, use_container_width=True)

# CSV 下载
csv_bytes = comparison.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
st.download_button(
    "⬇️ 下载对比表格 CSV",
    data=csv_bytes,
    file_name="comparison_table.csv",
    mime="text/csv",
)

st.markdown("---")

# ===================== 净值曲线预览 =====================
st.subheader("📈 净值曲线快速预览")

if curves:
    fig = plot_equity_curves(curves, title="各策略净值走势", height=400)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("净值曲线数据不可用，请运行回测后刷新。")

st.markdown("---")

# ===================== 指标柱状图 =====================
st.subheader("📊 关键指标对比")

col1, col2 = st.columns(2)
with col1:
    fig_ret = plot_metrics_bar(
        comparison, "total_return", title="总收益率对比", pct=True, height=350
    )
    st.plotly_chart(fig_ret, use_container_width=True)

with col2:
    fig_sharpe = plot_metrics_bar(
        comparison, "sharpe", title="夏普比率对比", pct=False, height=350
    )
    st.plotly_chart(fig_sharpe, use_container_width=True)
