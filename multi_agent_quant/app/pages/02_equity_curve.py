# app/pages/02_equity_curve.py
"""
资金曲线页面 - 多策略净值走势对比、回撤展示
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from multi_agent_quant.app.utils.data_loader import load_equity_curves
from multi_agent_quant.app.utils.plot_utils import plot_equity_curves, plot_drawdown

st.set_page_config(page_title="资金曲线", page_icon="📈", layout="wide")
st.title("📈 资金曲线")
st.markdown("多策略净值走势对比与回撤分析。")
st.markdown("---")

# ===================== 加载数据 =====================
all_curves = load_equity_curves()

if not all_curves:
    st.warning(
        "⚠️ 未找到净值曲线数据。请先运行：\n"
        "```bash\n"
        "python -m multi_agent_quant.experiments.run_backtest\n"
        "```"
    )
    st.stop()

# ===================== 策略选择 =====================
strategy_names = list(all_curves.keys())
selected = st.multiselect(
    "选择要对比的策略",
    options=strategy_names,
    default=strategy_names,
)

if not selected:
    st.info("请至少选择一个策略。")
    st.stop()

curves_selected = {k: v for k, v in all_curves.items() if k in selected}

# ===================== 日期范围筛选 =====================
# 找出公共日期范围
all_dates = []
for df in curves_selected.values():
    if "date" in df.columns:
        all_dates.extend(pd.to_datetime(df["date"]).tolist())

if all_dates:
    min_date = min(all_dates).date()
    max_date = max(all_dates).date()
    date_range = st.date_input(
        "日期范围",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if len(date_range) == 2:
        start_dt, end_dt = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        filtered_curves = {}
        for name, df in curves_selected.items():
            if "date" in df.columns:
                mask = (pd.to_datetime(df["date"]) >= start_dt) & (pd.to_datetime(df["date"]) <= end_dt)
                filtered_curves[name] = df[mask].reset_index(drop=True)
            else:
                filtered_curves[name] = df
        curves_selected = filtered_curves

st.markdown("---")

# ===================== 净值曲线 =====================
st.subheader("📈 净值走势（归一化至 1）")
fig_nav = plot_equity_curves(curves_selected, title="策略净值对比", height=500)
st.plotly_chart(fig_nav, use_container_width=True)

# ===================== 回撤曲线 =====================
st.subheader("📉 回撤曲线")
fig_dd = plot_drawdown(curves_selected, title="策略回撤对比", height=350)
st.plotly_chart(fig_dd, use_container_width=True)

# ===================== 数据下载 =====================
st.markdown("---")
st.subheader("⬇️ 下载净值曲线数据")

for name, df in curves_selected.items():
    csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        f"下载 {name} 净值曲线",
        data=csv_bytes,
        file_name=f"equity_curve_{name}.csv",
        mime="text/csv",
        key=f"dl_{name}",
    )
