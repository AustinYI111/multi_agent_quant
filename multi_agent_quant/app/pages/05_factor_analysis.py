# app/pages/05_factor_analysis.py
"""
因子分析页面 - Alpha 因子 IC、衰减、分层回测可视化
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from multi_agent_quant.app.utils.data_loader import (
    load_factor_ic_scores,
    load_factor_ic_decay,
    load_factor_summary,
    get_available_symbols,
)
from multi_agent_quant.app.utils.plot_utils import (
    plot_ic_bar,
    plot_ic_decay,
)

st.set_page_config(page_title="因子分析", page_icon="📉", layout="wide")
st.title("📉 因子分析")
st.markdown("Alpha 因子 IC 分布、衰减曲线与最优因子筛选。")
st.markdown("---")

# ===================== 侧边栏 =====================
with st.sidebar:
    st.header("🔬 因子分析设置")
    available_symbols = get_available_symbols()
    symbol = st.selectbox("股票代码", options=available_symbols, index=0)
    top_n = st.slider("展示 Top N 因子", min_value=5, max_value=20, value=15)

    st.info(
        "如需更新因子数据，请运行：\n"
        "```bash\n"
        "python -m multi_agent_quant.experiments.analyze_factors\n"
        "```"
    )

# ===================== 加载数据 =====================
ic_scores = load_factor_ic_scores(symbol)
ic_decay = load_factor_ic_decay(symbol)
factor_summary = load_factor_summary(symbol)

no_data = ic_scores.empty and ic_decay.empty and factor_summary.empty
if no_data:
    st.warning(
        f"⚠️ 未找到 {symbol} 的因子分析结果。请先运行：\n"
        "```bash\n"
        "python -m multi_agent_quant.experiments.analyze_factors "
        f"--symbol {symbol}\n"
        "```"
    )

# ===================== IC 评分排行 =====================
st.subheader("🏆 因子 IC / Rank IC 排行")

if not ic_scores.empty:
    col1, col2 = st.columns(2)
    with col1:
        fig_ic = plot_ic_bar(
            ic_scores, metric="ic", top_n=top_n, title="因子 IC（Pearson）排行", height=450
        )
        st.plotly_chart(fig_ic, use_container_width=True)

    with col2:
        fig_rank_ic = plot_ic_bar(
            ic_scores, metric="rank_ic", top_n=top_n, title="因子 Rank IC（Spearman）排行", height=450
        )
        st.plotly_chart(fig_rank_ic, use_container_width=True)

    # ICIR 展示
    if "icir" in ic_scores.columns:
        st.subheader("📐 ICIR（IC 信息比率）")
        fig_icir = plot_ic_bar(
            ic_scores, metric="icir", top_n=top_n, title="因子 ICIR 排行", height=400
        )
        st.plotly_chart(fig_icir, use_container_width=True)

else:
    st.info("IC 评分数据不可用。")

st.markdown("---")

# ===================== 因子 IC 衰减 =====================
st.subheader("📉 因子 IC 衰减曲线")
st.markdown("IC 随持有期延长的衰减情况（越慢衰减，因子持续性越强）。")

if not ic_decay.empty:
    # 因子筛选
    if "factor" in ic_decay.columns:
        available_factors = ic_decay["factor"].unique().tolist()
        selected_factors = st.multiselect(
            "选择要展示的因子",
            options=available_factors,
            default=available_factors[:5],
        )
        if selected_factors:
            filtered_decay = ic_decay[ic_decay["factor"].isin(selected_factors)]
            fig_decay = plot_ic_decay(filtered_decay, title="因子 Rank IC 衰减曲线", height=400)
            st.plotly_chart(fig_decay, use_container_width=True)
else:
    st.info("因子衰减数据不可用。")

st.markdown("---")

# ===================== 因子综合报告 =====================
st.subheader("📋 因子综合报告")

if not factor_summary.empty:
    display_cols = [c for c in ["factor", "ic", "rank_ic", "icir", "ic_abs", "halflife"] if c in factor_summary.columns]
    display_df = factor_summary[display_cols].copy()

    # 格式化
    for col in ["ic", "rank_ic", "icir", "ic_abs"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    if "halflife" in display_df.columns:
        display_df["halflife"] = display_df["halflife"].apply(lambda x: f"{x:.1f} 天")

    st.dataframe(display_df.head(top_n), use_container_width=True)

    # 下载
    csv_bytes = factor_summary.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "⬇️ 下载因子报告 CSV",
        data=csv_bytes,
        file_name=f"factor_summary_{symbol}.csv",
        mime="text/csv",
    )
elif not no_data:
    st.info("因子综合报告不可用。")

st.markdown("---")

# ===================== Alpha 因子说明 =====================
with st.expander("📚 Alpha 因子库说明"):
    st.markdown(
        """
        | 因子名 | 类型 | 说明 |
        |--------|------|------|
        | momentum_5d | 动量 | 5 日价格动量 |
        | momentum_20d | 动量 | 20 日价格动量 |
        | momentum_60d | 动量 | 60 日价格动量 |
        | price_acceleration | 动量 | 价格加速度（短期动量 - 长期动量） |
        | relative_strength | 动量 | 收益率滚动 Z-Score |
        | short_term_reversal | 反转 | 5 日短期反转 |
        | long_term_reversal | 反转 | 长期反转（60日 - 20日动量） |
        | mean_reversion_zscore | 反转 | 均值回归 Z-Score |
        | macd_signal | 技术 | MACD 柱状图（DIF - DEA） |
        | rsi_factor | 技术 | RSI 归一化因子 |
        | kdj_k | 技术 | KDJ 的 K 值 |
        | adx_trend_strength | 技术 | ADX 趋势强度 |
        | hist_volatility_5d | 波动率 | 5 日历史波动率（年化） |
        | hist_volatility_20d | 波动率 | 20 日历史波动率（年化） |
        | hist_volatility_60d | 波动率 | 60 日历史波动率（年化） |
        | volatility_ratio | 波动率 | 短期/长期波动率比率 |
        | downside_volatility | 波动率 | 下行波动率 |
        | volume_price_divergence | 成交量 | 量价背离 |
        | volume_ratio | 成交量 | 量比（当日/N日均量）|
        | volume_trend | 成交量 | 成交量趋势（短期均量/长期均量）|
        | obv_momentum | 成交量 | OBV 动量 |
        | high_low_range | 成交量 | 高低价格范围 |
        """
    )
