# app/pages/04_parameter_tuning.py
"""
参数调优页面 - 交互式调整策略参数并实时回测
"""

from __future__ import annotations

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from agents.data_agent import DataAgent
from agents.trend_agent import TrendAgent
from agents.mean_reversion_agent import MeanReversionAgent
from agents.ml_agent import MLAgent
from agents.coordinator_agent import CoordinatorAgent
from backtest.backtest_engine import BacktestEngine
from utils.metrics import compute_all_metrics
from app.utils.plot_utils import plot_equity_curves, plot_drawdown

st.set_page_config(page_title="参数调优", page_icon="⚙️", layout="wide")
st.title("⚙️ 参数调优")
st.markdown("交互式调整策略参数，点击「运行回测」即可实时查看结果。")
st.markdown("---")

# ===================== 参数面板 =====================
with st.sidebar:
    st.header("🔧 回测参数")

    st.subheader("📅 数据设置")
    symbol = st.text_input("股票代码", value="600519")
    start_date = st.text_input("开始日期 (YYYYMMDD)", value="20210101")
    end_date = st.text_input("结束日期 (YYYYMMDD)", value="20241231")

    st.subheader("💰 资金设置")
    init_cash = st.number_input("初始资金", value=100000, step=10000, min_value=10000)
    fee_rate = st.number_input("佣金率", value=0.0003, format="%.4f", step=0.0001)
    slippage_bps = st.number_input("滑点 (bps)", value=5.0, step=1.0)

    st.subheader("📈 趋势策略参数")
    trend_short = st.slider("短期均线", min_value=3, max_value=20, value=5)
    trend_long = st.slider("长期均线", min_value=10, max_value=60, value=20)

    st.subheader("🔄 均值回归策略参数")
    mr_window = st.slider("布林带窗口", min_value=10, max_value=40, value=20)
    mr_num_std = st.slider("布林带标准差倍数", min_value=0.5, max_value=3.0, value=1.2, step=0.1)

    st.subheader("🤖 策略选择")
    strategy_type = st.selectbox(
        "运行策略",
        options=["Trend-only", "MeanRev-only", "Fusion(Trend+MeanRev)"],
        index=0,
    )

    run_btn = st.button("▶ 运行回测", type="primary", use_container_width=True)

# ===================== 回测逻辑 =====================
if run_btn:
    with st.spinner("⏳ 正在获取数据并运行回测..."):
        t0 = time.time()

        # 加载数据（使用缓存）
        try:
            da = DataAgent(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
                data_dir=str(
                    Path(__file__).resolve().parent.parent.parent / "data" / "raw"
                ),
            )
            df = da.get_feature_data(use_cache=True, add_indicators=True)
            start_dt = pd.to_datetime(start_date, format="%Y%m%d")
            end_dt = pd.to_datetime(end_date, format="%Y%m%d")
            df = df.sort_index().loc[start_dt:end_dt]
        except Exception as e:
            st.error(f"❌ 数据加载失败：{e}")
            st.stop()

        if len(df) < 30:
            st.error("❌ 数据行数不足，请检查股票代码和日期范围。")
            st.stop()

        # 初始化引擎
        engine = BacktestEngine(
            init_cash=float(init_cash),
            fee_rate=fee_rate,
            slippage_bps=slippage_bps,
        )

        # 运行选定策略
        try:
            if strategy_type == "Trend-only":
                agent = TrendAgent(trend_short, trend_long)
                metrics, trades = engine.run_single_agent(df, agent, "Trend-only")
            elif strategy_type == "MeanRev-only":
                agent = MeanReversionAgent(mr_window, mr_num_std)
                metrics, trades = engine.run_single_agent(df, agent, "MeanRev-only")
            else:
                trend_agent = TrendAgent(trend_short, trend_long)
                mr_agent = MeanReversionAgent(mr_window, mr_num_std)
                coord = CoordinatorAgent(
                    agent_weights={"trend": 0.6, "mean_reversion": 0.4},
                    ml_veto_enabled=False,
                )
                metrics, trades = engine.run_fusion(
                    df,
                    {"trend": trend_agent, "mean_reversion": mr_agent},
                    coord,
                    "Fusion(Trend+MeanRev)",
                )
        except Exception as e:
            st.error(f"❌ 回测失败：{e}")
            st.stop()

        elapsed = time.time() - t0

    # ===================== 展示结果 =====================
    st.success(f"✅ 回测完成（耗时 {elapsed:.2f}s）")

    # KPI
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("策略", strategy_type)
    col2.metric("总收益率", f"{metrics.get('total_return', 0) * 100:.2f}%")
    col3.metric("年化收益率", f"{metrics.get('annual_return', 0) * 100:.2f}%")
    col4.metric("最大回撤", f"{metrics.get('max_drawdown', 0) * 100:.2f}%")
    col5.metric("夏普比率", f"{metrics.get('sharpe', 0):.3f}")

    col6, col7, col8 = st.columns(3)
    extra = compute_all_metrics(metrics.get("equity_curve", []), trades)
    col6.metric("胜率", f"{extra.get('win_rate', 0) * 100:.2f}%")
    col7.metric("Calmar 比率", f"{extra.get('calmar', 0):.3f}")
    col8.metric("交易次数", str(metrics.get("num_trades", 0)))

    st.markdown("---")

    # 净值曲线
    if metrics.get("equity_curve") and metrics.get("dates"):
        curve_df = pd.DataFrame({
            "date": metrics["dates"],
            "equity": metrics["equity_curve"],
        })
        curves = {strategy_type: curve_df}

        st.subheader("📈 净值曲线")
        fig_nav = plot_equity_curves(curves, title=f"{strategy_type} 净值曲线", height=450)
        st.plotly_chart(fig_nav, use_container_width=True)

        st.subheader("📉 回撤曲线")
        fig_dd = plot_drawdown(curves, title="回撤曲线", height=300)
        st.plotly_chart(fig_dd, use_container_width=True)

        # 下载净值曲线
        csv_bytes = curve_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "⬇️ 下载净值曲线 CSV",
            data=csv_bytes,
            file_name=f"equity_curve_{strategy_type}_{symbol}.csv",
            mime="text/csv",
        )
else:
    st.info("👈 请在左侧设置参数，然后点击「运行回测」。")
