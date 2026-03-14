# app/streamlit_app.py
"""
Multi-Agent 量化交易系统 - 可视化主界面

启动方式:
    streamlit run app/streamlit_app.py

页面:
    - 仪表板 (Dashboard)
    - 资金曲线 (Equity Curve)
    - 策略对比 (Strategy Comparison)
    - 参数调优 (Parameter Tuning)
    - 因子分析 (Factor Analysis)
"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保包路径可被找到
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import streamlit as st

st.set_page_config(
    page_title="Multi-Agent 量化交易系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# 主页内容
# =========================================================
st.title("📈 Multi-Agent 量化交易系统")
st.markdown(
    """
    欢迎使用 **Multi-Agent 量化交易可视化平台**！

    本系统集成了三种量化策略 Agent（趋势跟踪、均值回归、机器学习）和先进的 Alpha 因子库，
    支持交互式回测与因子分析。

    ---
    """
)

col1, col2, col3 = st.columns(3)

with col1:
    st.info(
        """
        ### 📊 仪表板
        - 策略回测总览
        - 关键绩效指标
        - 策略对比表格
        """,
        icon="📊",
    )

with col2:
    st.info(
        """
        ### 🔬 因子分析
        - 15+ Alpha 因子
        - IC / Rank IC 评分
        - 因子衰减分析
        """,
        icon="🔬",
    )

with col3:
    st.info(
        """
        ### ⚙️ 参数调优
        - 交互式参数调整
        - 实时回测反馈
        - 结果 CSV 导出
        """,
        icon="⚙️",
    )

st.markdown("---")
st.markdown(
    """
    **👈 请从左侧导航栏选择功能页面**

    | 页面 | 说明 |
    |------|------|
    | 📊 仪表板 | 总体回测指标、策略对比 |
    | 📈 资金曲线 | 多策略净值走势、回撤展示 |
    | 🔄 策略对比 | 详细指标对比（夏普、Calmar、胜率等）|
    | ⚙️ 参数调优 | 交互式调整策略参数并实时回测 |
    | 📉 因子分析 | 因子 IC 分布、衰减曲线、最优因子筛选 |

    ---
    > **数据来源**：通过 akshare 获取 A 股日线数据，支持前复权处理。  
    > **使用前**：请先运行 `python -m multi_agent_quant.experiments.run_backtest` 生成回测结果。
    """
)
