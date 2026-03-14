# app/utils/data_loader.py
"""
Streamlit 数据加载工具

负责从 outputs/ 目录加载回测结果数据，并提供缓存支持。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st


OUTPUTS_DIR = Path(__file__).resolve().parent.parent.parent / "outputs"


def _find_outputs_dir() -> Path:
    """智能搜索 outputs/ 目录"""
    candidates = [
        OUTPUTS_DIR,
        Path("outputs"),
        Path("multi_agent_quant/outputs"),
        Path(__file__).resolve().parent.parent.parent / "outputs",
    ]
    for p in candidates:
        if p.exists():
            return p
    return Path("outputs")


@st.cache_data(ttl=60)
def load_comparison_table(outputs_dir: Optional[str] = None) -> pd.DataFrame:
    """
    加载策略对比表格

    Returns
    -------
    pd.DataFrame
        columns: strategy, total_return, annual_return, max_drawdown, sharpe, ...
    """
    base = Path(outputs_dir) if outputs_dir else _find_outputs_dir()
    path = base / "comparison_table.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df


@st.cache_data(ttl=60)
def load_equity_curves(outputs_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    加载所有策略的净值曲线

    Returns
    -------
    dict
        key=策略名, value=DataFrame(date, equity)
    """
    base = Path(outputs_dir) if outputs_dir else _find_outputs_dir()
    curves: Dict[str, pd.DataFrame] = {}
    for csv_file in base.glob("equity_curve_*.csv"):
        strategy_name = csv_file.stem.replace("equity_curve_", "")
        try:
            df = pd.read_csv(csv_file, encoding="utf-8-sig")
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            curves[strategy_name] = df
        except Exception:
            pass
    return curves


@st.cache_data(ttl=60)
def load_factor_ic_scores(symbol: str = "600519", outputs_dir: Optional[str] = None) -> pd.DataFrame:
    """加载因子 IC 评分表"""
    base = Path(outputs_dir) if outputs_dir else _find_outputs_dir()
    path = base / f"factor_ic_scores_{symbol}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_data(ttl=60)
def load_factor_ic_decay(symbol: str = "600519", outputs_dir: Optional[str] = None) -> pd.DataFrame:
    """加载因子 IC 衰减数据"""
    base = Path(outputs_dir) if outputs_dir else _find_outputs_dir()
    path = base / f"factor_ic_decay_{symbol}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_data(ttl=60)
def load_factor_summary(symbol: str = "600519", outputs_dir: Optional[str] = None) -> pd.DataFrame:
    """加载因子综合报告"""
    base = Path(outputs_dir) if outputs_dir else _find_outputs_dir()
    path = base / f"factor_summary_{symbol}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def get_available_symbols(outputs_dir: Optional[str] = None) -> List[str]:
    """获取 outputs/ 目录中有结果的股票代码列表"""
    base = Path(outputs_dir) if outputs_dir else _find_outputs_dir()
    symbols = set()
    for f in base.glob("factor_ic_scores_*.csv"):
        sym = f.stem.replace("factor_ic_scores_", "")
        symbols.add(sym)
    if not symbols:
        symbols.add("600519")  # 默认返回贵州茅台
    return sorted(symbols)
