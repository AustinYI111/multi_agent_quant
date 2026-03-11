# utils/indicators.py
"""
统一技术指标计算模块
消除各 Agent 和回测引擎中重复的指标计算逻辑
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_ma(series: pd.Series, window: int) -> pd.Series:
    """计算简单移动平均"""
    return series.rolling(window).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """计算 RSI（相对强弱指数）"""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0):
    """
    计算布林带

    Returns
    -------
    tuple: (upper, middle, lower)
    """
    middle = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def compute_volatility(series: pd.Series, window: int = 20) -> pd.Series:
    """计算滚动历史波动率（收益率标准差）"""
    rets = series.pct_change()
    return rets.rolling(window).std()


def add_all_indicators(df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
    """
    一次性添加所有常用技术指标到 DataFrame

    添加列：
    - return: 日收益率
    - ma_5, ma_10, ma_20: 移动平均线
    - vol_20: 20日波动率
    - rsi: RSI(14)
    - boll_upper, boll_mid, boll_lower: 布林带(20,2)
    """
    df = df.copy()
    close = pd.to_numeric(df[close_col], errors="coerce")

    if "return" not in df.columns:
        df["return"] = close.pct_change()

    if "ma_5" not in df.columns:
        df["ma_5"] = compute_ma(close, 5)
    if "ma_10" not in df.columns:
        df["ma_10"] = compute_ma(close, 10)
    if "ma_20" not in df.columns:
        df["ma_20"] = compute_ma(close, 20)

    if "vol_20" not in df.columns:
        df["vol_20"] = compute_volatility(close, 20)

    if "rsi" not in df.columns:
        df["rsi"] = compute_rsi(close, 14)

    if "boll_upper" not in df.columns or "boll_lower" not in df.columns:
        upper, mid, lower = compute_bollinger(close, 20, 2.0)
        df["boll_upper"] = upper
        df["boll_mid"] = mid
        df["boll_lower"] = lower

    return df
