# utils/factor_evaluator.py
"""
因子评估模块

功能：
- 单因子回测（分层回测）
- 因子表现统计（收益、波动、夏普）
- 因子贡献度分析

使用示例:
    from utils.factor_evaluator import FactorEvaluator
    fe = FactorEvaluator(df, "momentum_20d")
    result = fe.run_quantile_backtest()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class FactorEvaluator:
    """
    单因子评估器（分层回测）

    Parameters
    ----------
    df : pd.DataFrame
        含因子列和 close 列的 DataFrame，须有 DatetimeIndex
    factor_name : str
        要评估的因子列名
    n_quantiles : int
        分层数量（默认 5 层）
    close_col : str
        收盘价列名
    forward_period : int
        前瞻持有周期（天），默认 1
    """

    def __init__(
        self,
        df: pd.DataFrame,
        factor_name: str,
        n_quantiles: int = 5,
        close_col: str = "close",
        forward_period: int = 1,
    ) -> None:
        self.df = df.copy()
        self.factor_name = factor_name
        self.n_quantiles = n_quantiles
        self.close_col = close_col
        self.forward_period = forward_period

        # 计算前瞻收益
        self.df["_fwd_ret"] = (
            self.df[close_col].shift(-forward_period).pct_change(forward_period, fill_method=None)
        )

    def run_quantile_backtest(self) -> Dict[str, Any]:
        """
        分层回测：将因子值分成 N 等分，计算每层的平均前瞻收益

        Returns
        -------
        dict
            keys: quantile_returns (DataFrame), long_short_return (float),
                  long_short_sharpe (float), factor_monotonicity (float)
        """
        valid = self.df[[self.factor_name, "_fwd_ret"]].dropna()
        if len(valid) < self.n_quantiles * 5:
            return {
                "quantile_returns": pd.DataFrame(),
                "long_short_return": 0.0,
                "long_short_sharpe": 0.0,
                "factor_monotonicity": 0.0,
                "error": "insufficient_data",
            }

        # 分层
        valid["quantile"] = pd.qcut(
            valid[self.factor_name], self.n_quantiles, labels=False, duplicates="drop"
        )
        # 各分层平均收益
        quantile_returns = (
            valid.groupby("quantile")["_fwd_ret"].mean().reset_index()
        )
        quantile_returns.columns = ["quantile", "avg_return"]
        quantile_returns["quantile"] = quantile_returns["quantile"] + 1  # 1-based

        # 多空收益（最高分层 - 最低分层）
        if len(quantile_returns) >= 2:
            top_ret = quantile_returns["avg_return"].iloc[-1]
            bottom_ret = quantile_returns["avg_return"].iloc[0]
            ls_ret = float(top_ret - bottom_ret)
        else:
            ls_ret = 0.0

        # 多空夏普（按日计算）：对齐两个分层的日收益序列
        long_ret = valid[valid["quantile"] == valid["quantile"].max()]["_fwd_ret"].reset_index(drop=True)
        short_ret = valid[valid["quantile"] == valid["quantile"].min()]["_fwd_ret"].reset_index(drop=True)
        min_len = min(len(long_ret), len(short_ret))
        if min_len > 2:
            ls_daily = long_ret.values[:min_len] - short_ret.values[:min_len]
            if ls_daily.std() > 0:
                ls_sharpe = float(ls_daily.mean() / ls_daily.std() * np.sqrt(252))
            else:
                ls_sharpe = 0.0
        else:
            ls_sharpe = 0.0

        # 单调性：分层收益与分层排名的 Spearman 相关性
        if len(quantile_returns) >= 3:
            from scipy import stats as sp_stats
            corr, _ = sp_stats.spearmanr(
                quantile_returns["quantile"], quantile_returns["avg_return"]
            )
            monotonicity = float(corr) if not np.isnan(corr) else 0.0
        else:
            monotonicity = 0.0

        return {
            "quantile_returns": quantile_returns,
            "long_short_return": ls_ret,
            "long_short_sharpe": ls_sharpe,
            "factor_monotonicity": monotonicity,
        }

    def compute_turnover(self, top_pct: float = 0.2) -> float:
        """
        计算因子换手率（多头组合的日均换手率）

        Parameters
        ----------
        top_pct : float
            多头持仓比例（取因子值最高的前 top_pct 比例）

        Returns
        -------
        float
            平均换手率（0-1）
        """
        valid = self.df[[self.factor_name]].dropna()
        n_top = max(1, int(len(valid) * top_pct))

        holdings: List[set] = []
        dates = valid.index.tolist()
        for i in range(len(dates)):
            # 取当日截面（单股情况下，每日只有一个值，此处做时序处理）
            # 在单股场景下换手率无意义，返回 NaN
            pass

        # 单股情形下换手率不适用
        return float("nan")

    def get_factor_stats(self) -> Dict[str, float]:
        """
        计算因子值的基本统计信息

        Returns
        -------
        dict
            mean, std, skew, kurt, min, max, pct_positive
        """
        vals = self.df[self.factor_name].dropna()
        if len(vals) == 0:
            return {}
        return {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "skew": float(vals.skew()),
            "kurt": float(vals.kurt()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "pct_positive": float((vals > 0).mean()),
        }

    def compute_cumulative_returns_by_quantile(self) -> pd.DataFrame:
        """
        计算各分层的累计收益曲线

        Returns
        -------
        pd.DataFrame
            index=date, columns=[Q1, Q2, ..., QN, long_short]
        """
        valid = self.df[[self.factor_name, "_fwd_ret"]].dropna().copy()
        if len(valid) < self.n_quantiles * 5:
            return pd.DataFrame()

        valid["quantile"] = pd.qcut(
            valid[self.factor_name], self.n_quantiles, labels=False, duplicates="drop"
        )

        result = pd.DataFrame(index=valid.index)
        for q in range(self.n_quantiles):
            mask = valid["quantile"] == q
            rets = valid.loc[mask, "_fwd_ret"]
            result[f"Q{q+1}"] = rets
        result = result.fillna(0)

        # 累计收益
        cum = (1 + result).cumprod()

        # 多空组合
        if f"Q{self.n_quantiles}" in cum.columns and "Q1" in cum.columns:
            ls_daily = result[f"Q{self.n_quantiles}"] - result["Q1"]
            cum["long_short"] = (1 + ls_daily).cumprod()

        return cum


def evaluate_all_factors(
    df: pd.DataFrame,
    factor_names: List[str],
    close_col: str = "close",
    forward_period: int = 1,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """
    批量评估所有因子的表现

    Parameters
    ----------
    df : pd.DataFrame
    factor_names : list of str
    close_col : str
    forward_period : int
    n_quantiles : int

    Returns
    -------
    pd.DataFrame
        每行对应一个因子，columns=['factor', 'ls_return', 'ls_sharpe', 'monotonicity']
    """
    rows = []
    for factor in factor_names:
        if factor not in df.columns:
            continue
        ev = FactorEvaluator(df, factor, n_quantiles, close_col, forward_period)
        result = ev.run_quantile_backtest()
        stats = ev.get_factor_stats()
        rows.append({
            "factor": factor,
            "ls_return": result.get("long_short_return", 0.0),
            "ls_sharpe": result.get("long_short_sharpe", 0.0),
            "monotonicity": result.get("factor_monotonicity", 0.0),
            "mean": stats.get("mean", 0.0),
            "std": stats.get("std", 0.0),
            "skew": stats.get("skew", 0.0),
        })
    return pd.DataFrame(rows).sort_values("ls_sharpe", ascending=False).reset_index(drop=True)
