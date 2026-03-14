# utils/factor_selector.py
"""
因子筛选与评估模块

功能：
- IC（信息系数）计算：因子值与未来收益的相关性
- Rank IC 计算：基于排名的 IC，更鲁棒
- 因子衰减分析：IC 随持有期延长的衰减系数
- 自动因子筛选：选择最优因子组合
- 因子相关性分析：避免因子过度相关

使用示例:
    from utils.factor_selector import FactorSelector
    fs = FactorSelector(df_with_factors, factor_names)
    scores = fs.compute_ic_scores()
    best = fs.select_top_factors(n=5)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


class FactorSelector:
    """
    因子筛选器

    Parameters
    ----------
    df : pd.DataFrame
        含因子列和 close 列的 DataFrame，必须有 DatetimeIndex
    factor_names : list of str
        要分析的因子列名列表
    forward_periods : list of int
        前瞻收益周期（天数），默认 [1, 5, 10, 20]
    """

    def __init__(
        self,
        df: pd.DataFrame,
        factor_names: List[str],
        forward_periods: Optional[List[int]] = None,
        close_col: str = "close",
    ) -> None:
        self.df = df.copy()
        self.factor_names = [f for f in factor_names if f in df.columns]
        self.forward_periods = forward_periods or [1, 5, 10, 20]
        self.close_col = close_col

        # 计算各周期前瞻收益
        self._fwd_returns: Dict[int, pd.Series] = {}
        for period in self.forward_periods:
            fwd = (
                self.df[close_col]
                .shift(-period)
                .pct_change(period, fill_method=None)
            )
            self._fwd_returns[period] = fwd

    # =========================================================
    # IC 计算
    # =========================================================

    def compute_ic(
        self, factor: str, period: int = 5
    ) -> pd.Series:
        """
        计算单因子的滚动 IC（Pearson 相关系数）

        Parameters
        ----------
        factor : str
            因子列名
        period : int
            前瞻收益周期

        Returns
        -------
        pd.Series
            每日 IC 值（单次 bar 的因子值与 period 日后收益的相关性）
        """
        if factor not in self.df.columns:
            raise ValueError(f"因子 {factor!r} 不在 DataFrame 中")
        if period not in self._fwd_returns:
            fwd = self.df[self.close_col].shift(-period).pct_change(period, fill_method=None)
            self._fwd_returns[period] = fwd

        factor_vals = self.df[factor]
        fwd_ret = self._fwd_returns[period]

        # 按滚动窗口计算横截面 IC（此处为时序 IC，适用单股分析）
        combined = pd.DataFrame({"factor": factor_vals, "fwd_ret": fwd_ret}).dropna()
        if len(combined) < 10:
            return pd.Series(dtype=float)

        # 滚动 60 日相关系数
        ic = combined["factor"].rolling(60).corr(combined["fwd_ret"])
        return ic

    def compute_mean_ic(
        self, factor: str, period: int = 5
    ) -> float:
        """
        计算因子的平均 IC（整体相关系数）

        Returns
        -------
        float
            Pearson 相关系数（IC），范围 [-1, 1]
        """
        if factor not in self.df.columns:
            return 0.0
        if period not in self._fwd_returns:
            fwd = self.df[self.close_col].shift(-period).pct_change(period, fill_method=None)
            self._fwd_returns[period] = fwd

        factor_vals = self.df[factor]
        fwd_ret = self._fwd_returns[period]
        combined = pd.DataFrame({"factor": factor_vals, "fwd_ret": fwd_ret}).dropna()
        if len(combined) < 10:
            return 0.0
        corr, _ = stats.pearsonr(combined["factor"], combined["fwd_ret"])
        return float(corr) if not np.isnan(corr) else 0.0

    def compute_rank_ic(
        self, factor: str, period: int = 5
    ) -> float:
        """
        计算因子的 Rank IC（Spearman 秩相关系数）
        比 Pearson IC 更鲁棒，不受极值影响

        Returns
        -------
        float
            Spearman 相关系数（Rank IC）
        """
        if factor not in self.df.columns:
            return 0.0
        if period not in self._fwd_returns:
            fwd = self.df[self.close_col].shift(-period).pct_change(period, fill_method=None)
            self._fwd_returns[period] = fwd

        factor_vals = self.df[factor]
        fwd_ret = self._fwd_returns[period]
        combined = pd.DataFrame({"factor": factor_vals, "fwd_ret": fwd_ret}).dropna()
        if len(combined) < 10:
            return 0.0
        corr, _ = stats.spearmanr(combined["factor"], combined["fwd_ret"])
        return float(corr) if not np.isnan(corr) else 0.0

    def compute_icir(
        self, factor: str, period: int = 5, window: int = 60
    ) -> float:
        """
        ICIR（IC 信息比率）= IC 均值 / IC 标准差
        衡量因子预测能力的稳定性，越高越好（> 0.5 通常认为稳定）

        Returns
        -------
        float
            ICIR 值
        """
        ic_series = self.compute_ic(factor, period).dropna()
        if len(ic_series) < 5:
            return 0.0
        ic_std = ic_series.std()
        if ic_std == 0:
            return 0.0
        return float(ic_series.mean() / ic_std)

    # =========================================================
    # 衰减分析
    # =========================================================

    def compute_ic_decay(
        self, factor: str, max_period: int = 20
    ) -> pd.DataFrame:
        """
        计算因子 IC 随持有期的衰减情况

        Parameters
        ----------
        factor : str
            因子列名
        max_period : int
            最长分析周期（天）

        Returns
        -------
        pd.DataFrame
            columns=['period', 'ic', 'rank_ic']
        """
        results = []
        for period in range(1, max_period + 1):
            ic = self.compute_mean_ic(factor, period)
            rank_ic = self.compute_rank_ic(factor, period)
            results.append({"period": period, "ic": ic, "rank_ic": rank_ic})
        return pd.DataFrame(results)

    def compute_decay_halflife(self, factor: str, max_period: int = 20) -> float:
        """
        估算因子 IC 的衰减半衰期（期数）

        Returns
        -------
        float
            IC 降至最大值 50% 所需的持有期天数，无法估算时返回 max_period
        """
        decay_df = self.compute_ic_decay(factor, max_period)
        ic_abs = decay_df["ic"].abs()
        if ic_abs.max() == 0:
            return float(max_period)
        half_val = ic_abs.max() * 0.5
        below_half = decay_df[ic_abs <= half_val]
        if len(below_half) == 0:
            return float(max_period)
        return float(below_half["period"].iloc[0])

    # =========================================================
    # 批量评分
    # =========================================================

    def compute_ic_scores(
        self, period: int = 5
    ) -> pd.DataFrame:
        """
        批量计算所有因子的 IC、Rank IC 和 ICIR

        Returns
        -------
        pd.DataFrame
            columns=['factor', 'ic', 'rank_ic', 'icir', 'ic_abs']
            按 |IC| 降序排列
        """
        rows = []
        for factor in self.factor_names:
            ic = self.compute_mean_ic(factor, period)
            rank_ic = self.compute_rank_ic(factor, period)
            icir = self.compute_icir(factor, period)
            rows.append({
                "factor": factor,
                "ic": ic,
                "rank_ic": rank_ic,
                "icir": icir,
                "ic_abs": abs(ic),
            })
        result = pd.DataFrame(rows).sort_values("ic_abs", ascending=False).reset_index(drop=True)
        return result

    # =========================================================
    # 相关性分析
    # =========================================================

    def compute_factor_correlation(self) -> pd.DataFrame:
        """
        计算因子间的相关性矩阵

        Returns
        -------
        pd.DataFrame
            因子相关性矩阵
        """
        factor_df = self.df[self.factor_names].dropna()
        if len(factor_df) < 5:
            return pd.DataFrame()
        return factor_df.corr()

    def get_low_correlation_factors(
        self, threshold: float = 0.7
    ) -> List[str]:
        """
        筛选相关性低于阈值的因子子集（贪心算法）
        避免同质化因子造成过度拟合

        Parameters
        ----------
        threshold : float
            相关性绝对值上限，超过则认为高度相关（默认 0.7）

        Returns
        -------
        list of str
            低相关性因子列表
        """
        corr_matrix = self.compute_factor_correlation()
        if corr_matrix.empty:
            return self.factor_names

        selected: List[str] = []
        for factor in self.factor_names:
            if factor not in corr_matrix.columns:
                continue
            is_redundant = False
            for sel in selected:
                if sel in corr_matrix.columns and factor in corr_matrix.index:
                    if abs(corr_matrix.loc[factor, sel]) >= threshold:
                        is_redundant = True
                        break
            if not is_redundant:
                selected.append(factor)
        return selected

    # =========================================================
    # 自动筛选
    # =========================================================

    def select_top_factors(
        self,
        n: int = 5,
        period: int = 5,
        min_ic_abs: float = 0.02,
        corr_threshold: float = 0.7,
    ) -> List[str]:
        """
        自动筛选最优因子组合

        算法：
        1. 计算所有因子的 IC/Rank IC
        2. 过滤 |IC| 低于 min_ic_abs 的因子
        3. 按 |IC| 降序排列
        4. 依次选择（排除与已选因子高度相关的）
        5. 返回前 n 个

        Parameters
        ----------
        n : int
            期望选出的因子数量
        period : int
            前瞻收益周期
        min_ic_abs : float
            最小有效 |IC| 阈值
        corr_threshold : float
            因子相关性过滤阈值

        Returns
        -------
        list of str
            最优因子列表
        """
        scores = self.compute_ic_scores(period)
        # 过滤低 IC 因子
        valid = scores[scores["ic_abs"] >= min_ic_abs]["factor"].tolist()
        if not valid:
            # 如果全部因子 IC 太低，直接返回 IC 最高的 n 个
            return scores.head(n)["factor"].tolist()

        # 去相关性
        low_corr = self.get_low_correlation_factors(corr_threshold)
        filtered = [f for f in valid if f in low_corr]
        if not filtered:
            filtered = valid

        return filtered[:n]

    def get_factor_summary(self, period: int = 5) -> pd.DataFrame:
        """
        获取因子综合评分表，含 IC、Rank IC、ICIR、半衰期

        Returns
        -------
        pd.DataFrame
        """
        scores = self.compute_ic_scores(period)
        halflives = []
        for factor in scores["factor"]:
            hl = self.compute_decay_halflife(factor, max_period=20)
            halflives.append(hl)
        scores["halflife"] = halflives
        return scores
