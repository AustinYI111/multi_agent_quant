# utils/alpha_factors.py
"""
Alpha 因子库 - 包含 15+ 个高质量 Alpha 因子

因子分类：
- 动量因子 (Momentum): 多周期价格动量
- 反转因子 (Reversion): 短期和长期价格反转
- 成交量因子 (Volume): 量价关系
- 技术面因子 (Technical): OBV、MACD、ADX 等
- 波动率因子 (Volatility): 多周期历史波动率

使用示例:
    import pandas as pd
    from utils.alpha_factors import AlphaFactors

    df = pd.read_csv("data.csv")
    af = AlphaFactors(df)
    result = af.compute_all()
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class AlphaFactors:
    """
    Alpha 因子计算器

    Parameters
    ----------
    df : pd.DataFrame
        必须包含 close 列；如需量价因子，还需 open/high/low/volume 列
    close_col : str
        收盘价列名，默认 "close"
    """

    def __init__(
        self,
        df: pd.DataFrame,
        close_col: str = "close",
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        volume_col: str = "volume",
    ) -> None:
        self.df = df.copy()
        self.close_col = close_col
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.volume_col = volume_col

        # 统一转为 float 类型，避免类型错误
        for col in [close_col, open_col, high_col, low_col, volume_col]:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        self._close = self.df[close_col]
        self._has_ohlv = all(
            c in self.df.columns for c in [open_col, high_col, low_col, volume_col]
        )

    # =========================================================
    # 动量因子 (Momentum Factors)
    # =========================================================

    def momentum_5d(self) -> pd.Series:
        """5 日动量：当日收盘价相对 5 日前涨跌幅"""
        return self._close.pct_change(5).rename("momentum_5d")

    def momentum_20d(self) -> pd.Series:
        """20 日动量：当日收盘价相对 20 日前涨跌幅"""
        return self._close.pct_change(20).rename("momentum_20d")

    def momentum_60d(self) -> pd.Series:
        """60 日动量：当日收盘价相对 60 日前涨跌幅"""
        return self._close.pct_change(60).rename("momentum_60d")

    def price_acceleration(self, short: int = 5, long: int = 20) -> pd.Series:
        """
        价格加速度：短期动量 - 长期动量
        正值代表动量加速（趋势增强），负值代表动量减速（趋势衰减）
        """
        mom_short = self._close.pct_change(short)
        mom_long = self._close.pct_change(long)
        return (mom_short - mom_long).rename("price_acceleration")

    def relative_strength(self, window: int = 20) -> pd.Series:
        """
        相对强度：与市场自身均值的偏差（简化版，不依赖外部基准）
        用滚动 Z-score 衡量价格动量的相对强度
        """
        ret = self._close.pct_change()
        rolling_mean = ret.rolling(window).mean()
        rolling_std = ret.rolling(window).std()
        rs = (ret - rolling_mean) / rolling_std.replace(0, np.nan)
        return rs.rename("relative_strength")

    # =========================================================
    # 反转因子 (Reversion Factors)
    # =========================================================

    def short_term_reversal(self, window: int = 5) -> pd.Series:
        """
        短期反转因子：过去 N 日累计收益的负值
        收益越高的股票，未来往往出现均值回归（反转）
        """
        ret = self._close.pct_change(window)
        return (-ret).rename("short_term_reversal")

    def long_term_reversal(self, short_win: int = 20, long_win: int = 60) -> pd.Series:
        """
        长期反转因子：长期动量扣除短期动量
        识别中长期超涨/超跌后的回归机会
        """
        mom_long = self._close.pct_change(long_win)
        mom_short = self._close.pct_change(short_win)
        return (mom_long - mom_short).rename("long_term_reversal")

    def mean_reversion_zscore(self, window: int = 20) -> pd.Series:
        """
        均值回归 Z-Score：价格偏离滚动均值的标准差倍数
        用于识别超买/超卖状态
        """
        rolling_mean = self._close.rolling(window).mean()
        rolling_std = self._close.rolling(window).std()
        z = (self._close - rolling_mean) / rolling_std.replace(0, np.nan)
        return (-z).rename("mean_reversion_zscore")  # 负 Z-Score 作为反转信号

    # =========================================================
    # 成交量因子 (Volume Factors)
    # =========================================================

    def volume_price_divergence(self, window: int = 10) -> pd.Series:
        """
        量价背离因子：价格动量与成交量动量的差值
        量价背离（价涨量缩 or 价跌量增）往往预示反转
        """
        if not self._has_ohlv:
            return pd.Series(np.nan, index=self.df.index, name="volume_price_divergence")
        vol = self.df[self.volume_col]
        price_mom = self._close.pct_change(window)
        vol_mom = vol.pct_change(window)
        return (vol_mom - price_mom).rename("volume_price_divergence")

    def volume_ratio(self, window: int = 20) -> pd.Series:
        """
        量比（成交量比）：当日成交量 / 近 N 日平均成交量
        量比 > 1 代表放量，< 1 代表缩量
        """
        if not self._has_ohlv:
            return pd.Series(np.nan, index=self.df.index, name="volume_ratio")
        vol = self.df[self.volume_col]
        avg_vol = vol.rolling(window).mean()
        return (vol / avg_vol.replace(0, np.nan)).rename("volume_ratio")

    def volume_trend(self, short: int = 5, long: int = 20) -> pd.Series:
        """
        成交量趋势：短期均量 / 长期均量
        > 1 表示近期放量（量能扩张），< 1 表示近期缩量
        """
        if not self._has_ohlv:
            return pd.Series(np.nan, index=self.df.index, name="volume_trend")
        vol = self.df[self.volume_col]
        short_avg = vol.rolling(short).mean()
        long_avg = vol.rolling(long).mean()
        return (short_avg / long_avg.replace(0, np.nan)).rename("volume_trend")

    def turnover_adjusted_momentum(self, mom_window: int = 20, vol_window: int = 20) -> pd.Series:
        """
        换手率调整动量：价格动量 / 成交量动量
        筛选出有效的、成交活跃支撑的价格上涨
        """
        if not self._has_ohlv:
            return pd.Series(np.nan, index=self.df.index, name="turnover_adjusted_momentum")
        price_mom = self._close.pct_change(mom_window)
        vol = self.df[self.volume_col]
        vol_mom = (vol.pct_change(vol_window) + 1).replace(0, np.nan)
        return (price_mom / vol_mom).rename("turnover_adjusted_momentum")

    # =========================================================
    # 技术面因子 (Technical Factors)
    # =========================================================

    def obv(self) -> pd.Series:
        """
        OBV（能量潮，On Balance Volume）
        量价关系的累积指标，判断资金流向
        """
        if not self._has_ohlv:
            return pd.Series(np.nan, index=self.df.index, name="obv")
        close = self._close
        vol = self.df[self.volume_col]
        direction = np.sign(close.diff())
        direction.iloc[0] = 0
        obv_vals = (direction * vol).cumsum()
        return obv_vals.rename("obv")

    def obv_momentum(self, window: int = 20) -> pd.Series:
        """OBV 动量：OBV 的 N 日变化率"""
        obv_vals = self.obv()
        return obv_vals.pct_change(window, fill_method=None).rename("obv_momentum")

    def macd_signal(self, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """
        MACD 信号：MACD 线与信号线的差值（柱状图 Histogram）
        正值代表多头动能，负值代表空头动能
        """
        ema_fast = self._close.ewm(span=fast, adjust=False).mean()
        ema_slow = self._close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return (macd_line - signal_line).rename("macd_signal")

    def rsi_factor(self, period: int = 14) -> pd.Series:
        """
        RSI 因子：将 RSI 转化为 [-1, 1] 区间的因子值
        RSI > 70 超买（负值），RSI < 30 超卖（正值）
        """
        delta = self._close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        # 归一化：RSI=50 → 0，RSI=100 → -1，RSI=0 → +1
        return ((50 - rsi) / 50).rename("rsi_factor")

    def adx_trend_strength(self, period: int = 14) -> pd.Series:
        """
        ADX（平均趋向指数）：衡量趋势强度（0-100）
        ADX > 25 表示强趋势，> 50 表示极强趋势
        """
        if not self._has_ohlv:
            return pd.Series(np.nan, index=self.df.index, name="adx_trend_strength")
        high = self.df[self.high_col]
        low = self.df[self.low_col]
        close = self._close

        # 真实波幅 (True Range)
        hl = high - low
        hc = (high - close.shift(1)).abs()
        lc = (low - close.shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)

        # 方向性运动
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        # ATR 平滑
        atr = tr.rolling(period).mean()
        plus_di = 100 * plus_dm.rolling(period).mean() / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm.rolling(period).mean() / atr.replace(0, np.nan)

        # ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.rolling(period).mean()
        return adx.rename("adx_trend_strength")

    def kdj_k(self, period: int = 9) -> pd.Series:
        """
        KDJ 指标中的 K 值
        K 值 > 80 超买区，K 值 < 20 超卖区
        """
        if not self._has_ohlv:
            return pd.Series(np.nan, index=self.df.index, name="kdj_k")
        high = self.df[self.high_col]
        low = self.df[self.low_col]
        close = self._close

        lowest_low = low.rolling(period).min()
        highest_high = high.rolling(period).max()
        rsv = (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan) * 100

        k = rsv.ewm(com=2, adjust=False).mean()
        return k.rename("kdj_k")

    # =========================================================
    # 波动率因子 (Volatility Factors)
    # =========================================================

    def hist_volatility_5d(self) -> pd.Series:
        """5 日历史波动率（年化）"""
        ret = self._close.pct_change()
        return (ret.rolling(5).std() * np.sqrt(252)).rename("hist_volatility_5d")

    def hist_volatility_20d(self) -> pd.Series:
        """20 日历史波动率（年化）"""
        ret = self._close.pct_change()
        return (ret.rolling(20).std() * np.sqrt(252)).rename("hist_volatility_20d")

    def hist_volatility_60d(self) -> pd.Series:
        """60 日历史波动率（年化）"""
        ret = self._close.pct_change()
        return (ret.rolling(60).std() * np.sqrt(252)).rename("hist_volatility_60d")

    def volatility_ratio(self, short: int = 5, long: int = 20) -> pd.Series:
        """
        波动率比率：短期波动率 / 长期波动率
        > 1 代表近期波动放大（市场活跃），< 1 代表近期波动收缩（市场平静）
        """
        ret = self._close.pct_change()
        vol_short = ret.rolling(short).std()
        vol_long = ret.rolling(long).std()
        return (vol_short / vol_long.replace(0, np.nan)).rename("volatility_ratio")

    def downside_volatility(self, window: int = 20) -> pd.Series:
        """
        下行波动率：只计算负收益的标准差（年化）
        用于衡量下行风险，更贴近实际风控需求
        """
        ret = self._close.pct_change()
        downside = ret.where(ret < 0, 0.0)
        return (downside.rolling(window).std() * np.sqrt(252)).rename("downside_volatility")

    def high_low_range(self, window: int = 20) -> pd.Series:
        """
        高低价格范围因子：N 日内（最高价 - 最低价）/ 均价
        衡量价格波幅，作为波动率的代理指标
        """
        if not self._has_ohlv:
            return pd.Series(np.nan, index=self.df.index, name="high_low_range")
        high = self.df[self.high_col]
        low = self.df[self.low_col]
        rolling_high = high.rolling(window).max()
        rolling_low = low.rolling(window).min()
        avg_price = self._close.rolling(window).mean()
        return ((rolling_high - rolling_low) / avg_price.replace(0, np.nan)).rename("high_low_range")

    # =========================================================
    # 组合计算
    # =========================================================

    def compute_all(self, include_volume: bool = True) -> pd.DataFrame:
        """
        计算所有因子，返回包含所有因子列的 DataFrame

        Parameters
        ----------
        include_volume : bool
            是否包含需要成交量数据的因子（默认 True）

        Returns
        -------
        pd.DataFrame
            原始 df 加上所有因子列
        """
        result = self.df.copy()

        # 动量因子
        result["momentum_5d"] = self.momentum_5d()
        result["momentum_20d"] = self.momentum_20d()
        result["momentum_60d"] = self.momentum_60d()
        result["price_acceleration"] = self.price_acceleration()
        result["relative_strength"] = self.relative_strength()

        # 反转因子
        result["short_term_reversal"] = self.short_term_reversal()
        result["long_term_reversal"] = self.long_term_reversal()
        result["mean_reversion_zscore"] = self.mean_reversion_zscore()

        # 技术面因子
        result["macd_signal"] = self.macd_signal()
        result["rsi_factor"] = self.rsi_factor()
        result["kdj_k"] = self.kdj_k()
        result["hist_volatility_5d"] = self.hist_volatility_5d()
        result["hist_volatility_20d"] = self.hist_volatility_20d()
        result["hist_volatility_60d"] = self.hist_volatility_60d()
        result["volatility_ratio"] = self.volatility_ratio()
        result["downside_volatility"] = self.downside_volatility()

        if include_volume and self._has_ohlv:
            result["volume_price_divergence"] = self.volume_price_divergence()
            result["volume_ratio"] = self.volume_ratio()
            result["volume_trend"] = self.volume_trend()
            result["turnover_adjusted_momentum"] = self.turnover_adjusted_momentum()
            result["obv_momentum"] = self.obv_momentum()
            result["adx_trend_strength"] = self.adx_trend_strength()
            result["high_low_range"] = self.high_low_range()

        return result

    @staticmethod
    def get_factor_names(include_volume: bool = True) -> List[str]:
        """返回所有因子名称列表"""
        base = [
            "momentum_5d",
            "momentum_20d",
            "momentum_60d",
            "price_acceleration",
            "relative_strength",
            "short_term_reversal",
            "long_term_reversal",
            "mean_reversion_zscore",
            "macd_signal",
            "rsi_factor",
            "kdj_k",
            "hist_volatility_5d",
            "hist_volatility_20d",
            "hist_volatility_60d",
            "volatility_ratio",
            "downside_volatility",
        ]
        if include_volume:
            base += [
                "volume_price_divergence",
                "volume_ratio",
                "volume_trend",
                "turnover_adjusted_momentum",
                "obv_momentum",
                "adx_trend_strength",
                "high_low_range",
            ]
        return base
