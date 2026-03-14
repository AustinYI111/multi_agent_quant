# tests/test_alpha_factors.py
"""
Alpha 因子库单元测试

测试 AlphaFactors、FactorSelector、FactorEvaluator 的核心功能
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.alpha_factors import AlphaFactors
from utils.factor_selector import FactorSelector
from utils.factor_evaluator import FactorEvaluator, evaluate_all_factors


# =========================================================
# 测试用数据生成
# =========================================================

def _make_ohlcv_df(n: int = 120) -> pd.DataFrame:
    """生成模拟 OHLCV 数据"""
    np.random.seed(42)
    close = np.cumprod(1 + np.random.normal(0.0005, 0.015, n)) * 100.0
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_ = close * (1 + np.random.normal(0, 0.003, n))
    volume = np.random.randint(1_000_000, 5_000_000, n).astype(float)

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    df.index = pd.date_range("2022-01-01", periods=n, freq="B")
    return df


def _make_close_only_df(n: int = 100) -> pd.DataFrame:
    """生成只有 close 的数据"""
    np.random.seed(0)
    close = np.cumprod(1 + np.random.normal(0.0003, 0.012, n)) * 50.0
    df = pd.DataFrame({"close": close})
    df.index = pd.date_range("2022-01-01", periods=n, freq="B")
    return df


# =========================================================
# AlphaFactors 测试
# =========================================================

class TestAlphaFactors:
    def test_momentum_5d(self):
        df = _make_ohlcv_df()
        af = AlphaFactors(df)
        result = af.momentum_5d()
        assert isinstance(result, pd.Series)
        assert result.name == "momentum_5d"
        # 前 5 行应为 NaN（pct_change(5) 需要 5 行）
        assert result.iloc[:5].isna().all()
        # 其余大部分非 NaN
        assert result.iloc[5:].notna().sum() > 50

    def test_momentum_20d(self):
        df = _make_ohlcv_df()
        af = AlphaFactors(df)
        result = af.momentum_20d()
        assert isinstance(result, pd.Series)
        assert result.name == "momentum_20d"

    def test_momentum_60d(self):
        df = _make_ohlcv_df()
        af = AlphaFactors(df)
        result = af.momentum_60d()
        assert isinstance(result, pd.Series)
        assert result.name == "momentum_60d"

    def test_short_term_reversal(self):
        df = _make_ohlcv_df()
        af = AlphaFactors(df)
        result = af.short_term_reversal()
        assert isinstance(result, pd.Series)
        assert result.name == "short_term_reversal"
        # 反转因子是动量的负值
        mom = af.momentum_5d()
        pd.testing.assert_series_equal(result, -mom.rename("short_term_reversal"))

    def test_mean_reversion_zscore(self):
        df = _make_ohlcv_df()
        af = AlphaFactors(df)
        result = af.mean_reversion_zscore()
        assert isinstance(result, pd.Series)
        assert result.name == "mean_reversion_zscore"

    def test_macd_signal(self):
        df = _make_ohlcv_df()
        af = AlphaFactors(df)
        result = af.macd_signal()
        assert isinstance(result, pd.Series)
        assert result.name == "macd_signal"

    def test_rsi_factor(self):
        df = _make_ohlcv_df()
        af = AlphaFactors(df)
        result = af.rsi_factor()
        assert isinstance(result, pd.Series)
        assert result.name == "rsi_factor"
        valid = result.dropna()
        # RSI 因子归一化到 [-1, 1]
        assert (valid >= -1).all() and (valid <= 1).all()

    def test_hist_volatility_20d(self):
        df = _make_ohlcv_df()
        af = AlphaFactors(df)
        result = af.hist_volatility_20d()
        assert isinstance(result, pd.Series)
        assert result.name == "hist_volatility_20d"
        valid = result.dropna()
        # 波动率应为非负
        assert (valid >= 0).all()

    def test_volatility_ratio(self):
        df = _make_ohlcv_df()
        af = AlphaFactors(df)
        result = af.volatility_ratio()
        assert isinstance(result, pd.Series)
        assert result.name == "volatility_ratio"

    def test_volume_factors_with_ohlcv(self):
        df = _make_ohlcv_df()
        af = AlphaFactors(df)
        assert af._has_ohlv

        vol_ratio = af.volume_ratio()
        assert vol_ratio.name == "volume_ratio"
        assert vol_ratio.dropna().notna().sum() > 50

        vol_trend = af.volume_trend()
        assert vol_trend.name == "volume_trend"

        obv_mom = af.obv_momentum()
        assert obv_mom.name == "obv_momentum"

    def test_volume_factors_without_volume_return_nan(self):
        df = _make_close_only_df()
        af = AlphaFactors(df)
        assert not af._has_ohlv

        vol_ratio = af.volume_ratio()
        assert vol_ratio.isna().all()

        obv_mom = af.obv_momentum()
        assert obv_mom.isna().all()

    def test_adx_trend_strength(self):
        df = _make_ohlcv_df()
        af = AlphaFactors(df)
        result = af.adx_trend_strength()
        assert result.name == "adx_trend_strength"
        valid = result.dropna()
        # ADX 取值范围 [0, 100]
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_kdj_k(self):
        df = _make_ohlcv_df()
        af = AlphaFactors(df)
        result = af.kdj_k()
        assert result.name == "kdj_k"

    def test_compute_all_with_volume(self):
        df = _make_ohlcv_df()
        af = AlphaFactors(df)
        result = af.compute_all(include_volume=True)
        expected_factors = AlphaFactors.get_factor_names(include_volume=True)
        for f in expected_factors:
            assert f in result.columns, f"缺少因子: {f}"

    def test_compute_all_without_volume(self):
        df = _make_close_only_df()
        af = AlphaFactors(df)
        result = af.compute_all(include_volume=False)
        expected_factors = AlphaFactors.get_factor_names(include_volume=False)
        for f in expected_factors:
            assert f in result.columns, f"缺少因子: {f}"

    def test_factor_count(self):
        """确保至少有 15 个基础因子"""
        base_factors = AlphaFactors.get_factor_names(include_volume=False)
        assert len(base_factors) >= 15, f"基础因子数不足 15：{len(base_factors)}"

        all_factors = AlphaFactors.get_factor_names(include_volume=True)
        assert len(all_factors) >= 20, f"全部因子数不足 20：{len(all_factors)}"

    def test_price_acceleration(self):
        df = _make_ohlcv_df()
        af = AlphaFactors(df)
        result = af.price_acceleration()
        assert result.name == "price_acceleration"

    def test_relative_strength(self):
        df = _make_ohlcv_df()
        af = AlphaFactors(df)
        result = af.relative_strength()
        assert result.name == "relative_strength"

    def test_long_term_reversal(self):
        df = _make_ohlcv_df()
        af = AlphaFactors(df)
        result = af.long_term_reversal()
        assert result.name == "long_term_reversal"

    def test_downside_volatility(self):
        df = _make_ohlcv_df()
        af = AlphaFactors(df)
        result = af.downside_volatility()
        assert result.name == "downside_volatility"
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_high_low_range(self):
        df = _make_ohlcv_df()
        af = AlphaFactors(df)
        result = af.high_low_range()
        assert result.name == "high_low_range"
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_turnover_adjusted_momentum(self):
        df = _make_ohlcv_df()
        af = AlphaFactors(df)
        result = af.turnover_adjusted_momentum()
        assert result.name == "turnover_adjusted_momentum"

    def test_volume_price_divergence(self):
        df = _make_ohlcv_df()
        af = AlphaFactors(df)
        result = af.volume_price_divergence()
        assert result.name == "volume_price_divergence"


# =========================================================
# FactorSelector 测试
# =========================================================

class TestFactorSelector:
    def _get_df_with_factors(self, n: int = 150) -> tuple:
        df = _make_ohlcv_df(n)
        af = AlphaFactors(df)
        df_factors = af.compute_all(include_volume=True)
        factor_names = [
            f for f in AlphaFactors.get_factor_names(include_volume=True)
            if df_factors[f].notna().sum() > 20
        ]
        return df_factors, factor_names

    def test_compute_mean_ic(self):
        df_factors, factor_names = self._get_df_with_factors()
        fs = FactorSelector(df_factors, factor_names)
        ic = fs.compute_mean_ic(factor_names[0], period=5)
        assert isinstance(ic, float)
        assert -1.0 <= ic <= 1.0

    def test_compute_rank_ic(self):
        df_factors, factor_names = self._get_df_with_factors()
        fs = FactorSelector(df_factors, factor_names)
        rank_ic = fs.compute_rank_ic(factor_names[0], period=5)
        assert isinstance(rank_ic, float)
        assert -1.0 <= rank_ic <= 1.0

    def test_compute_ic_scores_returns_dataframe(self):
        df_factors, factor_names = self._get_df_with_factors()
        fs = FactorSelector(df_factors, factor_names)
        scores = fs.compute_ic_scores(period=5)
        assert isinstance(scores, pd.DataFrame)
        assert "factor" in scores.columns
        assert "ic" in scores.columns
        assert "rank_ic" in scores.columns
        assert len(scores) == len(factor_names)

    def test_ic_scores_sorted_by_abs_ic(self):
        df_factors, factor_names = self._get_df_with_factors()
        fs = FactorSelector(df_factors, factor_names)
        scores = fs.compute_ic_scores(period=5)
        ic_abs = scores["ic_abs"].values
        assert all(ic_abs[i] >= ic_abs[i + 1] for i in range(len(ic_abs) - 1))

    def test_compute_ic_decay(self):
        df_factors, factor_names = self._get_df_with_factors()
        fs = FactorSelector(df_factors, factor_names)
        decay = fs.compute_ic_decay(factor_names[0], max_period=10)
        assert isinstance(decay, pd.DataFrame)
        assert "period" in decay.columns
        assert "ic" in decay.columns
        assert len(decay) == 10

    def test_select_top_factors(self):
        df_factors, factor_names = self._get_df_with_factors()
        fs = FactorSelector(df_factors, factor_names)
        best = fs.select_top_factors(n=3, period=5, min_ic_abs=0.0)
        assert isinstance(best, list)
        assert len(best) <= 3

    def test_compute_factor_correlation(self):
        df_factors, factor_names = self._get_df_with_factors()
        fs = FactorSelector(df_factors, factor_names[:5])
        corr = fs.compute_factor_correlation()
        assert isinstance(corr, pd.DataFrame)
        # 对角线应为 1
        for f in factor_names[:5]:
            if f in corr.columns and f in corr.index:
                assert abs(corr.loc[f, f] - 1.0) < 1e-6

    def test_get_low_correlation_factors(self):
        df_factors, factor_names = self._get_df_with_factors()
        fs = FactorSelector(df_factors, factor_names)
        low_corr = fs.get_low_correlation_factors(threshold=0.9)
        assert isinstance(low_corr, list)
        assert len(low_corr) <= len(factor_names)

    def test_invalid_factor_returns_zero(self):
        """不在 DataFrame 中的因子，compute_mean_ic 应返回 0.0 而不是抛出异常"""
        df_factors, factor_names = self._get_df_with_factors()
        fs = FactorSelector(df_factors, factor_names)
        ic = fs.compute_mean_ic("non_existent_factor", period=5)
        assert ic == 0.0

    def test_compute_ic_invalid_factor_raises(self):
        """compute_ic（逐日 IC 序列）对不存在的因子应抛出 ValueError"""
        df_factors, factor_names = self._get_df_with_factors()
        fs = FactorSelector(df_factors, factor_names)
        with pytest.raises(ValueError):
            fs.compute_ic("non_existent_factor", period=5)


# =========================================================
# FactorEvaluator 测试
# =========================================================

class TestFactorEvaluator:
    def _get_df_with_factor(self) -> pd.DataFrame:
        df = _make_ohlcv_df(150)
        af = AlphaFactors(df)
        return af.compute_all(include_volume=True)

    def test_run_quantile_backtest(self):
        df = self._get_df_with_factor()
        ev = FactorEvaluator(df, "momentum_20d", n_quantiles=5)
        result = ev.run_quantile_backtest()
        assert isinstance(result, dict)
        assert "quantile_returns" in result
        assert "long_short_return" in result
        assert "long_short_sharpe" in result
        assert "factor_monotonicity" in result

    def test_quantile_returns_shape(self):
        df = self._get_df_with_factor()
        ev = FactorEvaluator(df, "momentum_20d", n_quantiles=5)
        result = ev.run_quantile_backtest()
        qr = result["quantile_returns"]
        assert len(qr) <= 5  # 最多 5 层

    def test_get_factor_stats(self):
        df = self._get_df_with_factor()
        ev = FactorEvaluator(df, "momentum_20d")
        stats = ev.get_factor_stats()
        assert isinstance(stats, dict)
        assert "mean" in stats
        assert "std" in stats
        assert "skew" in stats
        assert "kurt" in stats

    def test_evaluate_all_factors(self):
        df = self._get_df_with_factor()
        factor_names = ["momentum_5d", "momentum_20d", "short_term_reversal", "macd_signal"]
        result = evaluate_all_factors(df, factor_names)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(factor_names)
        assert "factor" in result.columns
        assert "ls_return" in result.columns
        assert "ls_sharpe" in result.columns

    def test_insufficient_data_returns_gracefully(self):
        """数据不足时不应抛出异常"""
        df = _make_close_only_df(n=20)
        af = AlphaFactors(df)
        df_f = af.compute_all(include_volume=False)
        ev = FactorEvaluator(df_f, "momentum_5d", n_quantiles=5)
        result = ev.run_quantile_backtest()
        assert isinstance(result, dict)
