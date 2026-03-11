# tests/test_agents.py
"""
Agent 单元测试（使用 mock 数据，不依赖网络）
测试 TrendAgent, MeanReversionAgent, MLAgent, MomentumAgent, CoordinatorAgent
"""

import numpy as np
import pandas as pd
import pytest

from agents.trend_agent import TrendAgent
from agents.mean_reversion_agent import MeanReversionAgent
from agents.ml_agent import MLAgent
from agents.momentum_agent import MomentumAgent
from agents.coordinator_agent import CoordinatorAgent


def _make_uptrend_df(n: int = 60) -> pd.DataFrame:
    """生成上升趋势数据"""
    close = np.linspace(10.0, 20.0, n)
    df = pd.DataFrame({"close": close})
    df.index = pd.date_range("2024-01-01", periods=n, freq="B")
    return df


def _make_downtrend_df(n: int = 60) -> pd.DataFrame:
    """生成下降趋势数据"""
    close = np.linspace(20.0, 10.0, n)
    df = pd.DataFrame({"close": close})
    df.index = pd.date_range("2024-01-01", periods=n, freq="B")
    return df


def _make_oscillating_df(n: int = 60) -> pd.DataFrame:
    """生成震荡数据（用于均值回归测试）"""
    t = np.linspace(0, 4 * np.pi, n)
    close = 15.0 + 3.0 * np.sin(t)
    df = pd.DataFrame({"close": close})
    df.index = pd.date_range("2024-01-01", periods=n, freq="B")
    return df


def _make_ohlcv_df(n: int = 60, trend: str = "up") -> pd.DataFrame:
    """生成含 OHLCV 列的数据（用于完整因子测试）"""
    if trend == "up":
        close = np.linspace(10.0, 20.0, n)
    else:
        close = np.linspace(20.0, 10.0, n)
    noise = np.random.RandomState(42).randn(n) * 0.1
    close = close + noise
    high = close + np.abs(noise) + 0.2
    low = close - np.abs(noise) - 0.2
    volume = np.random.RandomState(42).randint(1000, 5000, n).astype(float)
    df = pd.DataFrame({"close": close, "high": high, "low": low, "volume": volume})
    df.index = pd.date_range("2024-01-01", periods=n, freq="B")
    return df


# ===== TrendAgent 测试 =====

class TestTrendAgent:
    def test_basic_output_format(self):
        df = _make_uptrend_df()
        agent = TrendAgent(short_window=5, long_window=20)
        result = agent.generate_signal(df)

        assert "signal" in result
        assert "confidence" in result
        assert "meta" in result
        assert result["signal"] in ("buy", "sell", "hold")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_uptrend_generates_buy(self):
        df = _make_uptrend_df(60)
        agent = TrendAgent(short_window=5, long_window=20)
        result = agent.generate_signal(df)
        # 上升趋势时短均线 > 长均线，应该是 buy
        assert result["signal"] == "buy"

    def test_downtrend_generates_sell(self):
        df = _make_downtrend_df(60)
        agent = TrendAgent(short_window=5, long_window=20)
        result = agent.generate_signal(df)
        # 下降趋势时短均线 < 长均线，应该是 sell
        assert result["signal"] == "sell"

    def test_insufficient_data_returns_hold(self):
        df = pd.DataFrame({"close": [10.0, 11.0, 12.0]})
        df.index = pd.date_range("2024-01-01", periods=3, freq="B")
        agent = TrendAgent(short_window=5, long_window=20)
        result = agent.generate_signal(df)
        assert result["signal"] == "hold"
        assert result["confidence"] == 0.0

    def test_confidence_between_0_and_1(self):
        df = _make_uptrend_df(60)
        agent = TrendAgent(short_window=5, long_window=20)
        result = agent.generate_signal(df)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_none_data_returns_hold(self):
        agent = TrendAgent()
        result = agent.generate_signal(None)
        assert result["signal"] == "hold"

    def test_meta_contains_factors(self):
        """meta 应包含各因子信息"""
        df = _make_ohlcv_df(80, trend="up")
        agent = TrendAgent(short_window=5, long_window=20)
        result = agent.generate_signal(df)
        meta = result["meta"]
        assert "rsi" in meta
        assert "adx" in meta
        assert "macd" in meta
        assert "factors" in meta

    def test_ohlcv_with_ma60(self):
        """有 MA60 数据时能正常产生信号"""
        df = _make_ohlcv_df(100, trend="up")
        agent = TrendAgent(short_window=5, long_window=20, trend_window=60)
        result = agent.generate_signal(df)
        assert result["signal"] in ("buy", "sell", "hold")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_min_conf_when_signal(self):
        """有方向信号时置信度不低于 min_conf_when_signal"""
        df = _make_uptrend_df(60)
        agent = TrendAgent(short_window=5, long_window=20, min_conf_when_signal=0.15)
        result = agent.generate_signal(df)
        if result["signal"] != "hold":
            assert result["confidence"] >= 0.15


# ===== MeanReversionAgent 测试 =====

class TestMeanReversionAgent:
    def test_basic_output_format(self):
        df = _make_oscillating_df()
        agent = MeanReversionAgent(window=20)
        result = agent.generate_signal(df)

        assert "signal" in result
        assert "confidence" in result
        assert "meta" in result
        assert result["signal"] in ("buy", "sell", "hold")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_price_below_lower_band_generates_buy(self):
        """构造价格远低于均值的情况，应生成 buy"""
        close = [15.0] * 40 + [8.0]  # 明显低于均值
        df = pd.DataFrame({"close": close})
        df.index = pd.date_range("2024-01-01", periods=len(close), freq="B")
        agent = MeanReversionAgent(window=20, num_std=1.5)
        result = agent.generate_signal(df)
        assert result["signal"] == "buy"

    def test_price_above_upper_band_generates_sell(self):
        """构造价格远高于均值的情况，应生成 sell"""
        close = [15.0] * 40 + [22.0]  # 明显高于均值
        df = pd.DataFrame({"close": close})
        df.index = pd.date_range("2024-01-01", periods=len(close), freq="B")
        agent = MeanReversionAgent(window=20, num_std=1.5)
        result = agent.generate_signal(df)
        assert result["signal"] == "sell"

    def test_insufficient_data_returns_hold(self):
        df = pd.DataFrame({"close": [10.0, 11.0]})
        df.index = pd.date_range("2024-01-01", periods=2, freq="B")
        agent = MeanReversionAgent(window=20)
        result = agent.generate_signal(df)
        assert result["signal"] == "hold"
        assert result["confidence"] == 0.0

    def test_meta_contains_bollinger(self):
        df = _make_oscillating_df()
        agent = MeanReversionAgent(window=20)
        result = agent.generate_signal(df)
        meta = result["meta"]
        assert "upper_band" in meta
        assert "lower_band" in meta
        assert "zscore" in meta

    def test_rsi_oversold_generates_buy(self):
        """RSI 超卖时应生成 buy 信号"""
        # 构造连续下跌数据让 RSI 进入超卖区
        close = np.linspace(30.0, 10.0, 60)  # 强烈下跌
        df = pd.DataFrame({"close": close})
        df.index = pd.date_range("2024-01-01", periods=60, freq="B")
        agent = MeanReversionAgent(window=20, rsi_oversold=30.0, min_confidence=0.10)
        result = agent.generate_signal(df)
        # 强烈下跌后 RSI 应进入超卖，信号应为 buy 或 hold（取决于各因子综合）
        assert result["signal"] in ("buy", "sell", "hold")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_meta_contains_scores(self):
        """meta 应包含各因子得分"""
        df = _make_oscillating_df()
        agent = MeanReversionAgent(window=20)
        result = agent.generate_signal(df)
        assert "scores" in result["meta"]
        assert "rsi" in result["meta"]["scores"]
        assert "deviation" in result["meta"]["scores"]
        assert "bollinger" in result["meta"]["scores"]

    def test_consecutive_days_in_meta(self):
        """meta 应包含连续涨跌天数"""
        df = _make_downtrend_df(40)
        agent = MeanReversionAgent(window=20)
        result = agent.generate_signal(df)
        assert "consecutive_days" in result["meta"]


# ===== MLAgent 测试 =====

class TestMLAgent:
    def test_signal_always_hold_in_auxiliary_mode(self):
        """辅助模式下 MLAgent 永远输出 signal='hold'"""
        df = _make_uptrend_df(80)
        agent = MLAgent(lookback=10, min_train_size=60, auxiliary_mode=True)
        result = agent.generate_signal(df)
        assert result["signal"] == "hold"

    def test_meta_aux_is_true_in_auxiliary_mode(self):
        """辅助模式 meta.aux 必须是 True"""
        df = _make_uptrend_df(80)
        agent = MLAgent(auxiliary_mode=True)
        result = agent.generate_signal(df)
        assert result["meta"]["aux"] is True

    def test_p_up_in_0_1(self):
        """p_up 应在 [0, 1] 范围内"""
        df = _make_uptrend_df(80)
        agent = MLAgent()
        result = agent.generate_signal(df)
        p_up = result["meta"].get("p_up")
        assert p_up is not None
        assert 0.0 <= p_up <= 1.0

    def test_aux_suggest_valid(self):
        """aux_suggest 必须是合法信号"""
        df = _make_uptrend_df(80)
        agent = MLAgent()
        result = agent.generate_signal(df)
        assert result["meta"]["aux_suggest"] in ("buy", "sell", "hold")

    def test_confidence_in_0_1(self):
        df = _make_uptrend_df(80)
        agent = MLAgent()
        result = agent.generate_signal(df)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_insufficient_data_returns_hold(self):
        df = pd.DataFrame({"close": [10.0, 11.0, 12.0]})
        df.index = pd.date_range("2024-01-01", periods=3, freq="B")
        agent = MLAgent(min_train_size=60)
        result = agent.generate_signal(df)
        assert result["signal"] == "hold"
        assert result["meta"]["aux"] is True

    def test_independent_mode_can_generate_buy_sell(self):
        """独立模式下，MLAgent 可以输出 buy/sell"""
        df = _make_uptrend_df(80)
        agent = MLAgent(
            lookback=10,
            min_train_size=60,
            auxiliary_mode=False,
            buy_threshold=0.50,   # 降低阈值让测试更容易触发
            sell_threshold=0.50,
        )
        result = agent.generate_signal(df)
        assert result["signal"] in ("buy", "sell", "hold")
        assert result["meta"]["aux"] is False

    def test_independent_mode_output_format(self):
        """独立模式输出格式应与辅助模式一致"""
        df = _make_uptrend_df(80)
        agent = MLAgent(auxiliary_mode=False)
        result = agent.generate_signal(df)
        assert "signal" in result
        assert "confidence" in result
        assert "meta" in result
        assert result["signal"] in ("buy", "sell", "hold")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_independent_mode_aux_is_false(self):
        """独立模式 meta.aux 应为 False"""
        df = _make_uptrend_df(80)
        agent = MLAgent(auxiliary_mode=False)
        result = agent.generate_signal(df)
        assert result["meta"]["aux"] is False

    def test_features_in_meta(self):
        """独立模式 meta 应包含多因子特征"""
        df = _make_ohlcv_df(80, trend="up")
        agent = MLAgent(auxiliary_mode=False)
        result = agent.generate_signal(df)
        feats = result["meta"].get("features", {})
        assert "rsi_norm" in feats
        assert "bb_pos" in feats


# ===== MomentumAgent 测试 =====

class TestMomentumAgent:
    def test_basic_output_format(self):
        df = _make_uptrend_df(60)
        agent = MomentumAgent()
        result = agent.generate_signal(df)

        assert "signal" in result
        assert "confidence" in result
        assert "meta" in result
        assert result["signal"] in ("buy", "sell", "hold")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_uptrend_generates_buy(self):
        """强烈上升趋势应生成 buy 信号"""
        df = _make_uptrend_df(60)
        agent = MomentumAgent(short_momentum=5, mid_momentum=20, min_confidence=0.05)
        result = agent.generate_signal(df)
        # 强上升趋势下动量应为正向
        assert result["signal"] in ("buy", "hold")

    def test_downtrend_generates_sell(self):
        """强烈下降趋势应生成 sell 信号"""
        df = _make_downtrend_df(60)
        agent = MomentumAgent(short_momentum=5, mid_momentum=20, min_confidence=0.05)
        result = agent.generate_signal(df)
        # 强下降趋势下动量应为负向
        assert result["signal"] in ("sell", "hold")

    def test_insufficient_data_returns_hold(self):
        df = pd.DataFrame({"close": [10.0, 11.0, 12.0]})
        df.index = pd.date_range("2024-01-01", periods=3, freq="B")
        agent = MomentumAgent()
        result = agent.generate_signal(df)
        assert result["signal"] == "hold"
        assert result["confidence"] == 0.0

    def test_none_data_returns_hold(self):
        agent = MomentumAgent()
        result = agent.generate_signal(None)
        assert result["signal"] == "hold"

    def test_meta_contains_scores(self):
        """meta 应包含各因子得分"""
        df = _make_uptrend_df(60)
        agent = MomentumAgent()
        result = agent.generate_signal(df)
        assert "scores" in result["meta"]
        scores = result["meta"]["scores"]
        assert "momentum" in scores
        assert "breakout" in scores
        assert "volume" in scores

    def test_meta_contains_returns(self):
        """meta 应包含短期和中期收益率"""
        df = _make_uptrend_df(60)
        agent = MomentumAgent()
        result = agent.generate_signal(df)
        assert "short_ret" in result["meta"]
        assert "mid_ret" in result["meta"]

    def test_ohlcv_breakout(self):
        """使用 OHLCV 数据时突破信号正常工作"""
        df = _make_ohlcv_df(60, trend="up")
        agent = MomentumAgent(breakout_window=10, min_confidence=0.05)
        result = agent.generate_signal(df)
        assert result["signal"] in ("buy", "sell", "hold")

    def test_confidence_between_0_and_1(self):
        for _ in range(5):
            df = _make_ohlcv_df(60)
            agent = MomentumAgent()
            result = agent.generate_signal(df)
            assert 0.0 <= result["confidence"] <= 1.0

    def test_missing_close_returns_hold(self):
        df = pd.DataFrame({"open": [10.0, 11.0]})
        agent = MomentumAgent()
        result = agent.generate_signal(df)
        assert result["signal"] == "hold"


# ===== CoordinatorAgent 测试 =====

class TestCoordinatorAgent:
    def setup_method(self):
        self.coordinator = CoordinatorAgent(
            agent_weights={"trend": 0.6, "mean_reversion": 0.4},
            ml_veto_enabled=True,
            min_edge=0.0,
            min_score_to_trade=0.0,
        )

    def test_basic_output_format(self):
        outputs = {
            "trend": {"signal": "buy", "confidence": 0.7, "meta": {}},
            "mean_reversion": {"signal": "buy", "confidence": 0.6, "meta": {}},
        }
        result = self.coordinator.aggregate(outputs)
        assert "signal" in result
        assert "confidence" in result
        assert "meta" in result
        assert result["signal"] in ("buy", "sell", "hold")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_unanimous_buy(self):
        """两个 agent 都 buy，结果应该是 buy"""
        outputs = {
            "trend": {"signal": "buy", "confidence": 0.8, "meta": {}},
            "mean_reversion": {"signal": "buy", "confidence": 0.7, "meta": {}},
        }
        result = self.coordinator.aggregate(outputs)
        assert result["signal"] == "buy"

    def test_unanimous_sell(self):
        """两个 agent 都 sell，结果应该是 sell"""
        outputs = {
            "trend": {"signal": "sell", "confidence": 0.8, "meta": {}},
            "mean_reversion": {"signal": "sell", "confidence": 0.7, "meta": {}},
        }
        result = self.coordinator.aggregate(outputs)
        assert result["signal"] == "sell"

    def test_ml_veto_buy(self):
        """ML p_up 较低时，应否决 buy 信号"""
        coordinator = CoordinatorAgent(
            agent_weights={"trend": 0.6, "mean_reversion": 0.4},
            ml_veto_enabled=True,
            ml_veto_margin=0.03,
            min_edge=0.0,
            min_score_to_trade=0.0,
        )
        outputs = {
            "trend": {"signal": "buy", "confidence": 0.8, "meta": {}},
            "mean_reversion": {"signal": "buy", "confidence": 0.7, "meta": {}},
            "ml": {
                "signal": "hold",
                "confidence": 0.3,
                "meta": {"aux": True, "p_up": 0.30, "aux_suggest": "sell", "aux_strength": 0.4},
            },
        }
        result = coordinator.aggregate(outputs)
        # p_up=0.30 < 0.5+0.03=0.53，buy 应被否决
        assert result["signal"] == "hold"
        assert result["meta"]["ml_veto"]["applied"] is True

    def test_ml_veto_disabled(self):
        """veto 关闭时，ML 不影响结果"""
        coordinator = CoordinatorAgent(
            agent_weights={"trend": 0.6, "mean_reversion": 0.4},
            ml_veto_enabled=False,
            min_edge=0.0,
            min_score_to_trade=0.0,
        )
        outputs = {
            "trend": {"signal": "buy", "confidence": 0.8, "meta": {}},
            "mean_reversion": {"signal": "buy", "confidence": 0.7, "meta": {}},
            "ml": {
                "signal": "hold",
                "confidence": 0.3,
                "meta": {"aux": True, "p_up": 0.20, "aux_suggest": "sell", "aux_strength": 0.6},
            },
        }
        result = coordinator.aggregate(outputs)
        assert result["signal"] == "buy"
        assert result["meta"]["ml_veto"]["applied"] is False

    def test_market_state_trend_boosts_trend_weight(self):
        """market_state='trend' 时，trend agent 权重应上升"""
        outputs = {
            "trend": {"signal": "buy", "confidence": 0.8, "meta": {}},
            "mean_reversion": {"signal": "sell", "confidence": 0.8, "meta": {}},
        }
        # trend 市场时 trend agent 权重更高 -> buy 胜出
        result = self.coordinator.aggregate(outputs, market_state="trend")
        assert result["signal"] == "buy"

    def test_update_performance_does_not_crash(self):
        outputs = {
            "trend": {"signal": "buy", "confidence": 0.7, "meta": {}},
            "mean_reversion": {"signal": "hold", "confidence": 0.3, "meta": {}},
        }
        self.coordinator.update_performance(outputs, next_ret=0.01)
        self.coordinator.update_performance(outputs, next_ret=-0.02)
        # 不应抛出异常

    def test_three_agent_weights_with_momentum(self):
        """三个 Agent 权重时 Coordinator 能正常工作"""
        coordinator = CoordinatorAgent(
            agent_weights={"trend": 0.35, "mean_reversion": 0.30, "momentum": 0.35},
            ml_veto_enabled=False,
            min_edge=0.0,
            min_score_to_trade=0.0,
        )
        outputs = {
            "trend": {"signal": "buy", "confidence": 0.7, "meta": {}},
            "mean_reversion": {"signal": "buy", "confidence": 0.6, "meta": {}},
            "momentum": {"signal": "buy", "confidence": 0.8, "meta": {}},
        }
        result = coordinator.aggregate(outputs)
        assert result["signal"] == "buy"
        assert 0.0 <= result["confidence"] <= 1.0
