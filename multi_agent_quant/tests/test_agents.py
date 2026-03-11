# tests/test_agents.py
"""
Agent 单元测试（使用 mock 数据，不依赖网络）
测试 TrendAgent, MeanReversionAgent, MLAgent, CoordinatorAgent
"""

import numpy as np
import pandas as pd
import pytest

from agents.trend_agent import TrendAgent
from agents.mean_reversion_agent import MeanReversionAgent
from agents.ml_agent import MLAgent
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


# ===== MLAgent 测试 =====

class TestMLAgent:
    def test_signal_always_hold(self):
        """MLAgent 永远输出 signal='hold'"""
        df = _make_uptrend_df(80)
        agent = MLAgent(lookback=10, min_train_size=60)
        result = agent.generate_signal(df)
        assert result["signal"] == "hold"

    def test_meta_aux_is_true(self):
        """meta.aux 必须是 True"""
        df = _make_uptrend_df(80)
        agent = MLAgent()
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
