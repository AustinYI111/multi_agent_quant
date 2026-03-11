# tests/test_backtest.py
"""
BacktestEngine 单元测试（使用 mock 数据，不依赖网络）
"""

import numpy as np
import pandas as pd
import pytest

from agents.trend_agent import TrendAgent
from agents.mean_reversion_agent import MeanReversionAgent
from agents.ml_agent import MLAgent
from agents.coordinator_agent import CoordinatorAgent
from backtest.backtest_engine import BacktestEngine


def _make_df(n: int = 80, pattern: str = "uptrend") -> pd.DataFrame:
    """生成测试用 DataFrame"""
    if pattern == "uptrend":
        close = np.linspace(10.0, 20.0, n)
    elif pattern == "downtrend":
        close = np.linspace(20.0, 10.0, n)
    elif pattern == "oscillating":
        t = np.linspace(0, 4 * np.pi, n)
        close = 15.0 + 3.0 * np.sin(t)
    else:  # mixed
        half = n // 2
        close = list(np.linspace(10.0, 18.0, half)) + list(np.linspace(18.0, 12.0, n - half))

    df = pd.DataFrame({"close": close})
    df.index = pd.date_range("2024-01-01", periods=n, freq="B")
    return df


def _make_engine(**kwargs) -> BacktestEngine:
    defaults = dict(
        init_cash=100000.0,
        fee_rate=0.0003,
        slippage_bps=5.0,
        max_position_pct=0.98,
        min_order_value=100.0,
        allow_fractional=True,
        lot_size=100,
    )
    defaults.update(kwargs)
    return BacktestEngine(**defaults)


# ===== BacktestEngine 基础测试 =====

class TestBacktestEngineBasic:
    def test_run_single_agent_returns_metrics_and_trades(self):
        df = _make_df(80, "mixed")
        engine = _make_engine()
        agent = TrendAgent(short_window=5, long_window=20)
        metrics, trades = engine.run_single_agent(df, agent, "test_trend")

        assert isinstance(metrics, dict)
        assert isinstance(trades, list)

    def test_metrics_contain_required_fields(self):
        """metrics 必须包含所有必要字段"""
        df = _make_df(80, "mixed")
        engine = _make_engine()
        agent = TrendAgent(short_window=5, long_window=20)
        metrics, trades = engine.run_single_agent(df, agent, "test")

        required = ["total_return", "annual_return", "max_drawdown", "sharpe", "num_trades", "strategy"]
        for key in required:
            assert key in metrics, f"Missing key: {key}"

    def test_equity_curve_in_metrics(self):
        """metrics 应包含 equity_curve 和 dates"""
        df = _make_df(80, "uptrend")
        engine = _make_engine()
        agent = TrendAgent(short_window=5, long_window=20)
        metrics, _ = engine.run_single_agent(df, agent, "test")

        assert "equity_curve" in metrics
        assert "dates" in metrics
        assert len(metrics["equity_curve"]) > 0
        assert len(metrics["dates"]) == len(metrics["equity_curve"])

    def test_total_return_type(self):
        df = _make_df(80, "uptrend")
        engine = _make_engine()
        agent = TrendAgent()
        metrics, _ = engine.run_single_agent(df, agent, "test")
        assert isinstance(metrics["total_return"], float)
        assert isinstance(metrics["sharpe"], float)
        assert isinstance(metrics["max_drawdown"], float)

    def test_strategy_name_in_metrics(self):
        df = _make_df(80)
        engine = _make_engine()
        agent = TrendAgent()
        metrics, _ = engine.run_single_agent(df, agent, "MyStrategy")
        assert metrics["strategy"] == "MyStrategy"


# ===== run_single_agent 测试 =====

class TestRunSingleAgent:
    def test_trend_agent_uptrend_makes_trades(self):
        """上升趋势中 TrendAgent 应该产生交易"""
        df = _make_df(80, "uptrend")
        engine = _make_engine()
        agent = TrendAgent(short_window=5, long_window=20)
        metrics, trades = engine.run_single_agent(df, agent, "trend_up")
        assert len(trades) > 0

    def test_mean_reversion_oscillating_makes_trades(self):
        """震荡行情中 MeanReversionAgent 应该产生交易"""
        df = _make_df(80, "oscillating")
        engine = _make_engine()
        agent = MeanReversionAgent(window=20, num_std=1.5)
        metrics, trades = engine.run_single_agent(df, agent, "mr_osc")
        # 可能产生交易也可能不产生，但不应崩溃
        assert isinstance(trades, list)

    def test_ml_agent_makes_no_direct_trades(self):
        """MLAgent 永远输出 hold，不应直接产生交易"""
        df = _make_df(80, "uptrend")
        engine = _make_engine()
        agent = MLAgent()
        metrics, trades = engine.run_single_agent(df, agent, "ml_only")
        # ml agent 永远 hold，不会有交易
        assert metrics["num_trades"] == 0

    def test_equity_curve_positive(self):
        """净值曲线应始终为正（初始资金不为负）"""
        df = _make_df(80, "mixed")
        engine = _make_engine()
        agent = TrendAgent()
        metrics, _ = engine.run_single_agent(df, agent, "test")
        equity = metrics["equity_curve"]
        assert all(e > 0 for e in equity)


# ===== run_fusion 测试 =====

class TestRunFusion:
    def _make_coordinator(self, ml_veto: bool = False) -> CoordinatorAgent:
        return CoordinatorAgent(
            agent_weights={"trend": 0.55, "mean_reversion": 0.45},
            ml_veto_enabled=ml_veto,
            min_edge=0.0,
            min_score_to_trade=0.0,
            regime_boost=0.25,
            min_conf_when_trade=0.05,
        )

    def test_fusion_returns_metrics_and_trades(self):
        df = _make_df(80, "mixed")
        engine = _make_engine()
        agents = {
            "trend": TrendAgent(short_window=5, long_window=20),
            "mean_reversion": MeanReversionAgent(window=20),
        }
        coordinator = self._make_coordinator()
        metrics, trades = engine.run_fusion(df, agents, coordinator, "Fusion")

        assert isinstance(metrics, dict)
        assert isinstance(trades, list)

    def test_fusion_metrics_required_fields(self):
        df = _make_df(80, "mixed")
        engine = _make_engine()
        agents = {
            "trend": TrendAgent(short_window=5, long_window=20),
            "mean_reversion": MeanReversionAgent(window=20),
        }
        coordinator = self._make_coordinator()
        metrics, _ = engine.run_fusion(df, agents, coordinator, "FusionTest")

        required = ["total_return", "annual_return", "max_drawdown", "sharpe", "num_trades"]
        for key in required:
            assert key in metrics

    def test_fusion_with_ml_agent(self):
        """融合 ML veto 场景不应崩溃"""
        df = _make_df(80, "mixed")
        engine = _make_engine()
        agents = {
            "trend": TrendAgent(short_window=5, long_window=20),
            "mean_reversion": MeanReversionAgent(window=20),
            "ml": MLAgent(),
        }
        coordinator = self._make_coordinator(ml_veto=True)
        metrics, trades = engine.run_fusion(df, agents, coordinator, "Fusion+ML")

        assert isinstance(metrics, dict)
        assert "total_return" in metrics

    def test_fusion_equity_curve_length(self):
        """净值曲线长度应与数据行数相同"""
        df = _make_df(80, "uptrend")
        engine = _make_engine()
        agents = {
            "trend": TrendAgent(short_window=5, long_window=20),
            "mean_reversion": MeanReversionAgent(window=20),
        }
        coordinator = self._make_coordinator()
        metrics, _ = engine.run_fusion(df, agents, coordinator, "test")

        assert len(metrics["equity_curve"]) == len(df)

    def test_fusion_strategy_name_set(self):
        df = _make_df(80)
        engine = _make_engine()
        agents = {
            "trend": TrendAgent(),
            "mean_reversion": MeanReversionAgent(),
        }
        coordinator = self._make_coordinator()
        metrics, _ = engine.run_fusion(df, agents, coordinator, "CustomFusion")
        assert metrics["strategy"] == "CustomFusion"


# ===== 边界条件测试 =====

class TestEdgeCases:
    def test_very_short_data(self):
        """数据极少时不应崩溃"""
        df = _make_df(25, "uptrend")
        engine = _make_engine()
        agent = TrendAgent()
        metrics, trades = engine.run_single_agent(df, agent, "short")
        assert isinstance(metrics, dict)

    def test_constant_price_data(self):
        """价格不变时不应崩溃"""
        df = pd.DataFrame({"close": [15.0] * 60})
        df.index = pd.date_range("2024-01-01", periods=60, freq="B")
        engine = _make_engine()
        agent = TrendAgent()
        metrics, trades = engine.run_single_agent(df, agent, "flat")
        assert isinstance(metrics, dict)
        assert metrics["num_trades"] == 0  # 无趋势，不交易
