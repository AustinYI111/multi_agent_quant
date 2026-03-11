"""
multi_agent_quant/run.py — 项目主入口
正确调用 DataAgent, TrendAgent, MeanReversionAgent, MLAgent, CoordinatorAgent, BacktestEngine
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from agents.data_agent import DataAgent
from agents.trend_agent import TrendAgent
from agents.mean_reversion_agent import MeanReversionAgent
from agents.ml_agent import MLAgent
from agents.coordinator_agent import CoordinatorAgent
from backtest.backtest_engine import BacktestEngine


def _print_summary(name: str, metrics: Dict[str, Any], trades: List[Any]) -> None:
    keys = ["total_return", "annual_return", "max_drawdown", "sharpe", "num_trades"]
    summary = {k: metrics.get(k) for k in keys}
    print(f"\n===== {name} =====")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


def _to_row(metrics: Dict[str, Any]) -> Dict[str, Any]:
    keys = ["strategy", "total_return", "annual_return", "max_drawdown", "sharpe", "num_trades"]
    return {k: metrics.get(k) for k in keys}


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Quantitative Trading Backtest")
    parser.add_argument("--symbol", type=str, default="600519", help="股票代码")
    parser.add_argument("--start", type=str, default="20200101", help="开始日期 YYYYMMDD")
    parser.add_argument("--end", type=str, default="20241231", help="结束日期 YYYYMMDD")
    parser.add_argument("--adjust", type=str, default="qfq", help="复权方式")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="数据缓存目录")
    parser.add_argument("--init_cash", type=float, default=100000.0, help="初始资金")
    parser.add_argument("--outdir", type=str, default="outputs", help="结果输出目录")
    parser.add_argument("--verbose", action="store_true", help="打印每笔交易")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. 数据 Agent
    print(f"[DataAgent] 获取 {args.symbol} 数据 {args.start} ~ {args.end}")
    data_agent = DataAgent(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        adjust=args.adjust,
        data_dir=args.data_dir,
    )
    df = data_agent.get_feature_data(use_cache=True, add_indicators=True)
    # 按日期切片
    start_dt = pd.to_datetime(args.start, format="%Y%m%d")
    end_dt = pd.to_datetime(args.end, format="%Y%m%d")
    df = df.sort_index().loc[start_dt:end_dt]
    print(f"[DataAgent] 数据行数: {len(df)}，日期范围: {df.index.min()} ~ {df.index.max()}")

    # 2. 策略 Agents
    trend_agent = TrendAgent(short_window=5, long_window=20)
    mr_agent = MeanReversionAgent(window=20, num_std=1.2, conf_scale=1.0)
    ml_agent = MLAgent(lookback=10, min_train_size=60, prob_threshold=0.55)

    # 3. 协调 Agent（ml 是辅助 agent，不需要在 agent_weights 里）
    coordinator = CoordinatorAgent(
        agent_weights={"trend": 0.60, "mean_reversion": 0.40},
        ml_veto_enabled=True,
        min_edge=0.01,
        min_score_to_trade=0.01,
        regime_boost=0.25,
        min_conf_when_trade=0.05,
    )

    # 4. 回测引擎
    engine = BacktestEngine(
        init_cash=args.init_cash,
        fee_rate=0.0003,
        slippage_bps=5.0,
        max_position_pct=0.98,
        min_order_value=1000.0,
        lot_size=100,
    )

    # 5. 单 Agent 回测
    print("\n[回测] Trend-only ...")
    m1, t1 = engine.run_single_agent(df, trend_agent, "Trend-only", verbose=args.verbose)

    print("[回测] MeanRev-only ...")
    m2, t2 = engine.run_single_agent(df, mr_agent, "MeanRev-only", verbose=args.verbose)

    print("[回测] ML-only ...")
    m3, t3 = engine.run_single_agent(df, ml_agent, "ML-only", verbose=args.verbose)

    # 6. 融合策略回测（Trend + MeanRev）
    print("[回测] Fusion(Trend+MeanRev) ...")
    fusion_agents = {"trend": trend_agent, "mean_reversion": mr_agent}
    coordinator_no_ml = CoordinatorAgent(
        agent_weights={"trend": 0.60, "mean_reversion": 0.40},
        ml_veto_enabled=False,
    )
    m4, t4 = engine.run_fusion(
        df, fusion_agents, coordinator_no_ml, "Fusion(Trend+MeanRev)", verbose=args.verbose
    )

    # 7. 融合策略回测（Trend + MeanRev + ML veto）
    print("[回测] Fusion(Trend+MeanRev+ML) ...")
    fusion_agents_ml = {"trend": trend_agent, "mean_reversion": mr_agent, "ml": ml_agent}
    m5, t5 = engine.run_fusion(
        df, fusion_agents_ml, coordinator, "Fusion(Trend+MeanRev+ML)", verbose=args.verbose
    )

    # 8. 打印结果对比
    _print_summary("Trend-only", m1, t1)
    _print_summary("MeanRev-only", m2, t2)
    _print_summary("ML-only", m3, t3)
    _print_summary("Fusion(Trend+MeanRev)", m4, t4)
    _print_summary("Fusion(Trend+MeanRev+ML)", m5, t5)

    table = pd.DataFrame([_to_row(m) for m in [m1, m2, m3, m4, m5]])
    print("\n=== 策略对比汇总 ===")
    print(table.to_string(index=False))

    table_path = outdir / "comparison_table.csv"
    table.to_csv(table_path, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存: {table_path}")


if __name__ == "__main__":
    main()

