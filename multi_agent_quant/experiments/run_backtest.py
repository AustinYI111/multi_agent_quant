# experiments/run_backtest.py
"""
完整回测实验入口
支持命令行参数配置，输出对比表格并保存到 outputs/ 目录
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
from utils.metrics import compute_all_metrics


def _to_row(metrics: Dict[str, Any], trades: List[Any]) -> Dict[str, Any]:
    """合并 engine metrics 与 utils.metrics 计算结果"""
    equity = metrics.get("equity_curve", [])
    extra = compute_all_metrics(equity, trades)
    row = {
        "strategy": metrics.get("strategy", ""),
        "total_return": metrics.get("total_return", extra["total_return"]),
        "annual_return": metrics.get("annual_return", extra["annual_return"]),
        "max_drawdown": metrics.get("max_drawdown", extra["max_drawdown"]),
        "sharpe": metrics.get("sharpe", extra["sharpe"]),
        "calmar": extra["calmar"],
        "win_rate": extra["win_rate"],
        "profit_loss_ratio": extra["profit_loss_ratio"],
        "max_consec_losses": extra["max_consecutive_losses"],
        "num_trades": metrics.get("num_trades", extra["num_trades"]),
    }
    return row


def _save_equity(metrics: Dict[str, Any], outdir: Path) -> None:
    """将净值曲线保存为 CSV"""
    equity = metrics.get("equity_curve", [])
    dates = metrics.get("dates", [])
    strategy = metrics.get("strategy", "strategy")
    if not equity:
        return
    df = pd.DataFrame({"date": dates, "equity": equity})
    fname = outdir / f"equity_curve_{strategy}.csv"
    df.to_csv(fname, index=False, encoding="utf-8-sig")
    print(f"  保存净值曲线: {fname}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent 量化策略回测实验")

    # 数据参数
    parser.add_argument("--symbol", type=str, default="600519", help="股票代码")
    parser.add_argument("--start_date", type=str, default="20200101", help="开始日期 YYYYMMDD")
    parser.add_argument("--end_date", type=str, default="20241231", help="结束日期 YYYYMMDD")
    parser.add_argument("--adjust", type=str, default="qfq")
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--force_refresh", action="store_true")

    # 回测引擎参数
    parser.add_argument("--init_cash", type=float, default=100000.0)
    parser.add_argument("--fee_rate", type=float, default=0.0003)
    parser.add_argument("--slippage_bps", type=float, default=5.0)
    parser.add_argument("--max_position_pct", type=float, default=0.98)
    parser.add_argument("--min_order_value", type=float, default=1000.0)
    parser.add_argument("--lot_size", type=int, default=100)

    # 策略参数
    parser.add_argument("--trend_short", type=int, default=5)
    parser.add_argument("--trend_long", type=int, default=20)
    parser.add_argument("--mr_window", type=int, default=20)
    parser.add_argument("--mr_num_std", type=float, default=1.2)
    parser.add_argument("--mr_conf_scale", type=float, default=1.0)
    parser.add_argument("--ml_lookback", type=int, default=10)
    parser.add_argument("--ml_min_train_size", type=int, default=60)
    parser.add_argument("--ml_prob_threshold", type=float, default=0.55)

    # 协调器参数
    parser.add_argument("--w_trend", type=float, default=0.60)
    parser.add_argument("--w_mr", type=float, default=0.40)
    parser.add_argument("--min_edge", type=float, default=0.01)
    parser.add_argument("--min_score", type=float, default=0.01)
    parser.add_argument("--regime_boost", type=float, default=0.25)
    parser.add_argument("--min_conf_when_trade", type=float, default=0.05)

    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ===== 数据 =====
    print(f"[DataAgent] 获取 {args.symbol} 数据 {args.start_date} ~ {args.end_date}")
    da = DataAgent(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        adjust=args.adjust,
        data_dir=args.data_dir,
    )
    df = da.get_feature_data(
        use_cache=(not args.no_cache),
        force_refresh=args.force_refresh,
        add_indicators=True,
    )
    start_dt = pd.to_datetime(args.start_date, format="%Y%m%d")
    end_dt = pd.to_datetime(args.end_date, format="%Y%m%d")
    df = df.sort_index().loc[start_dt:end_dt]
    print(f"[DataAgent] 行数={len(df)}，范围={df.index.min()} ~ {df.index.max()}")

    # ===== Agents =====
    trend_agent = TrendAgent(args.trend_short, args.trend_long)
    mr_agent = MeanReversionAgent(args.mr_window, args.mr_num_std, args.mr_conf_scale)
    ml_agent = MLAgent(args.ml_lookback, args.ml_min_train_size, args.ml_prob_threshold)

    # ===== 引擎 =====
    engine = BacktestEngine(
        init_cash=args.init_cash,
        fee_rate=args.fee_rate,
        slippage_bps=args.slippage_bps,
        max_position_pct=args.max_position_pct,
        min_order_value=args.min_order_value,
        lot_size=args.lot_size,
    )

    all_rows = []

    # 1) Trend-only
    print("\n[回测] Trend-only ...")
    m, t = engine.run_single_agent(df, trend_agent, "Trend-only", args.verbose)
    all_rows.append(_to_row(m, t))
    _save_equity(m, outdir)

    # 2) MeanRev-only
    print("[回测] MeanRev-only ...")
    m, t = engine.run_single_agent(df, mr_agent, "MeanRev-only", args.verbose)
    all_rows.append(_to_row(m, t))
    _save_equity(m, outdir)

    # 3) ML-only
    print("[回测] ML-only ...")
    m, t = engine.run_single_agent(df, ml_agent, "ML-only", args.verbose)
    all_rows.append(_to_row(m, t))
    _save_equity(m, outdir)

    # 4) Fusion(Trend+MeanRev) — veto 关闭
    print("[回测] Fusion(Trend+MeanRev) ...")
    coord_no_ml = CoordinatorAgent(
        agent_weights={"trend": args.w_trend, "mean_reversion": args.w_mr},
        ml_veto_enabled=False,
        min_edge=args.min_edge,
        min_score_to_trade=args.min_score,
        regime_boost=args.regime_boost,
        min_conf_when_trade=args.min_conf_when_trade,
    )
    m, t = engine.run_fusion(
        df, {"trend": trend_agent, "mean_reversion": mr_agent},
        coord_no_ml, "Fusion(Trend+MeanRev)", args.verbose,
    )
    all_rows.append(_to_row(m, t))
    _save_equity(m, outdir)

    # 5) Fusion(Trend+MeanRev+ML) — veto 开启
    print("[回测] Fusion(Trend+MeanRev+ML) ...")
    coord_ml = CoordinatorAgent(
        agent_weights={"trend": args.w_trend, "mean_reversion": args.w_mr},
        ml_veto_enabled=True,
        min_edge=args.min_edge,
        min_score_to_trade=args.min_score,
        regime_boost=args.regime_boost,
        min_conf_when_trade=args.min_conf_when_trade,
    )
    m, t = engine.run_fusion(
        df, {"trend": trend_agent, "mean_reversion": mr_agent, "ml": ml_agent},
        coord_ml, "Fusion(Trend+MeanRev+ML)", args.verbose,
    )
    all_rows.append(_to_row(m, t))
    _save_equity(m, outdir)

    # ===== 输出对比表格 =====
    table = pd.DataFrame(all_rows)
    print("\n=== 策略回测对比 ===")
    print(table.to_string(index=False))

    table_path = outdir / "comparison_table.csv"
    table.to_csv(table_path, index=False, encoding="utf-8-sig")
    print(f"\n对比表格已保存: {table_path}")


if __name__ == "__main__":
    main()
