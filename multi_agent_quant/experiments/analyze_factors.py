# experiments/analyze_factors.py
"""
因子分析脚本 - 对 Alpha 因子库中的所有因子进行系统性分析

功能：
1. 单因子回测（分层回测）
2. IC 和 Rank IC 计算
3. 因子衰减系数分析
4. 最优因子筛选
5. 结果导出到 outputs/ 目录

运行示例:
    python -m experiments.analyze_factors
    python -m experiments.analyze_factors --symbol 600519 --top_n 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 确保 multi_agent_quant 包可被找到
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from multi_agent_quant.agents.data_agent import DataAgent
from multi_agent_quant.utils.alpha_factors import AlphaFactors
from multi_agent_quant.utils.factor_selector import FactorSelector
from multi_agent_quant.utils.factor_evaluator import FactorEvaluator, evaluate_all_factors


def main():
    parser = argparse.ArgumentParser(description="因子分析脚本")
    parser.add_argument("--symbol", type=str, default="600519", help="股票代码")
    parser.add_argument("--start_date", type=str, default="20200101")
    parser.add_argument("--end_date", type=str, default="20241231")
    parser.add_argument("--adjust", type=str, default="qfq")
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--forward_period", type=int, default=5, help="前瞻收益周期（天）")
    parser.add_argument("--top_n", type=int, default=5, help="选出的最优因子数")
    parser.add_argument("--outdir", type=str, default="outputs")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ===== 1. 获取数据 =====
    print(f"[DataAgent] 获取 {args.symbol} 数据 {args.start_date} ~ {args.end_date}")
    da = DataAgent(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        adjust=args.adjust,
        data_dir=args.data_dir,
    )
    df = da.get_feature_data(use_cache=True, add_indicators=True)
    start_dt = pd.to_datetime(args.start_date, format="%Y%m%d")
    end_dt = pd.to_datetime(args.end_date, format="%Y%m%d")
    df = df.sort_index().loc[start_dt:end_dt]
    print(f"[DataAgent] 行数={len(df)}，范围={df.index.min()} ~ {df.index.max()}")

    # ===== 2. 计算 Alpha 因子 =====
    print("\n[AlphaFactors] 计算所有因子...")
    has_volume = "volume" in df.columns
    af = AlphaFactors(df)
    df_factors = af.compute_all(include_volume=has_volume)
    factor_names = AlphaFactors.get_factor_names(include_volume=has_volume)
    # 过滤掉全 NaN 的因子
    valid_factors = [f for f in factor_names if f in df_factors.columns and df_factors[f].notna().sum() > 30]
    print(f"[AlphaFactors] 有效因子数: {len(valid_factors)}")

    # ===== 3. 因子 IC 评分 =====
    print(f"\n[FactorSelector] 计算 IC/Rank IC（前瞻 {args.forward_period} 日）...")
    fs = FactorSelector(df_factors, valid_factors, forward_periods=[args.forward_period, 10, 20])
    scores = fs.compute_ic_scores(period=args.forward_period)
    print("\n=== 因子 IC 评分 ===")
    print(scores.to_string(index=False))

    ic_path = outdir / f"factor_ic_scores_{args.symbol}.csv"
    scores.to_csv(ic_path, index=False, encoding="utf-8-sig")
    print(f"\n因子 IC 评分已保存: {ic_path}")

    # ===== 4. 因子衰减分析（Top 5 因子）=====
    print("\n[FactorSelector] 分析因子 IC 衰减...")
    top5 = scores.head(5)["factor"].tolist()
    decay_rows = []
    for factor in top5:
        decay_df = fs.compute_ic_decay(factor, max_period=20)
        decay_df.insert(0, "factor", factor)
        decay_rows.append(decay_df)

    if decay_rows:
        all_decay = pd.concat(decay_rows, ignore_index=True)
        decay_path = outdir / f"factor_ic_decay_{args.symbol}.csv"
        all_decay.to_csv(decay_path, index=False, encoding="utf-8-sig")
        print(f"因子 IC 衰减已保存: {decay_path}")

    # ===== 5. 分层回测（Top 5 因子）=====
    print(f"\n[FactorEvaluator] 分层回测（前瞻 {args.forward_period} 日）...")
    eval_result = evaluate_all_factors(
        df_factors, top5, forward_period=args.forward_period
    )
    print("\n=== 分层回测结果 ===")
    print(eval_result.to_string(index=False))

    eval_path = outdir / f"factor_quantile_backtest_{args.symbol}.csv"
    eval_result.to_csv(eval_path, index=False, encoding="utf-8-sig")
    print(f"\n分层回测结果已保存: {eval_path}")

    # ===== 6. 自动选择最优因子 =====
    print(f"\n[FactorSelector] 自动筛选最优 {args.top_n} 个因子...")
    best_factors = fs.select_top_factors(
        n=args.top_n, period=args.forward_period, min_ic_abs=0.01
    )
    print(f"最优因子组合: {best_factors}")

    best_path = outdir / f"best_factors_{args.symbol}.txt"
    best_path.write_text("\n".join(best_factors), encoding="utf-8")
    print(f"最优因子已保存: {best_path}")

    # ===== 7. 因子综合报告 =====
    summary = fs.get_factor_summary(period=args.forward_period)
    summary_path = outdir / f"factor_summary_{args.symbol}.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n因子综合报告已保存: {summary_path}")

    print("\n✅ 因子分析完成！")


if __name__ == "__main__":
    main()
