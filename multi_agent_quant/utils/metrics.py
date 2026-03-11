# utils/metrics.py
"""
回测绩效指标计算模块
从 backtest_engine.py 中提取并增强，支持独立使用
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def total_return(equity_curve: List[float]) -> float:
    """总收益率"""
    eq = np.array(equity_curve, dtype=float)
    if len(eq) < 2 or eq[0] == 0:
        return 0.0
    return float(eq[-1] / eq[0] - 1.0)


def annual_return(equity_curve: List[float], trading_days: int = 252) -> float:
    """年化收益率"""
    eq = np.array(equity_curve, dtype=float)
    n = len(eq)
    if n < 2 or eq[0] == 0:
        return 0.0
    tr = eq[-1] / eq[0] - 1.0
    return float((1.0 + tr) ** (trading_days / max(1.0, n - 1)) - 1.0)


def max_drawdown(equity_curve: List[float]) -> float:
    """最大回撤（负数，如 -0.20 表示 20% 回撤）"""
    eq = np.array(equity_curve, dtype=float)
    if len(eq) == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = eq / np.maximum(peak, 1e-12) - 1.0
    return float(dd.min())


def sharpe_ratio(
    equity_curve: List[float], rf: float = 0.0, trading_days: int = 252
) -> float:
    """
    夏普比率（扣除无风险利率）

    Parameters
    ----------
    equity_curve : list of float
    rf : float
        年化无风险利率（默认 0）
    trading_days : int
        年交易日数（默认 252）
    """
    eq = np.array(equity_curve, dtype=float)
    if len(eq) < 3:
        return 0.0
    rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    rf_daily = rf / trading_days
    excess = rets - rf_daily
    if np.std(excess) == 0:
        return 0.0
    return float(np.mean(excess) / np.std(excess) * np.sqrt(trading_days))


def calmar_ratio(equity_curve: List[float], trading_days: int = 252) -> float:
    """Calmar 比率 = 年化收益 / |最大回撤|"""
    ann = annual_return(equity_curve, trading_days)
    mdd = abs(max_drawdown(equity_curve))
    if mdd == 0:
        return 0.0
    return float(ann / mdd)


def _get_trade_action_price(t: Any):
    """从 Trade 对象或 dict 中提取 (action, price)"""
    action = getattr(t, "action", None) or (t.get("action") if isinstance(t, dict) else None)
    price = getattr(t, "price", None) or (t.get("price") if isinstance(t, dict) else None)
    return action, price


def win_rate(trades: List[Any]) -> float:
    """
    胜率（盈利交易占总交易对数的比例）

    Parameters
    ----------
    trades : list
        Trade 对象列表或含 action/price 的 dict 列表

    Returns
    -------
    float: 胜率 ∈ [0, 1]
    """
    buys: List[float] = []
    wins = 0
    total = 0
    for t in trades:
        action, price = _get_trade_action_price(t)
        if action == "buy" and price is not None:
            buys.append(float(price))
        elif action == "sell" and price is not None and buys:
            buy_price = buys.pop(0)
            total += 1
            if float(price) > buy_price:
                wins += 1
    if total == 0:
        return 0.0
    return float(wins / total)


def profit_loss_ratio(trades: List[Any]) -> float:
    """
    盈亏比 = 平均盈利 / 平均亏损（绝对值）

    Parameters
    ----------
    trades : list
        Trade 对象列表或含 action/price 的 dict 列表

    Returns
    -------
    float: 盈亏比，无亏损交易时返回 0.0
    """
    buys: List[float] = []
    profits: List[float] = []
    losses: List[float] = []
    for t in trades:
        action, price = _get_trade_action_price(t)
        if action == "buy" and price is not None:
            buys.append(float(price))
        elif action == "sell" and price is not None and buys:
            buy_price = buys.pop(0)
            pnl = float(price) - buy_price
            if pnl > 0:
                profits.append(pnl)
            elif pnl < 0:
                losses.append(abs(pnl))
    if not losses:
        return 0.0
    avg_profit = float(np.mean(profits)) if profits else 0.0
    avg_loss = float(np.mean(losses))
    if avg_loss == 0:
        return 0.0
    return float(avg_profit / avg_loss)


def max_consecutive_losses(trades: List[Any]) -> int:
    """最大连续亏损次数（按买卖对计算）

    Parameters
    ----------
    trades : list
        Trade 对象列表或含 action/price 的 dict 列表

    Returns
    -------
    int: 最大连续亏损次数
    """
    buys: List[float] = []
    results: List[bool] = []  # True = win, False = loss
    for t in trades:
        action, price = _get_trade_action_price(t)
        if action == "buy" and price is not None:
            buys.append(float(price))
        elif action == "sell" and price is not None and buys:
            buy_price = buys.pop(0)
            results.append(float(price) > buy_price)

    max_consec = 0
    current = 0
    for win in results:
        if not win:
            current += 1
            max_consec = max(max_consec, current)
        else:
            current = 0
    return max_consec


def compute_all_metrics(
    equity_curve: List[float],
    trades: List[Any],
    rf: float = 0.0,
    trading_days: int = 252,
) -> Dict[str, Any]:
    """一次性计算所有回测指标"""
    return {
        "total_return": total_return(equity_curve),
        "annual_return": annual_return(equity_curve, trading_days),
        "max_drawdown": max_drawdown(equity_curve),
        "sharpe": sharpe_ratio(equity_curve, rf, trading_days),
        "calmar": calmar_ratio(equity_curve, trading_days),
        "win_rate": win_rate(trades),
        "profit_loss_ratio": profit_loss_ratio(trades),
        "max_consecutive_losses": max_consecutive_losses(trades),
        "num_trades": len(trades),
    }
