# agents/momentum_agent.py
"""
动量策略 Agent
核心逻辑：
- 短期动量(5日) vs 中期动量(20日) 的比较
- 突破近期高点/低点
- 成交量加权动量确认
- 资金流向（成交额变化）
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


class MomentumAgent:
    """
    动量策略 Agent

    信号逻辑：
    - 5 日收益率 > 0 且 > 20 日收益率 → 动量 buy
    - 突破近 breakout_window 日最高价 → buy +0.2
    - 成交量持续放大（近 5 日均量 > 近 20 日均量 × 1.3）→ +0.15
    - 反向同理产生 sell 信号

    输出格式：
        {"signal": "buy" | "sell" | "hold", "confidence": float, "meta": dict}
    """

    def __init__(
        self,
        short_momentum: int = 5,
        mid_momentum: int = 20,
        breakout_window: int = 10,   # 近N日高低点突破窗口
        volume_momentum: int = 10,   # 成交量动量对比窗口
        min_confidence: float = 0.15,
    ):
        self.short_momentum = short_momentum
        self.mid_momentum = mid_momentum
        self.breakout_window = breakout_window
        self.volume_momentum = volume_momentum
        self.min_confidence = min_confidence

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        输入：包含 'close' 列的 DataFrame（可选 'volume', 'high', 'low'）
        输出：{"signal": "buy" | "sell" | "hold", "confidence": float, "meta": dict}
        """
        # 1) 数据检查
        if data is None or "close" not in data.columns:
            return {"signal": "hold", "confidence": 0.0, "meta": {"reason": "missing_close"}}

        min_len = max(self.mid_momentum, self.breakout_window) + 1
        if len(data) < min_len:
            return {"signal": "hold", "confidence": 0.0, "meta": {"reason": "insufficient_data"}}

        close = pd.to_numeric(data["close"], errors="coerce")
        if close.isna().iloc[-1]:
            return {"signal": "hold", "confidence": 0.0, "meta": {"reason": "close_nan"}}

        curr_price = float(close.iloc[-1])

        # 2) 短期动量：5 日收益率
        short_ret = 0.0
        if len(close) > self.short_momentum:
            prev_short = float(close.iloc[-1 - self.short_momentum])
            if prev_short != 0:
                short_ret = (curr_price / prev_short) - 1.0

        # 3) 中期动量：20 日收益率
        mid_ret = 0.0
        if len(close) > self.mid_momentum:
            prev_mid = float(close.iloc[-1 - self.mid_momentum])
            if prev_mid != 0:
                mid_ret = (curr_price / prev_mid) - 1.0

        # 4) 确定动量方向和基础得分
        momentum_score = 0.0
        direction = 0

        if short_ret > 0 and short_ret > mid_ret:
            # 短期强于中期 → 正向动量
            direction = 1
            momentum_score = min(0.40, abs(short_ret) * 5.0)
        elif short_ret < 0 and short_ret < mid_ret:
            # 短期弱于中期 → 负向动量
            direction = -1
            momentum_score = min(0.40, abs(short_ret) * 5.0)
        elif short_ret > 0:
            # 短期正但不超过中期
            direction = 1
            momentum_score = min(0.20, abs(short_ret) * 3.0)
        elif short_ret < 0:
            # 短期负但不超过中期
            direction = -1
            momentum_score = min(0.20, abs(short_ret) * 3.0)

        # 5) 突破近期高点/低点
        breakout_score = 0.0
        breakout_direction = 0
        if "high" in data.columns and "low" in data.columns:
            high = pd.to_numeric(data["high"], errors="coerce")
            low = pd.to_numeric(data["low"], errors="coerce")
            # 排除最后一根 K 线本身，取前 breakout_window 根的高低
            # 确保数据长度足够，避免负索引截取过多数据
            if len(high) >= self.breakout_window + 1:
                prev_high_window = high.iloc[-(self.breakout_window + 1):-1]
                prev_low_window = low.iloc[-(self.breakout_window + 1):-1]
            else:
                prev_high_window = high.iloc[:-1]
                prev_low_window = low.iloc[:-1]
            if len(prev_high_window) > 0 and len(prev_low_window) > 0:
                recent_high = float(prev_high_window.max())
                recent_low = float(prev_low_window.min())
                curr_high = float(high.iloc[-1]) if not np.isnan(high.iloc[-1]) else curr_price
                curr_low = float(low.iloc[-1]) if not np.isnan(low.iloc[-1]) else curr_price
                if curr_high > recent_high:
                    breakout_direction = 1
                    breakout_score = 0.20   # 突破近期高点 → buy
                elif curr_low < recent_low:
                    breakout_direction = -1
                    breakout_score = 0.20   # 跌破近期低点 → sell
        else:
            # 没有高低价时用收盘价替代
            prev_close_window = close.iloc[-(self.breakout_window + 1):-1]
            if len(prev_close_window) > 0:
                recent_high = float(prev_close_window.max())
                recent_low = float(prev_close_window.min())
                if curr_price > recent_high:
                    breakout_direction = 1
                    breakout_score = 0.20
                elif curr_price < recent_low:
                    breakout_direction = -1
                    breakout_score = 0.20

        # 6) 成交量动量确认
        vol_score = 0.0
        vol_direction = 0
        if "volume" in data.columns:
            vol = pd.to_numeric(data["volume"], errors="coerce")
            short_vol_ma = float(vol.iloc[-self.short_momentum:].mean()) if len(vol) >= self.short_momentum else 0.0
            long_vol_ma = float(vol.iloc[-self.volume_momentum:].mean()) if len(vol) >= self.volume_momentum else 0.0
            if long_vol_ma > 0 and not np.isnan(short_vol_ma) and not np.isnan(long_vol_ma):
                vol_ratio = short_vol_ma / long_vol_ma
                if vol_ratio > 1.3:
                    # 量能放大，顺动量方向加分
                    vol_score = 0.15
                    vol_direction = direction  # 跟随当前动量方向

        # 7) 综合各因子得分
        buy_score = 0.0
        sell_score = 0.0

        # 动量得分
        if direction == 1:
            buy_score += momentum_score
        elif direction == -1:
            sell_score += momentum_score

        # 突破得分
        if breakout_direction == 1:
            buy_score += breakout_score
        elif breakout_direction == -1:
            sell_score += breakout_score

        # 成交量得分（顺方向加分）
        if vol_direction == 1:
            buy_score += vol_score
        elif vol_direction == -1:
            sell_score += vol_score

        # 8) 确定最终信号
        if buy_score > sell_score and buy_score >= self.min_confidence:
            signal = "buy"
            confidence = float(np.clip(buy_score, 0.0, 1.0))
        elif sell_score > buy_score and sell_score >= self.min_confidence:
            signal = "sell"
            confidence = float(np.clip(sell_score, 0.0, 1.0))
        else:
            signal = "hold"
            confidence = 0.0

        return {
            "signal": signal,
            "confidence": round(confidence, 4),
            "meta": {
                "short_momentum": self.short_momentum,
                "mid_momentum": self.mid_momentum,
                "short_ret": round(float(short_ret), 4),
                "mid_ret": round(float(mid_ret), 4),
                "breakout_direction": breakout_direction,
                "scores": {
                    "momentum": round(momentum_score, 4),
                    "breakout": round(breakout_score, 4),
                    "volume": round(vol_score, 4),
                    "buy_total": round(buy_score, 4),
                    "sell_total": round(sell_score, 4),
                },
            },
        }
