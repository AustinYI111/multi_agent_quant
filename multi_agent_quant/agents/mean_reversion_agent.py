# agents/mean_reversion_agent.py

import pandas as pd
import numpy as np


class MeanReversionAgent:
    """
    多因子均值回归策略 Agent

    主要信号来源（从高到低优先级）：
    1. RSI 超买超卖（最频繁触发，主要信号源）
    2. 价格偏离 MA20（偏离 ma_deviation_threshold 就开始产生信号）
    3. 连续涨跌天数（连续 ≥3 天下跌 → buy 加分）
    4. 布林带突破（原有逻辑，保留但降低权重）

    输出格式：
        {"signal": "buy" | "sell" | "hold", "confidence": float, "meta": dict}
    """

    def __init__(
        self,
        window: int = 20,
        num_std: float = 2.0,
        conf_scale: float = 3.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        ma_deviation_threshold: float = 0.015,  # 1.5% 偏离就开始触发
        consecutive_days: int = 3,               # 连续涨/跌天数阈值
        min_confidence: float = 0.15,            # 最低置信度门槛
    ):
        self.window = window
        self.num_std = num_std
        self.conf_scale = conf_scale
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.ma_deviation_threshold = ma_deviation_threshold
        self.consecutive_days = consecutive_days
        self.min_confidence = min_confidence

    # ---- 内部辅助：计算 RSI ----
    @staticmethod
    def _calc_rsi(close: pd.Series, period: int) -> float:
        """计算 RSI，不足数据时返回 50（中性）"""
        if len(close) < period + 1:
            return 50.0
        delta = close.diff().dropna()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        loss_val = loss.iloc[-1]
        if loss_val == 0:
            return 100.0
        rs = gain.iloc[-1] / loss_val
        return float(100.0 - 100.0 / (1.0 + rs))

    # ---- 内部辅助：计算连续涨跌天数 ----
    @staticmethod
    def _calc_consecutive_days(close: pd.Series, lookback: int = 10) -> int:
        """
        计算最近连续涨跌天数
        正数 = 连续上涨天数，负数 = 连续下跌天数
        """
        tail = close.tail(lookback).values
        if len(tail) < 2:
            return 0
        direction = 0
        count = 0
        for i in range(len(tail) - 1, 0, -1):
            d = 1 if tail[i] > tail[i - 1] else (-1 if tail[i] < tail[i - 1] else 0)
            if d == 0:
                break
            if direction == 0:
                direction = d
            if d == direction:
                count += 1
            else:
                break
        return direction * count

    def generate_signal(self, data: pd.DataFrame) -> dict:
        """
        输入：包含 'close' 列的 DataFrame
        输出：{"signal": "buy" | "sell" | "hold", "confidence": float, "meta": dict}
        """
        # 1) 数据检查
        if "close" not in data.columns or len(data) < self.window:
            return {
                "signal": "hold",
                "confidence": 0.0,
                "meta": {"reason": "insufficient_data", "window": self.window},
            }

        close = pd.to_numeric(data["close"], errors="coerce")
        if close.isna().iloc[-1]:
            return {
                "signal": "hold",
                "confidence": 0.0,
                "meta": {"reason": "close_nan"},
            }

        # 2) 计算布林带
        rolling_mean = close.rolling(self.window).mean()
        rolling_std = close.rolling(self.window).std()

        mean = rolling_mean.iloc[-1]
        std = rolling_std.iloc[-1]

        if std is None or np.isnan(std) or std == 0 or mean is None or np.isnan(mean):
            return {
                "signal": "hold",
                "confidence": 0.0,
                "meta": {"reason": "std_or_mean_invalid", "mean": mean, "std": std},
            }

        upper = mean + self.num_std * std
        lower = mean - self.num_std * std
        price = close.iloc[-1]
        z = (price - mean) / std

        # 3) RSI 信号（主要信号源）
        rsi_val = self._calc_rsi(close, self.rsi_period)
        rsi_score = 0.0
        rsi_direction = 0
        if rsi_val < self.rsi_oversold:
            # 超卖 → buy
            rsi_direction = 1
            if rsi_val < 20:
                rsi_score = 0.50    # 极度超卖
            else:
                rsi_score = 0.30    # 普通超卖
        elif rsi_val > self.rsi_overbought:
            # 超买 → sell
            rsi_direction = -1
            if rsi_val > 80:
                rsi_score = 0.50    # 极度超买
            else:
                rsi_score = 0.30    # 普通超买

        # 4) 价格偏离均线信号
        deviation = (price - mean) / mean if mean != 0 else 0.0
        dev_score = 0.0
        dev_direction = 0
        abs_dev = abs(deviation)
        if abs_dev > self.ma_deviation_threshold:
            # 偏离越大信号越强，线性映射到 0~0.4
            dev_score = min(0.40, (abs_dev - self.ma_deviation_threshold) / 0.03 * 0.10)
            dev_direction = -1 if deviation > 0 else 1  # 偏高 → sell，偏低 → buy

        # 5) 布林带突破信号（原有逻辑，保留但降低权重）
        bb_score = 0.0
        bb_direction = 0
        if price < lower:
            bb_direction = 1
            bb_score = min(0.30, (-z - self.num_std) / self.conf_scale)
        elif price > upper:
            bb_direction = -1
            bb_score = min(0.30, (z - self.num_std) / self.conf_scale)

        # 6) 连续涨跌天数信号
        consec = self._calc_consecutive_days(close)
        consec_score = 0.0
        consec_direction = 0
        if consec <= -self.consecutive_days:
            # 连续下跌 → buy（均值回归）
            consec_direction = 1
            consec_score = min(0.20, abs(consec) * 0.05)
        elif consec >= self.consecutive_days:
            # 连续上涨 → sell（均值回归）
            consec_direction = -1
            consec_score = min(0.20, abs(consec) * 0.05)

        # 7) 综合各因子：判断方向和总得分
        # 统计各方向的得分
        buy_score = 0.0
        sell_score = 0.0

        if rsi_direction == 1:
            buy_score += rsi_score
        elif rsi_direction == -1:
            sell_score += rsi_score

        if dev_direction == 1:
            buy_score += dev_score
        elif dev_direction == -1:
            sell_score += dev_score

        if bb_direction == 1:
            buy_score += bb_score
        elif bb_direction == -1:
            sell_score += bb_score

        if consec_direction == 1:
            buy_score += consec_score
        elif consec_direction == -1:
            sell_score += consec_score

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
                "window": self.window,
                "num_std": self.num_std,
                "price": round(float(price), 4),
                "mean": round(float(mean), 4),
                "std": round(float(std), 4),
                "zscore": round(float(z), 4),
                "upper_band": round(float(upper), 4),
                "lower_band": round(float(lower), 4),
                "rsi": round(float(rsi_val), 4),
                "deviation": round(float(deviation), 4),
                "consecutive_days": int(consec),
                "scores": {
                    "rsi": round(rsi_score, 4),
                    "deviation": round(dev_score, 4),
                    "bollinger": round(bb_score, 4),
                    "consecutive": round(consec_score, 4),
                    "buy_total": round(buy_score, 4),
                    "sell_total": round(sell_score, 4),
                },
            },
        }
