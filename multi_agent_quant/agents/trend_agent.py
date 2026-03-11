# agents/trend_agent.py
from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd
import numpy as np


@dataclass
class TrendAgent:
    short_window: int = 5
    long_window: int = 20
    trend_window: int = 60          # 长期趋势均线
    macd_fast: int = 12             # MACD 快线周期
    macd_slow: int = 26             # MACD 慢线周期
    macd_signal: int = 9            # MACD 信号线周期
    adx_period: int = 14            # ADX 周期
    adx_threshold: float = 20.0     # ADX 趋势确认阈值（降低到20让更多趋势被识别）
    volume_ratio_threshold: float = 1.2   # 成交量放大倍数阈值
    rsi_period: int = 14            # RSI 周期
    min_conf_when_signal: float = 0.15   # 降低最低置信度门槛
    scale: float = 20.0             # 增大放大系数

    # ---- 内部辅助：计算 RSI ----
    @staticmethod
    def _calc_rsi(close: pd.Series, period: int) -> float:
        """计算 RSI，不足数据时返回 50（中性）"""
        if len(close) < period + 1:
            return 50.0
        delta = close.diff().dropna()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        if loss.iloc[-1] == 0:
            return 100.0
        rs = gain.iloc[-1] / loss.iloc[-1]
        return float(100.0 - 100.0 / (1.0 + rs))

    # ---- 内部辅助：计算 MACD ----
    @staticmethod
    def _calc_macd(close: pd.Series, fast: int, slow: int, signal: int):
        """返回 (macd_line, signal_line)，不足数据时返回 (0, 0)"""
        if len(close) < slow + signal:
            return 0.0, 0.0
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return float(macd_line.iloc[-1]), float(signal_line.iloc[-1])

    # ---- 内部辅助：计算 ADX ----
    @staticmethod
    def _calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> float:
        """计算 ADX，不足数据时返回 0"""
        if len(close) < period * 2:
            return 0.0
        # True Range
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        # +DM / -DM
        up_move = high.diff()
        down_move = (-low.diff())
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = pd.Series(plus_dm, index=close.index)
        minus_dm = pd.Series(minus_dm, index=close.index)
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
        dx_denom = (plus_di + minus_di).replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / dx_denom
        adx = dx.ewm(span=period, adjust=False).mean()
        val = adx.iloc[-1]
        return float(val) if not np.isnan(val) else 0.0

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        # 数据检查
        if data is None or len(data) < self.long_window:
            return {"signal": "hold", "confidence": 0.0, "meta": {"reason": "insufficient_data"}}

        close = data["close"].astype(float)

        # ---- 均线计算 ----
        short_ma = close.rolling(self.short_window).mean()
        long_ma = close.rolling(self.long_window).mean()

        curr_short = float(short_ma.iloc[-1])
        curr_long = float(long_ma.iloc[-1])
        prev_short = float(short_ma.iloc[-2]) if len(close) > 1 else curr_short
        prev_long = float(long_ma.iloc[-2]) if len(close) > 1 else curr_long

        if np.isnan(curr_short) or np.isnan(curr_long) or curr_long == 0:
            return {"signal": "hold", "confidence": 0.0, "meta": {"reason": "nan_ma"}}

        # ---- MA60 长期趋势 ----
        ma60_val = None
        if len(close) >= self.trend_window:
            trend_ma = close.rolling(self.trend_window).mean()
            ma60_val = float(trend_ma.iloc[-1])

        # ---- 基础方向：MA5/MA20 ----
        if curr_short > curr_long:
            direction = 1   # buy 方向
        elif curr_short < curr_long:
            direction = -1  # sell 方向
        else:
            direction = 0

        # 均线相对差距（基础置信度）
        spread = abs(curr_short - curr_long) / abs(curr_long)
        base_conf = min(1.0, spread * self.scale)

        # ---- MACD 确认 ----
        macd_val, macd_sig_val = self._calc_macd(close, self.macd_fast, self.macd_slow, self.macd_signal)
        macd_delta = 0.0
        if macd_val > macd_sig_val and direction == 1:
            macd_delta = 0.20   # MACD 同向确认 buy
        elif macd_val < macd_sig_val and direction == -1:
            macd_delta = 0.20   # MACD 同向确认 sell
        elif macd_val > macd_sig_val and direction == -1:
            macd_delta = -0.10  # MACD 反向减弱 sell
        elif macd_val < macd_sig_val and direction == 1:
            macd_delta = -0.10  # MACD 反向减弱 buy
        # 当均线没有明确方向时，用 MACD 本身判断方向
        if direction == 0:
            if macd_val > macd_sig_val:
                direction = 1
                macd_delta = 0.20
            elif macd_val < macd_sig_val:
                direction = -1
                macd_delta = 0.20

        # ---- ADX 趋势强度 ----
        adx_delta = 0.0
        adx_val = 0.0
        if "high" in data.columns and "low" in data.columns:
            high = data["high"].astype(float)
            low = data["low"].astype(float)
            adx_val = self._calc_adx(high, low, close, self.adx_period)
        if adx_val > self.adx_threshold:
            adx_delta = 0.15  # 趋势确认

        # ---- 成交量确认 ----
        vol_delta = 0.0
        if "volume" in data.columns:
            vol = data["volume"].astype(float)
            vol_ma20 = vol.rolling(20).mean()
            curr_vol = float(vol.iloc[-1])
            curr_vol_ma = float(vol_ma20.iloc[-1])
            if not np.isnan(curr_vol_ma) and curr_vol_ma > 0:
                if curr_vol > curr_vol_ma * self.volume_ratio_threshold:
                    vol_delta = 0.10  # 成交量放大确认

        # ---- RSI 过滤 ----
        rsi_delta = 0.0
        rsi_val = self._calc_rsi(close, self.rsi_period)
        if direction == 1 and rsi_val > 70:
            rsi_delta = -0.15   # 超买区，减弱 buy
        elif direction == -1 and rsi_val < 30:
            rsi_delta = -0.15   # 超卖区，减弱 sell

        # ---- MA60 长期趋势同向 ----
        ma60_delta = 0.0
        curr_price = float(close.iloc[-1])
        if ma60_val is not None and not np.isnan(ma60_val) and ma60_val > 0:
            if curr_price > ma60_val and direction == 1:
                ma60_delta = 0.15   # 长期趋势同向
            elif curr_price < ma60_val and direction == -1:
                ma60_delta = 0.15   # 长期趋势同向

        # ---- 综合打分 ----
        if direction == 0:
            # 没有明确方向
            return {
                "signal": "hold",
                "confidence": 0.0,
                "meta": {"reason": "no_direction", "adx": adx_val, "rsi": rsi_val},
            }

        # 各因子加权求和
        conf = base_conf + macd_delta + adx_delta + vol_delta + rsi_delta + ma60_delta
        conf = float(np.clip(conf, 0.0, 1.0))

        sig = "buy" if direction == 1 else "sell"

        # 确保信号有最低置信度
        if conf < self.min_conf_when_signal:
            conf = self.min_conf_when_signal

        return {
            "signal": sig,
            "confidence": float(conf),
            "meta": {
                "short_window": self.short_window,
                "long_window": self.long_window,
                "prev_short_ma": prev_short,
                "prev_long_ma": prev_long,
                "curr_short_ma": curr_short,
                "curr_long_ma": curr_long,
                "spread": float(spread),
                "macd": round(macd_val, 4),
                "macd_signal": round(macd_sig_val, 4),
                "adx": round(adx_val, 4),
                "rsi": round(rsi_val, 4),
                "ma60": round(ma60_val, 4) if ma60_val is not None else None,
                "factors": {
                    "base": round(base_conf, 4),
                    "macd": round(macd_delta, 4),
                    "adx": round(adx_delta, 4),
                    "volume": round(vol_delta, 4),
                    "rsi": round(rsi_delta, 4),
                    "ma60": round(ma60_delta, 4),
                },
            },
        }
