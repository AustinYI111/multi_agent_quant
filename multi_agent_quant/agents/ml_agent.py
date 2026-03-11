# agents/ml_agent.py
import numpy as np
import pandas as pd


class MLAgent:
    """
    ML Agent（可配置辅助模式或独立交易模式）
    -----------------------------------------------
    auxiliary_mode=True（默认）：
        - 不直接输出 buy/sell，只输出"确认/否决/微调"信息给 Coordinator 使用
        - signal 永远是 "hold"

    auxiliary_mode=False：
        - 允许独立输出 buy/sell 信号
        - 使用更多特征（RSI、MACD histogram、布林带位置、成交量比率、连续涨跌天数）
        - p_up > 0.6 → buy，p_up < 0.4 → sell

    输出格式：
    {
      "signal": "hold" | "buy" | "sell",
      "confidence": float (0~1),
      "meta": {
          "aux": bool,
          "p_up": 0~1,
          "aux_suggest": "buy"|"sell"|"hold",
          "aux_strength": 0~1,
          "reason": ...,
          "features": dict,
      }
    }
    """

    def __init__(
        self,
        lookback: int = 10,
        min_train_size: int = 60,
        prob_threshold: float = 0.55,
        auxiliary_mode: bool = True,     # True=辅助模式，False=独立交易模式
        buy_threshold: float = 0.60,     # 独立模式下 buy 阈值
        sell_threshold: float = 0.40,    # 独立模式下 sell 阈值
    ):
        self.lookback = int(lookback)
        self.min_train_size = int(min_train_size)
        self.prob_threshold = float(prob_threshold)
        self.auxiliary_mode = auxiliary_mode
        self.buy_threshold = float(buy_threshold)
        self.sell_threshold = float(sell_threshold)

    # ---- 内部辅助：计算 RSI ----
    @staticmethod
    def _calc_rsi(close: pd.Series, period: int = 14) -> float:
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
        """正数=连续上涨，负数=连续下跌"""
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

    def _build_features(self, close: pd.Series, data: pd.DataFrame) -> dict:
        """构建多因子特征"""
        lb = min(self.lookback, len(close) - 1)

        # 动量：近 lookback 收益
        ret_lb = (close.iloc[-1] / close.iloc[-1 - lb]) - 1.0 if lb > 0 else 0.0

        # 波动：近 lookback 的收益率 std
        rets = close.pct_change().dropna()
        if len(rets) >= lb and lb > 2:
            vol = float(rets.iloc[-lb:].std())
        elif len(rets) > 2:
            vol = float(rets.std())
        else:
            vol = 0.0

        # 均线偏离：优先用 ma_20，否则自己算
        if "ma_20" in data.columns and not pd.isna(data["ma_20"].iloc[-1]):
            ma = float(data["ma_20"].iloc[-1])
        else:
            ma = float(close.iloc[-min(20, len(close)):].mean())
        dist = (float(close.iloc[-1]) / ma - 1.0) if ma != 0 else 0.0

        # RSI
        rsi = self._calc_rsi(close, 14)
        rsi_norm = (rsi - 50.0) / 50.0  # 归一化到 [-1, 1]

        # MACD histogram（正确计算：MACD线 - Signal线）
        macd_hist = 0.0
        if len(close) >= 35:
            macd_series = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
            signal_series = macd_series.ewm(span=9, adjust=False).mean()
            hist_val = float(macd_series.iloc[-1]) - float(signal_series.iloc[-1])
            # 归一化到 [-0.05, 0.05]
            macd_hist = np.sign(hist_val) * min(abs(hist_val) / (abs(float(close.iloc[-1])) + 1e-8), 0.05)

        # 布林带位置 (0=下轨, 0.5=均值, 1=上轨)
        bb_pos = 0.5
        if len(close) >= 20:
            mean20 = float(close.rolling(20).mean().iloc[-1])
            std20 = float(close.rolling(20).std().iloc[-1])
            if std20 > 0:
                bb_pos = float(np.clip((close.iloc[-1] - (mean20 - 2 * std20)) / (4 * std20), 0.0, 1.0))

        # 成交量比率
        vol_ratio = 0.0
        if "volume" in data.columns:
            vol_ser = pd.to_numeric(data["volume"], errors="coerce")
            if len(vol_ser) >= 20:
                vol_ma20 = float(vol_ser.rolling(20).mean().iloc[-1])
                curr_vol = float(vol_ser.iloc[-1])
                if vol_ma20 > 0:
                    vol_ratio = (curr_vol / vol_ma20) - 1.0

        # 连续涨跌天数（归一化）
        consec = self._calc_consecutive_days(close)
        consec_norm = float(np.clip(consec / 5.0, -1.0, 1.0))

        return {
            "ret_lb": float(ret_lb),
            "dist_ma": float(dist),
            "vol": float(vol),
            "rsi_norm": float(rsi_norm),
            "macd_hist": float(macd_hist),
            "bb_pos": float(bb_pos),
            "vol_ratio": float(vol_ratio),
            "consec_norm": float(consec_norm),
        }

    def generate_signal(self, data: pd.DataFrame) -> dict:
        # 基础检查
        if data is None or "close" not in data.columns:
            return {"signal": "hold", "confidence": 0.0, "meta": {"aux": self.auxiliary_mode, "reason": "missing_close"}}

        close = pd.to_numeric(data["close"], errors="coerce").dropna()
        if len(close) < max(5, self.min_train_size):
            return {"signal": "hold", "confidence": 0.0, "meta": {"aux": self.auxiliary_mode, "reason": "insufficient_data"}}

        # 构建特征
        feats = self._build_features(close, data)

        if self.auxiliary_mode:
            # ---- 辅助模式：3特征 sigmoid（原有行为，保持兼容）----
            x = 3.0 * feats["ret_lb"] + 2.0 * feats["dist_ma"] - 5.0 * feats["vol"]
        else:
            # ---- 独立交易模式：多因子加权 sigmoid ----
            # 正向因子（越高越倾向 buy）
            # RSI 超卖 → buy（rsi_norm < 0）
            # MACD 正 → buy
            # 布林带低位 → buy（bb_pos < 0.5）
            # 成交量放大 + 上涨 → buy
            x = (
                3.0 * feats["ret_lb"]           # 动量（主要权重）
                + 2.0 * feats["macd_hist"]       # MACD 确认
                - 1.5 * feats["rsi_norm"]        # RSI（低 → buy，rsi_norm 负 → buy）
                - 1.0 * (feats["bb_pos"] - 0.5) # 布林带位置（低位 → buy）
                + 1.0 * feats["vol_ratio"] * np.sign(feats["ret_lb"])  # 量能顺势
                + 1.5 * feats["consec_norm"]     # 连续涨跌势
                - 3.0 * feats["vol"]             # 高波动降低置信度
            )

        p_up = 1.0 / (1.0 + np.exp(-x))

        # ---- 输出辅助建议 ----
        aux_suggest = "hold"
        if p_up >= self.prob_threshold:
            aux_suggest = "buy"
        elif p_up <= (1.0 - self.prob_threshold):
            aux_suggest = "sell"

        # 强度：离 0.5 越远越强
        aux_strength = float(min(1.0, abs(p_up - 0.5) * 2.0))

        # ---- 决定最终 signal ----
        if self.auxiliary_mode:
            # 辅助模式：永远 hold
            final_signal = "hold"
        else:
            # 独立交易模式
            if p_up > self.buy_threshold:
                final_signal = "buy"
            elif p_up < self.sell_threshold:
                final_signal = "sell"
            else:
                final_signal = "hold"

        return {
            "signal": final_signal,
            "confidence": round(aux_strength, 4),
            "meta": {
                "aux": self.auxiliary_mode,
                "p_up": round(float(p_up), 4),
                "aux_suggest": aux_suggest,
                "aux_strength": round(aux_strength, 4),
                "features": {k: round(float(v), 4) for k, v in feats.items()},
            },
        }
