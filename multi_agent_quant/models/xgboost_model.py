# models/xgboost_model.py
"""
XGBoost 分类模型封装
用于预测下一日收益率正负（二分类）
若 xgboost 不可用，自动 fallback 到 sklearn GradientBoostingClassifier
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


class XGBoostModel:
    """
    XGBoost 二分类模型（预测明日收益率是否为正）

    特征列（feature_cols）示例：
        ['ret_lb', 'dist_ma', 'vol', 'rsi', 'boll_position']
    目标列（target_col）：
        由 train() 内部自动构造（下一日收益率 > 0）
    """

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        target_col: str = "target",
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.feature_cols = feature_cols or ["return", "ma_5", "ma_20", "vol_20", "rsi"]
        self.target_col = target_col
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

        self._model = None
        self._scaler = None
        self._is_trained = False

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """从 DataFrame 中提取可用特征列（跳过缺失列）"""
        available = [c for c in self.feature_cols if c in df.columns]
        if not available:
            raise ValueError(
                f"No feature columns found. Expected one of {self.feature_cols}, "
                f"got columns: {list(df.columns)}"
            )
        return df[available].copy()

    def train(self, df: pd.DataFrame) -> None:
        """
        训练模型

        Parameters
        ----------
        df : pd.DataFrame
            包含特征列和 close 列的 DataFrame
            目标变量将自动构造：close 明日收益率 > 0
        """
        if "close" not in df.columns:
            raise ValueError("df must contain 'close' column for target construction")

        df = df.copy()
        close = pd.to_numeric(df["close"], errors="coerce")
        # 目标：下一日收益率是否为正
        # 使用 shift(-1) 获取明日价格，最后一行因目标为 NaN 将在下方删除
        df[self.target_col] = (close.shift(-1) > close).astype(int)

        feat_df = self._build_features(df)
        feat_df[self.target_col] = df[self.target_col]

        # 删除含 NaN 的行
        feat_df = feat_df.dropna()
        # 最后一行因为 shift(-1) 会是 NaN
        feat_df = feat_df.iloc[:-1]

        if len(feat_df) < 10:
            raise ValueError(f"Not enough training samples after cleaning: {len(feat_df)}")

        feature_cols_used = [c for c in feat_df.columns if c != self.target_col]
        X = feat_df[feature_cols_used].values
        y = feat_df[self.target_col].values

        # 若只有一个类别（如完全单边行情），无法训练二分类器，跳过训练
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            import warnings
            warnings.warn(
                f"XGBoostModel.train() skipped: target has only one class {unique_classes}. "
                "Likely caused by a fully monotone price series. "
                "predict() will return 0.5 (neutral) until retrained with more diverse data.",
                UserWarning,
                stacklevel=2,
            )
            self._feature_cols_used = feature_cols_used
            self._is_trained = False
            return

        if _XGBOOST_AVAILABLE:
            self._model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                eval_metric="logloss",
            )
            self._model.fit(X, y)
        elif _SKLEARN_AVAILABLE:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)
            self._model = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                random_state=self.random_state,
            )
            self._model.fit(X, y)
        else:
            raise RuntimeError(
                "Neither xgboost nor scikit-learn is available. "
                "Please install one: pip install xgboost  or  pip install scikit-learn"
            )

        self._feature_cols_used = feature_cols_used
        self._is_trained = True

    def predict(self, df: pd.DataFrame) -> float:
        """
        预测上涨概率（标量）

        Returns
        -------
        float: P(up) ∈ [0, 1]
        """
        proba = self.predict_proba(df)
        if proba is None or len(proba) == 0:
            return 0.5
        return float(proba[-1, 1])

    def predict_proba(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        预测概率矩阵

        Returns
        -------
        np.ndarray of shape (n_samples, 2): [[P(down), P(up)], ...]
        """
        if not self._is_trained or self._model is None:
            return None

        feat_df = self._build_features(df)
        # 只使用训练时的特征
        available = [c for c in self._feature_cols_used if c in feat_df.columns]
        if not available:
            return None

        X = feat_df[available].values
        # 替换 NaN
        X = np.nan_to_num(X, nan=0.0)

        if self._scaler is not None:
            X = self._scaler.transform(X)

        return self._model.predict_proba(X)

    @property
    def is_trained(self) -> bool:
        return self._is_trained
