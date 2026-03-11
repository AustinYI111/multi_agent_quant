# agents/base_agent.py
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any


class BaseAgent(ABC):
    """所有策略 Agent 的抽象基类"""

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        生成交易信号

        Parameters
        ----------
        data : pd.DataFrame
            包含 'close' 等列的历史行情数据

        Returns
        -------
        dict
            {"signal": "buy"|"sell"|"hold", "confidence": float, "meta": dict}
        """
        pass
