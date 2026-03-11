multi\_agent\_quant/

│

├── README.md                     # 项目说明（后期可写论文简介）

│

├── requirements.txt              # Python 依赖

│

├── config/                       # 配置文件

│   ├── data\_config.yaml          # 数据相关配置

│   ├── agent\_config.yaml         # Agent 参数配置

│   ├── train\_config.yaml         # 训练 / 回测参数

│

├── data/                         # 数据层（DataAgent 管理）

│   ├── raw/                      # 原始行情数据

│   ├── processed/                # 处理后特征数据

│   └── data\_agent.py             # 数据Agent（获取 + 预处理）

│

├── agents/                       # 策略层（核心）

│   ├── \_\_init\_\_.py

│   │

│   ├── base\_agent.py             # 策略Agent抽象基类

│   │

│   ├── trend\_agent.py            # Agent A：趋势跟踪（MA / MACD）

│   ├── mean\_reversion\_agent.py   # Agent B：均值回归（BB / RSI）

│   ├── ml\_agent.py               # Agent C：机器学习预测

│   │

│   └── coordinator\_agent.py      # 协调Agent（策略融合 + 权重分配）

│

├── models/                       # 模型层（给 Agent C 用）

│   ├── \_\_init\_\_.py

│   ├── xgboost\_model.py          # XGBoost 模型

│   ├── lstm\_model.py             # LSTM 模型（可选）

│

├── env/                          # 交易环境

│   ├── \_\_init\_\_.py

│   └── trading\_env.py            # 交易/组合环境（可选）

│

├── utils/                        # 工具函数

│   ├── \_\_init\_\_.py

│   ├── indicators.py             # 技术指标计算

│   ├── metrics.py                # Sharpe、回撤等

│   └── logger.py                 # 日志工具

│

├── backtest/                     # 回测模块

│   ├── \_\_init\_\_.py

│   └── backtest\_engine.py        # 回测引擎

│

├── experiments/                  # 实验入口

│   ├── run\_strategy.py           # 单策略运行

│   ├── run\_multi\_agent.py        # 多Agent融合运行

│   └── run\_backtest.py           # 回测对比实验

│

└── results/                      # 实验结果

&nbsp;   ├── logs/

&nbsp;   ├── figures/

&nbsp;   └── reports/



