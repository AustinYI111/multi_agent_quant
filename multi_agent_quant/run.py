from agents.data_agent import DataAgent
from agents.trend_agent import TrendAgent
from agents.mean_reversion_agent import MeanReversionAgent
from agents.coordinator_agent import CoordinatorAgent


def main():
    # 1. 数据 Agent
    data_agent = DataAgent("data/sample.csv")
    data_agent.load_data()

    # 2. 策略 Agents
    trend_agent = TrendAgent(short_window=5, long_window=20)
    mean_agent = MeanReversionAgent(window=20)

    # 3. 协调 Agent
    coordinator = CoordinatorAgent(
        agent_weights={
            "trend": 0.5,
            "mean_reversion": 0.5
        }
    )

    # 4. 取一个时间点做决策（demo）
    window_data = data_agent.get_window(end_idx=100, window=50)

    trend_signal = trend_agent.generate_signal(window_data)
    mean_signal = mean_agent.generate_signal(window_data)

    final_signal = coordinator.aggregate({
        "trend": trend_signal,
        "mean_reversion": mean_signal
    })

    print("Final Decision:", final_signal)


if __name__ == "__main__":
    main()
