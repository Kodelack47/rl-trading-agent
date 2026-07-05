"""
rl_trading_agent.py

School project - a reinforcement learning agent that learns to trade a
simulated commodity (gold again, same as my other assignment, since I
already had the price simulation code and it made sense to reuse it).

The agent uses tabular Q-learning (Watkins & Dayan, 1992) to decide,
day by day, whether to Buy, Sell, or Hold, based on a simplified,
discretized view of recent price movement. No PyTorch/TensorFlow here -
I didn't have a working way to install those or Gymnasium in the
environment I built this in, so this is a from-scratch environment and
agent using just numpy, pandas, and matplotlib.

- Ousmana Tunkara
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(7)

OUTPUT_DIR = "rltrader_output"

# ---------------------------------------------------------
# Step 1: same price simulation idea as my other ML project -
# Geometric Brownian Motion, calibrated to gold's rough real-world
# numbers (10.9% annual return, 15.44% annual volatility)
# ---------------------------------------------------------
def simulate_price_series(num_days=1200, start_price=1800.0):
    annual_return = 0.109
    annual_vol = 0.1544
    daily_drift = annual_return / 252
    daily_vol = annual_vol / np.sqrt(252)

    prices = [start_price]
    for _ in range(1, num_days):
        shock = np.random.normal(daily_drift, daily_vol)
        prices.append(prices[-1] * (1 + shock))
    return np.array(prices)


# ---------------------------------------------------------
# Step 2: the trading environment
# ---------------------------------------------------------
class TradingEnv:
    """
    A simplified single-asset trading environment.

    State: (trend_bucket, position)
      trend_bucket - which bucket the last 5-day return falls into (0-4,
      from a strong drop to a strong rise). Buckets are based on the
      20th/40th/60th/80th percentile of 5-day returns in the price data,
      so they roughly split evenly.
      position - 0 if not holding the asset, 1 if currently holding it

    Actions: 0 = Hold, 1 = Buy, 2 = Sell
      Buy only does something if position is 0. Sell only does something
      if position is 1. Otherwise it just acts like Hold. This keeps the
      agent from "selling" something it doesn't have.

    Reward: the day's percent price change if holding, minus a small
    transaction cost (0.1%) whenever a Buy or Sell actually executes.
    """

    ACTIONS = {0: "Hold", 1: "Buy", 2: "Sell"}
    TRANSACTION_COST = 0.001  # 0.1% per trade, rough approximation

    def __init__(self, prices):
        self.prices = prices
        self.returns_5d = pd.Series(prices).pct_change(5).fillna(0).values
        # bucket edges based on the data itself
        self.bucket_edges = np.percentile(self.returns_5d, [20, 40, 60, 80])

    def _trend_bucket(self, t):
        r = self.returns_5d[t]
        return int(np.digitize(r, self.bucket_edges))  # 0-4

    def reset(self):
        self.t = 5  # skip the first few days since 5-day return needs history
        self.position = 0
        self.entry_price = None
        self.portfolio_log = []
        return (self._trend_bucket(self.t), self.position)

    def step(self, action):
        price_today = self.prices[self.t]
        reward = 0.0
        executed = False

        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = price_today
            reward -= self.TRANSACTION_COST
            executed = True
        elif action == 2 and self.position == 1:  # Sell
            self.position = 0
            reward -= self.TRANSACTION_COST
            self.entry_price = None
            executed = True
        # else: Hold, or an invalid Buy/Sell that just does nothing

        self.t += 1
        done = self.t >= len(self.prices) - 1

        if not done and self.position == 1:
            price_next = self.prices[self.t]
            daily_return = (price_next - price_today) / price_today
            reward += daily_return

        self.portfolio_log.append((self.t, self.position, action, executed))

        next_state = (self._trend_bucket(self.t), self.position) if not done else None
        return next_state, reward, done


# ---------------------------------------------------------
# Step 3: tabular Q-learning agent
# ---------------------------------------------------------
class QLearningAgent:
    def __init__(self, n_trend_buckets=5, n_positions=2, n_actions=3,
                 alpha=0.1, gamma=0.95, epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.q_table = np.zeros((n_trend_buckets, n_positions, n_actions))
        self.alpha = alpha        # learning rate
        self.gamma = gamma        # discount factor
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_actions = n_actions

    def choose_action(self, state):
        trend, position = state
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)  # explore
        return int(np.argmax(self.q_table[trend, position]))  # exploit

    def update(self, state, action, reward, next_state, done):
        trend, position = state
        current_q = self.q_table[trend, position, action]
        if done or next_state is None:
            target = reward
        else:
            next_trend, next_position = next_state
            target = reward + self.gamma * np.max(self.q_table[next_trend, next_position])
        self.q_table[trend, position, action] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ---------------------------------------------------------
# Step 4: training loop
# ---------------------------------------------------------
def train(env, agent, num_episodes=500):
    episode_rewards = []

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state if next_state is not None else state
            total_reward += reward

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if (ep + 1) % 50 == 0:
            avg_last_50 = np.mean(episode_rewards[-50:])
            print(f"Episode {ep + 1}/{num_episodes} - avg reward (last 50): {avg_last_50:.4f} - epsilon: {agent.epsilon:.3f}")

    return episode_rewards


def run_greedy_rollout(env, agent):
    """Run once with no exploration (epsilon=0) to see what the trained policy actually does."""
    state = env.reset()
    done = False
    portfolio_value = 1.0
    values = [portfolio_value]
    actions_taken = []

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        portfolio_value *= (1 + reward)
        values.append(portfolio_value)
        actions_taken.append(action)
        state = next_state if next_state is not None else state

    agent.epsilon = old_epsilon
    return values, actions_taken


def buy_and_hold_baseline(prices, start_t=5):
    return prices[start_t:] / prices[start_t]


def random_policy_rollout(env, seed=123):
    """A random-action baseline, just to see how much worse pure luck does
    compared to the trained agent and buy-and-hold."""
    rng = np.random.RandomState(seed)
    state = env.reset()
    done = False
    portfolio_value = 1.0
    values = [portfolio_value]

    while not done:
        action = rng.randint(3)
        next_state, reward, done = env.step(action)
        portfolio_value *= (1 + reward)
        values.append(portfolio_value)
        state = next_state if next_state is not None else state

    return values


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Simulating gold-like price data...")
    prices = simulate_price_series()

    env = TradingEnv(prices)
    agent = QLearningAgent()

    print("Training Q-learning agent...")
    episode_rewards = train(env, agent, num_episodes=500)

    print("\nRunning greedy rollout with trained policy (epsilon=0)...")
    values, actions_taken = run_greedy_rollout(env, agent)
    baseline = buy_and_hold_baseline(prices)
    random_values = random_policy_rollout(env)

    agent_return = (values[-1] - 1) * 100
    baseline_return = (baseline[-1] - 1) * 100
    random_return = (random_values[-1] - 1) * 100

    buy_count = actions_taken.count(1)
    sell_count = actions_taken.count(2)
    hold_count = actions_taken.count(0)

    print(f"\n--- Results ---")
    print(f"Agent total return:       {agent_return:.2f}%")
    print(f"Buy-and-hold return:      {baseline_return:.2f}%")
    print(f"Random-action return:     {random_return:.2f}%")
    print(f"Actions taken - Buy: {buy_count}, Sell: {sell_count}, Hold: {hold_count}")
    print(f"Final Q-table:\n{np.round(agent.q_table, 3)}")

    # ---- chart 1: learning curve ----
    plt.figure(figsize=(9, 5))
    plt.plot(episode_rewards, alpha=0.4, label="Episode reward")
    window = 20
    smoothed = pd.Series(episode_rewards).rolling(window).mean()
    plt.plot(smoothed, label=f"{window}-episode moving average", linewidth=2)
    plt.title("Q-Learning Trading Agent - Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total reward (approx. return)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "learning_curve.png"), dpi=150)
    plt.close()

    # ---- chart 2: portfolio value vs buy-and-hold vs random ----
    plt.figure(figsize=(9, 5))
    plt.plot(values, label="Q-learning agent")
    plt.plot(baseline, label="Buy-and-hold baseline", linestyle="--")
    plt.plot(random_values, label="Random-action agent", linestyle=":", color="gray")
    plt.title("Portfolio Value - Trained Agent vs. Buy-and-Hold vs. Random")
    plt.xlabel("Trading day")
    plt.ylabel("Portfolio value (starting at 1.0)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "portfolio_vs_baseline.png"), dpi=150)
    plt.close()

    print(f"\nCharts saved to the '{OUTPUT_DIR}' folder.")
    print("Done!")


if __name__ == "__main__":
    main()
