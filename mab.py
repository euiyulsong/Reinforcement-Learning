import argparse
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


class GaussianBandit:
    def __init__(self, k=10, seed=42):
        self.rng = np.random.default_rng(seed)
        self.k = k
        self.q_true = self.rng.normal(0.0, 1.0, size=k)

    def step(self, action):
        return self.rng.normal(self.q_true[action], 1.0)

    @property
    def optimal_action(self):
        return int(np.argmax(self.q_true))

    @property
    def optimal_value(self):
        return float(np.max(self.q_true))


def epsilon_greedy(env, steps=5000, epsilon=0.1, seed=0):
    rng = np.random.default_rng(seed)
    q = np.zeros(env.k)
    n = np.zeros(env.k)
    rewards, optimal, regret = [], [], []

    for _ in range(steps):
        if rng.random() < epsilon:
            action = int(rng.integers(env.k))
        else:
            action = int(np.argmax(q))

        reward = env.step(action)
        n[action] += 1
        q[action] += (reward - q[action]) / n[action]

        rewards.append(reward)
        optimal.append(action == env.optimal_action)
        regret.append(env.optimal_value - env.q_true[action])

    return np.asarray(rewards), np.asarray(optimal), np.asarray(regret)


def ucb(env, steps=5000, c=2.0):
    q = np.zeros(env.k)
    n = np.zeros(env.k)
    rewards, optimal, regret = [], [], []

    for t in range(steps):
        if t < env.k:
            action = t
        else:
            bonus = c * np.sqrt(np.log(t + 1) / (n + 1e-8))
            action = int(np.argmax(q + bonus))

        reward = env.step(action)
        n[action] += 1
        q[action] += (reward - q[action]) / n[action]

        rewards.append(reward)
        optimal.append(action == env.optimal_action)
        regret.append(env.optimal_value - env.q_true[action])

    return np.asarray(rewards), np.asarray(optimal), np.asarray(regret)


def moving_average(x, window=100):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="results/mab")
    args = p.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    env = GaussianBandit(seed=args.seed)
    eg_r, eg_o, eg_reg = epsilon_greedy(env, args.steps, epsilon=0.1, seed=args.seed)

    env = GaussianBandit(seed=args.seed)
    ucb_r, ucb_o, ucb_reg = ucb(env, args.steps, c=2.0)

    rows = [
        ["epsilon_greedy", eg_r.mean(), eg_o.mean(), eg_reg.sum()],
        ["ucb", ucb_r.mean(), ucb_o.mean(), ucb_reg.sum()],
    ]
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "mean_reward", "optimal_action_rate", "cumulative_regret"])
        w.writerows(rows)

    plt.figure()
    plt.plot(moving_average(eg_r), label="epsilon-greedy")
    plt.plot(moving_average(ucb_r), label="UCB")
    plt.xlabel("Step")
    plt.ylabel("Moving-average reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "reward.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(np.cumsum(eg_reg), label="epsilon-greedy")
    plt.plot(np.cumsum(ucb_reg), label="UCB")
    plt.xlabel("Step")
    plt.ylabel("Cumulative pseudo-regret")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "regret.png", dpi=150)
    plt.close()

    print(rows)


if __name__ == "__main__":
    main()
