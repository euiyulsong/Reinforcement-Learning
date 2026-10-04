import argparse
from pathlib import Path
import csv
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def eps_greedy(Q, s, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(Q.shape[1]))
    return int(np.argmax(Q[s]))


def train(episodes=1000, alpha=0.5, gamma=0.99, epsilon=0.1, seed=42):
    env = gym.make("CliffWalking-v0")
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    returns = []

    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        total = 0.0

        while True:
            a = eps_greedy(Q, s, epsilon, rng)
            ns, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            target = r if done else r + gamma * np.max(Q[ns])
            Q[s, a] += alpha * (target - Q[s, a])

            total += r
            s = ns

            if done:
                break

        returns.append(total)

    env.close()
    return Q, np.asarray(returns)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="results/q_learning")
    args = p.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    Q, returns = train(episodes=args.episodes, seed=args.seed)
    np.save(out / "q_table.npy", Q)

    with open(out / "returns.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "return"])
        for i, x in enumerate(returns):
            w.writerow([i, x])

    plt.figure()
    plt.plot(returns)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.tight_layout()
    plt.savefig(out / "returns.png", dpi=150)
    plt.close()

    print("Q-Learning last-100 mean return:", returns[-100:].mean())


if __name__ == "__main__":
    main()
