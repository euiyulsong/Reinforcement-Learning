import argparse
import csv
import random
from collections import deque
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.asarray(s, dtype=np.float32),
            np.asarray(a, dtype=np.int64),
            np.asarray(r, dtype=np.float32),
            np.asarray(ns, dtype=np.float32),
            np.asarray(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


def train(episodes=300, seed=42, lr=1e-3, gamma=0.99,
          batch_size=64, target_update=500):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("CartPole-v1")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    online = QNetwork(state_dim, action_dim).to(device)
    target = QNetwork(state_dim, action_dim).to(device)
    target.load_state_dict(online.state_dict())

    optimizer = optim.Adam(online.parameters(), lr=lr)
    replay = ReplayBuffer()

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    warmup = 1000
    global_step = 0
    returns = []

    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        total = 0.0

        while True:
            global_step += 1

            if random.random() < epsilon:
                a = env.action_space.sample()
            else:
                st = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    a = int(online(st).argmax(dim=1).item())

            ns, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            replay.push(s, a, r, ns, float(done))

            s = ns
            total += r

            if len(replay) >= max(batch_size, warmup):
                bs, ba, br, bns, bd = replay.sample(batch_size)
                bs = torch.as_tensor(bs, device=device)
                ba = torch.as_tensor(ba, device=device)
                br = torch.as_tensor(br, device=device)
                bns = torch.as_tensor(bns, device=device)
                bd = torch.as_tensor(bd, device=device)

                q = online(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = target(bns).max(dim=1).values
                    y = br + gamma * (1.0 - bd) * next_q

                loss = nn.functional.smooth_l1_loss(q, y)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                optimizer.step()

            if global_step % target_update == 0:
                target.load_state_dict(online.state_dict())

            if done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        returns.append(total)

        if (ep + 1) % 25 == 0:
            print(f"episode={ep+1:4d} return={total:6.1f} "
                  f"avg100={np.mean(returns[-100:]):6.1f} eps={epsilon:.3f}")

    env.close()
    return online, np.asarray(returns), device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="results/dqn")
    args = p.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    model, returns, device = train(args.episodes, args.seed)
    torch.save(model.state_dict(), out / "dqn.pt")

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

    print("Device:", device)
    print("DQN last-100 mean return:", returns[-100:].mean())


if __name__ == "__main__":
    main()
