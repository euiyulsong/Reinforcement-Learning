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


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class ReplayBuffer:
    def __init__(self, capacity=1_000_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.asarray(s, dtype=np.float32),
            np.asarray(a, dtype=np.float32),
            np.asarray(r, dtype=np.float32),
            np.asarray(ns, dtype=np.float32),
            np.asarray(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


def mlp(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, out_dim),
    )


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_scale, action_bias):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.register_buffer("action_scale", torch.as_tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(action_bias, dtype=torch.float32))

    def forward(self, s):
        h = self.backbone(s)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, s):
        mu, log_std = self(s)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        x = normal.rsample()
        y = torch.tanh(x)
        action = y * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x)
        log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        mean_action = torch.tanh(mu) * self.action_scale + self.action_bias
        return action, log_prob, mean_action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1 = mlp(state_dim + action_dim, 1)
        self.q2 = mlp(state_dim + action_dim, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        return self.q1(x), self.q2(x)


def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def train(episodes=120, seed=42, gamma=0.99, tau=0.005,
          lr=3e-4, batch_size=256, start_steps=1000,
          updates_after=1000):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("Pendulum-v1")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    high = env.action_space.high
    low = env.action_space.low
    scale = (high - low) / 2.0
    bias = (high + low) / 2.0

    actor = Actor(state_dim, action_dim, scale, bias).to(device)
    critic = Critic(state_dim, action_dim).to(device)
    critic_target = Critic(state_dim, action_dim).to(device)
    critic_target.load_state_dict(critic.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=lr)
    critic_opt = optim.Adam(critic.parameters(), lr=lr)

    target_entropy = -float(action_dim)
    log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
    alpha_opt = optim.Adam([log_alpha], lr=lr)

    replay = ReplayBuffer()
    returns = []
    total_steps = 0

    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        total = 0.0

        while True:
            total_steps += 1

            if total_steps <= start_steps:
                a = env.action_space.sample()
            else:
                st = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    a, _, _ = actor.sample(st)
                a = a.squeeze(0).cpu().numpy()

            ns, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            replay.push(s, a, r, ns, float(done))
            s = ns
            total += r

            if len(replay) >= max(batch_size, updates_after):
                bs, ba, br, bns, bd = replay.sample(batch_size)
                bs = torch.as_tensor(bs, dtype=torch.float32, device=device)
                ba = torch.as_tensor(ba, dtype=torch.float32, device=device)
                br = torch.as_tensor(br, dtype=torch.float32, device=device).unsqueeze(1)
                bns = torch.as_tensor(bns, dtype=torch.float32, device=device)
                bd = torch.as_tensor(bd, dtype=torch.float32, device=device).unsqueeze(1)

                alpha = log_alpha.exp().detach()

                with torch.no_grad():
                    next_a, next_logp, _ = actor.sample(bns)
                    tq1, tq2 = critic_target(bns, next_a)
                    target_q = torch.min(tq1, tq2) - alpha * next_logp
                    y = br + gamma * (1.0 - bd) * target_q

                q1, q2 = critic(bs, ba)
                critic_loss = nn.functional.mse_loss(q1, y) + nn.functional.mse_loss(q2, y)

                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()

                new_a, logp, _ = actor.sample(bs)
                q1_pi, q2_pi = critic(bs, new_a)
                q_pi = torch.min(q1_pi, q2_pi)
                actor_loss = (alpha * logp - q_pi).mean()

                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()

                alpha_loss = -(log_alpha * (logp + target_entropy).detach()).mean()
                alpha_opt.zero_grad()
                alpha_loss.backward()
                alpha_opt.step()

                soft_update(critic_target, critic, tau)

            if done:
                break

        returns.append(total)

        if (ep + 1) % 10 == 0:
            print(f"episode={ep+1:4d} return={total:8.1f} "
                  f"avg10={np.mean(returns[-10:]):8.1f} "
                  f"alpha={log_alpha.exp().item():.3f}")

    env.close()
    return actor, critic, np.asarray(returns), device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="results/sac")
    args = p.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    actor, critic, returns, device = train(args.episodes, args.seed)
    torch.save(actor.state_dict(), out / "actor.pt")
    torch.save(critic.state_dict(), out / "critic.pt")

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
    print("SAC last-10 mean return:", returns[-10:].mean())


if __name__ == "__main__":
    main()
