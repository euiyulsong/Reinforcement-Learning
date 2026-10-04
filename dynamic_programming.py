import argparse
from pathlib import Path
import csv
import gymnasium as gym
import numpy as np


def expected_action_value(env, V, s, a, gamma):
    q = 0.0
    for prob, ns, reward, terminated in env.unwrapped.P[s][a]:
        q += prob * (reward + gamma * V[ns] * (not terminated))
    return q


def value_iteration(env, gamma=0.99, theta=1e-10):
    n_s = env.observation_space.n
    n_a = env.action_space.n
    V = np.zeros(n_s)
    iterations = 0

    while True:
        delta = 0.0
        new_v = V.copy()
        for s in range(n_s):
            qs = [expected_action_value(env, V, s, a, gamma) for a in range(n_a)]
            new_v[s] = max(qs)
            delta = max(delta, abs(new_v[s] - V[s]))
        V = new_v
        iterations += 1
        if delta < theta:
            break

    policy = np.zeros(n_s, dtype=int)
    for s in range(n_s):
        qs = [expected_action_value(env, V, s, a, gamma) for a in range(n_a)]
        policy[s] = int(np.argmax(qs))

    return V, policy, iterations


def policy_iteration(env, gamma=0.99, theta=1e-10):
    n_s = env.observation_space.n
    n_a = env.action_space.n
    policy = np.zeros(n_s, dtype=int)
    V = np.zeros(n_s)
    improvement_steps = 0
    eval_sweeps = 0

    while True:
        while True:
            delta = 0.0
            new_v = V.copy()
            for s in range(n_s):
                a = policy[s]
                new_v[s] = expected_action_value(env, V, s, a, gamma)
                delta = max(delta, abs(new_v[s] - V[s]))
            V = new_v
            eval_sweeps += 1
            if delta < theta:
                break

        stable = True
        for s in range(n_s):
            old = policy[s]
            qs = [expected_action_value(env, V, s, a, gamma) for a in range(n_a)]
            policy[s] = int(np.argmax(qs))
            stable &= (old == policy[s])

        improvement_steps += 1
        if stable:
            break

    return V, policy, improvement_steps, eval_sweeps


def evaluate_policy(env, policy, episodes=1000, seed=42):
    returns = []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        total = 0.0
        while True:
            s, r, terminated, truncated, _ = env.step(int(policy[s]))
            total += r
            if terminated or truncated:
                break
        returns.append(total)
    return float(np.mean(returns))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eval-episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="results/dynamic_programming")
    args = p.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

    V_vi, pi_vi, vi_iter = value_iteration(env, gamma=args.gamma)
    V_pi, pi_pi, pi_improve, pi_eval = policy_iteration(env, gamma=args.gamma)

    vi_score = evaluate_policy(env, pi_vi, args.eval_episodes, args.seed)
    pi_score = evaluate_policy(env, pi_pi, args.eval_episodes, args.seed)

    np.savetxt(out / "value_iteration_values.csv", V_vi.reshape(4, 4), delimiter=",")
    np.savetxt(out / "value_iteration_policy.csv", pi_vi.reshape(4, 4), fmt="%d", delimiter=",")
    np.savetxt(out / "policy_iteration_values.csv", V_pi.reshape(4, 4), delimiter=",")
    np.savetxt(out / "policy_iteration_policy.csv", pi_pi.reshape(4, 4), fmt="%d", delimiter=",")

    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "eval_success_rate", "main_iterations", "extra_eval_sweeps"])
        w.writerow(["value_iteration", vi_score, vi_iter, 0])
        w.writerow(["policy_iteration", pi_score, pi_improve, pi_eval])

    print("Value Iteration success:", vi_score)
    print("Policy Iteration success:", pi_score)
    print("Policies identical:", np.array_equal(pi_vi, pi_pi))


if __name__ == "__main__":
    main()
