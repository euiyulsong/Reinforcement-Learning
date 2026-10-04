# Reinforcement Learning Basics Experiments

기본 강화학습 알고리즘 6개를 직접 구현하고 로컬에서 실행했다.

- Multi-Armed Bandit
- Dynamic Programming
- SARSA
- Q-Learning
- DQN
- SAC

---

## 1. Multi-Armed Bandit

### Architecture

```text
Agent
  ↓
Action Selection
  ↓
K-Armed Bandit
  ↓
Reward
  ↓
Update Q(a)
```

비교:

- Epsilon-Greedy
- UCB (Upper Confidence Bound)

### Formula

Epsilon-Greedy:

```text
with probability epsilon:
    choose random action

otherwise:
    a = argmax_a Q(a)
```

Incremental action-value update:

```text
Q(a) <- Q(a) + (1 / N(a)) * (R - Q(a))
```

UCB:

```text
a = argmax_a [
    Q(a) + c * sqrt(log(t) / N(a))
]
```

### Result

| Method | Mean Reward | Optimal Action Rate | Cumulative Regret |
|---|---:|---:|---:|
| Epsilon-Greedy | 0.778 | **0.895** | 710.8 |
| UCB | **0.852** | 0.810 | **341.6** |

UCB는 optimal action 선택 비율 자체는 낮았지만 exploration을 더 효율적으로 수행해 **평균 reward와 cumulative regret에서 더 좋은 결과**를 보였다.

---

## 2. Dynamic Programming

Environment:

```text
FrozenLake-v1
```

### Architecture

```text
Known MDP
 ├─ Transition Probability P(s' | s, a)
 └─ Reward R
          ↓
     Bellman Update
          ↓
     Value Function V(s)
          ↓
        Policy
```

비교:

- Value Iteration
- Policy Iteration

### Formula

Bellman optimality equation:

```text
V*(s) =
    max_a Σ_s' P(s' | s, a)
    * [R(s, a, s') + gamma * V*(s')]
```

Value Iteration:

```text
V_next(s) =
    max_a Σ_s' P(s' | s, a)
    * [R + gamma * V(s')]
```

Policy Iteration:

```text
Policy Evaluation
      ↓
Compute V_pi(s)
      ↓
Policy Improvement
      ↓
pi(s) = argmax_a Q_pi(s, a)
      ↓
Repeat until stable
```

### Result

| Method | Success Rate |
|---|---:|
| Value Iteration | 0.752 |
| Policy Iteration | 0.752 |

```text
Policies identical: True
```

두 방법 모두 동일한 policy로 수렴했으며 평가 성공률은 **75.2%**였다.

---

## 3. SARSA

Environment:

```text
CliffWalking-v1
```

### Architecture

```text
State s
  ↓
Epsilon-Greedy Policy
  ↓
Action a
  ↓
Environment
  ↓
Reward r + Next State s'
  ↓
Choose Next Action a'
  ↓
Update Q(s, a)
```

### Formula

```text
TD Target =
    r + gamma * Q(s', a')
```

```text
Q(s, a) <-
    Q(s, a)
    + alpha * [
        r
        + gamma * Q(s', a')
        - Q(s, a)
    ]
```

SARSA는 **on-policy TD control**이다.

즉, 현재 epsilon-greedy policy가 실제로 선택할 다음 action까지 포함해서 학습한다.

### Result

```text
Last 100 mean return: -27.49
```

CliffWalking에서는 exploration 중 cliff에 떨어질 가능성까지 반영하기 때문에 상대적으로 안전한 경로를 학습하는 경향이 있다.

---

## 4. Q-Learning

Environment:

```text
CliffWalking-v1
```

### Architecture

```text
State s
  ↓
Epsilon-Greedy Action
  ↓
Environment
  ↓
Reward r + Next State s'
  ↓
max_a Q(s', a)
  ↓
Update Q(s, a)
```

### Formula

TD Target:

```text
TD Target =
    r + gamma * max_a Q(s', a)
```

Q update:

```text
Q(s, a) <-
    Q(s, a)
    + alpha * [
        r
        + gamma * max_a Q(s', a)
        - Q(s, a)
    ]
```

Q-Learning은 **off-policy TD control**이다.

실제 다음 action 대신 항상 greedy한 future action을 기준으로 target을 계산한다.

### Result

| Method | Last 100 Mean Return |
|---|---:|
| SARSA | **-27.49** |
| Q-Learning | -51.86 |

이번 실행에서는 SARSA가 더 높은 return을 기록했다.

CliffWalking에서 Q-Learning은 더 짧은 cliff 근처 경로를 학습할 수 있지만, epsilon-greedy exploration이 계속되는 동안 cliff penalty가 발생할 수 있다.

---

## 5. DQN

Environment:

```text
CartPole-v1
```

### Architecture

```text
State
  ↓
Q Network
  ├─ Linear(4 -> 128)
  ├─ ReLU
  ├─ Linear(128 -> 128)
  ├─ ReLU
  └─ Linear(128 -> 2)
  ↓
Q(s, a)
  ↓
Epsilon-Greedy Action
  ↓
Environment
  ↓
Transition
  ↓
Replay Buffer
  ↓
Mini-batch Sampling
  ↓
Online Q Network Update
  ↓
Periodic Target Network Update
```

핵심 구성:

- Neural Q-function
- Experience Replay
- Target Network
- Epsilon-Greedy Exploration

### Formula

DQN target:

```text
y =
    r
    + gamma * (1 - done)
    * max_a' Q_target(s', a')
```

Prediction:

```text
Q_pred = Q_online(s, a)
```

Loss:

```text
Loss =
    SmoothL1Loss(Q_pred, y)
```

일반적인 MSE 형태로 쓰면:

```text
Loss =
    E[(Q_online(s, a) - y)^2]
```

### Result

```text
Episode  50 : avg100 = 20.7
Episode 100 : avg100 = 20.4
Episode 150 : avg100 = 24.2
Episode 200 : avg100 = 43.9
Episode 250 : avg100 = 69.8
Episode 300 : avg100 = 99.7
```

Final:

```text
Last 100 mean return: 99.66
Device: CUDA
```

초기에는 평균 return이 약 20 수준이었지만 이후 replay buffer와 Q-network 학습이 진행되면서 지속적으로 성능이 상승했다.

---

## 6. Soft Actor-Critic

Environment:

```text
Pendulum-v1
```

### Architecture

```text
                State
                  ↓
                Actor
              pi(a | s)
                  ↓
                Action
                  ↓
              Environment
                  ↓
           (s, a, r, s')
                  ↓
            Replay Buffer
                  ↓
        ┌─────────┴─────────┐
        ↓                   ↓
      Critic Q1           Critic Q2
        └─────────┬─────────┘
                  ↓
             min(Q1, Q2)
                  ↓
              Actor Update

        Temperature alpha
                  ↓
          Entropy Regulation
```

핵심 구성:

- Stochastic Actor
- Twin Critics
- Target Critics
- Replay Buffer
- Entropy Regularization
- Automatic Temperature Tuning

### Formula

Critic target:

```text
y =
    r
    + gamma * (1 - done)
    * [
        min(Q1_target(s', a'), Q2_target(s', a'))
        - alpha * log pi(a' | s')
      ]
```

Critic loss:

```text
Loss_Q =
    MSE(Q1(s, a), y)
    + MSE(Q2(s, a), y)
```

Actor objective:

```text
Loss_actor =
    E[
        alpha * log pi(a | s)
        - min(Q1(s, a), Q2(s, a))
    ]
```

Entropy-regularized objective:

```text
maximize:
    E[
        reward
        + alpha * entropy
    ]
```

Temperature alpha는 exploration 정도를 조절한다.

```text
large alpha
-> more entropy
-> more exploration

small alpha
-> more deterministic policy
-> more exploitation
```

### Result

```text
Episode  10 : avg10 = -1303.8
Episode  20 : avg10 = -1078.0
Episode  30 : avg10 =  -365.2
Episode  40 : avg10 =  -150.2
Episode  80 : avg10 =  -192.8
Episode 100 : avg10 =  -180.4
Episode 110 : return = -0.8
Episode 120 : return = -1.0
```

Final:

```text
Last 10 mean return: -130.44
Device: CUDA
```

초기 평균 return 약 `-1300`에서 빠르게 개선되었으며, 후반 일부 episode에서는 `-1`에 가까운 return까지 도달했다.

---

# Summary

| Algorithm | Environment | Category | Result |
|---|---|---|---:|
| Epsilon-Greedy | Gaussian Bandit | Bandit | Reward 0.778 |
| UCB | Gaussian Bandit | Bandit | **Reward 0.852** |
| Value Iteration | FrozenLake | Dynamic Programming | Success 0.752 |
| Policy Iteration | FrozenLake | Dynamic Programming | Success 0.752 |
| SARSA | CliffWalking | On-Policy TD | **-27.49** |
| Q-Learning | CliffWalking | Off-Policy TD | -51.86 |
| DQN | CartPole | Deep Value-Based | 99.66 |
| SAC | Pendulum | Actor-Critic | -130.44 |

## Overall Flow

```text
Multi-Armed Bandit
        ↓
Dynamic Programming
        ↓
Tabular Temporal Difference Learning
   ├─ SARSA
   └─ Q-Learning
        ↓
Deep Reinforcement Learning
   ├─ DQN
   │    └─ Discrete Action Space
   │
   └─ SAC
        └─ Continuous Action Space
```

이번 실험은 다음 강화학습 개념의 흐름을 직접 구현하는 것을 목표로 했다.

```text
Exploration vs Exploitation
        ↓
Bellman Equation
        ↓
Temporal Difference Learning
        ↓
Value-Based Deep RL
        ↓
Actor-Critic + Entropy Regularization
```
