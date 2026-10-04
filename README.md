# Reinforcement Learning Basics Experiments

기본 강화학습 알고리즘 6개를 동일한 로컬 환경에서 직접 구현하고 비교했다.

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
  ↓ action 선택
K-Armed Bandit
  ↓
Reward
  ↓
Action value Q(a) 업데이트
```

비교 방법:

- ε-greedy
- UCB (Upper Confidence Bound)

### Formula

ε-greedy:

\[
a_t =
\begin{cases}
\text{random action} & \text{with probability } \epsilon \\
\arg\max_a Q_t(a) & \text{otherwise}
\end{cases}
\]

Incremental Q update:

\[
Q_{n+1}(a)
=
Q_n(a)
+
\frac{1}{N(a)}
(R_n-Q_n(a))
\]

UCB:

\[
a_t
=
\arg\max_a
\left[
Q_t(a)
+
c\sqrt{\frac{\ln t}{N_t(a)}}
\right]
\]

### Result

| Method | Mean Reward | Optimal Action Rate | Cumulative Regret |
|---|---:|---:|---:|
| ε-greedy | 0.778 | **0.895** | 710.8 |
| UCB | **0.852** | 0.810 | **341.6** |

UCB가 optimal action 선택률 자체는 낮았지만 exploration을 더 효율적으로 수행하면서 **평균 reward와 cumulative regret에서 더 좋은 결과**를 보였다.

---

## 2. Dynamic Programming

Environment: `FrozenLake-v1`

### Architecture

```text
Known MDP
 ├─ Transition P(s'|s,a)
 └─ Reward R(s,a)
        ↓
Bellman Update
        ↓
Value Function V(s)
        ↓
Policy π(s)
```

비교:

- Value Iteration
- Policy Iteration

### Formula

Bellman optimality equation:

\[
V^*(s)
=
\max_a
\sum_{s'}
P(s'|s,a)
\left[
R(s,a,s')
+
\gamma V^*(s')
\right]
\]

Value Iteration:

\[
V_{k+1}(s)
=
\max_a
\sum_{s'}
P(s'|s,a)
[R+\gamma V_k(s')]
\]

Policy Iteration:

```text
Policy Evaluation
      ↓
Vπ 계산
      ↓
Policy Improvement
      ↓
argmax Qπ(s,a)
      ↓
수렴할 때까지 반복
```

### Result

| Method | Success Rate |
|---|---:|
| Value Iteration | 0.752 |
| Policy Iteration | 0.752 |

```text
Policies identical: True
```

두 알고리즘 모두 동일한 optimal policy로 수렴했고 평가 성공률도 **75.2%**로 동일했다.

---

## 3. SARSA

Environment: `CliffWalking-v1`

### Architecture

```text
State s
 ↓
ε-greedy Policy
 ↓
Action a
 ↓
Environment
 ↓
Reward r, Next State s'
 ↓
다음 실제 Action a' 선택
 ↓
Q(s,a) 업데이트
```

### Formula

\[
Q(s_t,a_t)
\leftarrow
Q(s_t,a_t)
+
\alpha
[
r_{t+1}
+
\gamma Q(s_{t+1},a_{t+1})
-
Q(s_t,a_t)
]
\]

SARSA는 **on-policy** 알고리즘으로, 실제 현재 policy가 다음에 선택할 행동 \(a_{t+1}\)까지 반영한다.

### Result

```text
Last 100 episode mean return: -27.49
```

CliffWalking에서 exploration에 의한 cliff 위험까지 학습하기 때문에 상대적으로 안전한 경로를 선택하는 경향을 보였다.

---

## 4. Q-Learning

Environment: `CliffWalking-v1`

### Architecture

```text
State s
 ↓
ε-greedy Action
 ↓
Environment
 ↓
Reward r, Next State s'
 ↓
max_a Q(s',a)
 ↓
Q(s,a) 업데이트
```

### Formula

\[
Q(s_t,a_t)
\leftarrow
Q(s_t,a_t)
+
\alpha
[
r_{t+1}
+
\gamma
\max_a Q(s_{t+1},a)
-
Q(s_t,a_t)
]
\]

Q-Learning은 **off-policy** 알고리즘으로 다음 행동을 실제 exploration policy가 아니라 greedy action으로 가정한다.

### Result

| Method | Last 100 Mean Return |
|---|---:|
| SARSA | **-27.49** |
| Q-Learning | -51.86 |

이번 실행에서는 SARSA가 Q-Learning보다 높은 return을 보였다.

CliffWalking에서는 Q-Learning이 cliff 근처의 더 짧은 optimal path를 학습할 수 있지만, ε-greedy exploration이 유지되는 동안 cliff 추락 penalty가 발생할 수 있다.

---

## 5. DQN

Environment: `CartPole-v1`

### Architecture

```text
State
 ↓
Q-Network
 ├─ Linear(4 → 128)
 ├─ ReLU
 ├─ Linear(128 → 128)
 ├─ ReLU
 └─ Linear(128 → 2)
 ↓
Q(s,a)
 ↓
ε-greedy Action
 ↓
Environment
 ↓
Replay Buffer
 ↓
Mini-batch Training
 ↓
Target Network
```

DQN의 핵심 구성:

- Neural Q-function
- Experience Replay
- Target Network
- ε-greedy exploration

### Formula

TD target:

\[
y
=
r
+
\gamma(1-d)
\max_{a'}
Q_{\text{target}}(s',a')
\]

Loss:

\[
L(\theta)
=
\mathbb{E}
[
(Q_\theta(s,a)-y)^2
]
\]

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

학습 초반에는 약 20 수준에 머물렀지만 이후 replay buffer가 충분히 쌓이고 Q-network가 학습되면서 return이 지속적으로 상승했다.

---

## 6. Soft Actor-Critic (SAC)

Environment: `Pendulum-v1`

### Architecture

```text
                State
                  ↓
               Actor
          π(a|s), stochastic
                  ↓
                Action
                  ↓
              Environment
                  ↓
       (s, a, r, s') Replay Buffer
                  ↓
        ┌─────────┴─────────┐
        ↓                   ↓
      Critic Q1           Critic Q2
        └─────────┬─────────┘
                  ↓
             min(Q1,Q2)
                  ↓
            Actor Update

             Temperature α
                  ↓
          Entropy Regulation
```

핵심 구성:

- Stochastic Actor
- Twin Q Critics
- Target Critics
- Replay Buffer
- Entropy regularization
- Automatic temperature tuning

### Formula

Critic target:

\[
y
=
r
+
\gamma
\left[
\min(Q_1',Q_2')
-
\alpha \log \pi(a'|s')
\right]
\]

Actor objective:

\[
J_\pi
=
\mathbb{E}
[
\alpha \log\pi(a|s)
-
\min(Q_1(s,a),Q_2(s,a))
]
\]

Entropy-regularized objective:

\[
J(\pi)
=
\sum_t
\mathbb{E}
[
r_t
+
\alpha H(\pi(\cdot|s_t))
]
\]

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

초기 약 `-1300` 수준에서 빠르게 개선되어 후반에는 일부 episode에서 **-1에 가까운 return**을 기록했다.

SAC는 exploration을 entropy term으로 직접 objective에 포함하며 continuous action 환경에서 안정적인 학습을 보였다.

---

# Summary

| Algorithm | Environment | Type | Result |
|---|---|---|
