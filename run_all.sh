#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"

mkdir -p results

echo "========================================"
echo "1/6 MAB"
echo "========================================"
$PYTHON mab.py --steps 5000 --seed 42

echo "========================================"
echo "2/6 Dynamic Programming"
echo "========================================"
$PYTHON dynamic_programming.py --gamma 0.99 --eval-episodes 1000 --seed 42

echo "========================================"
echo "3/6 SARSA"
echo "========================================"
$PYTHON sarsa.py --episodes 1000 --seed 42

echo "========================================"
echo "4/6 Q-Learning"
echo "========================================"
$PYTHON q_learning.py --episodes 1000 --seed 42

echo "========================================"
echo "5/6 DQN"
echo "========================================"
$PYTHON dqn.py --episodes 300 --seed 42

echo "========================================"
echo "6/6 SAC"
echo "========================================"
$PYTHON sac.py --episodes 120 --seed 42

echo
echo "Done. Results are under ./results/"
