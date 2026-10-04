#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"

mkdir -p results

$PYTHON mab.py --steps 1000 --seed 42
$PYTHON dynamic_programming.py --gamma 0.99 --eval-episodes 200 --seed 42
$PYTHON sarsa.py --episodes 200 --seed 42
$PYTHON q_learning.py --episodes 200 --seed 42
$PYTHON dqn.py --episodes 50 --seed 42
$PYTHON sac.py --episodes 15 --seed 42

echo "Quick smoke test finished."
