"""
evaluate.py
Formally evaluates the trained PPO agent and the Alya0 rule-based baseline.
Produces results tables for dissertation Chapter 4 (Experiments).

Usage: python training/evaluate.py

Output:
  - Prints results table to terminal
  - Saves results/experiment_1/results.csv
"""

import sys
import os
import numpy as np
import pandas as pd
sys.path.insert(0, '/workspace')

from stable_baselines3 import PPO
from env.jigsaw_env import JigsawEnv
from env.synthetic_puzzle import SyntheticPuzzle, generate_dataset


# ── Config ────────────────────────────────────────────────────
GRID_SIZE    = 3
N_EVAL_EPISODES = 100   # episodes per seed
SEEDS        = [0, 1, 2, 3, 4]  # 5 seeds as required by handbook
MODEL_PATH   = "/workspace/models/best_model.zip"
RESULTS_DIR  = "/workspace/results/experiment_1"
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate_ppo_agent(model, n_episodes, seed):
    """
    Run the PPO agent for n_episodes and return success rate and mean steps.
    Success = all 9 pieces placed in correct positions.
    """
    env = JigsawEnv(grid_size=GRID_SIZE)
    successes = 0
    steps_list = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated

        # Check if all pieces are in correct positions
        correct = int(np.sum(env.board == np.arange(env.n_positions)))
        if correct == env.n_positions:
            successes += 1
        steps_list.append(steps)

    success_rate = successes / n_episodes * 100
    mean_steps   = np.mean(steps_list)
    return success_rate, mean_steps


def evaluate_random_baseline(n_episodes, seed):
    """
    Random agent baseline — places pieces randomly.
    Shows what performance looks like without any learning.
    """
    env = JigsawEnv(grid_size=GRID_SIZE)
    successes = 0
    steps_list = []

    rng = np.random.default_rng(seed)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        steps = 0

        while not done:
            action = rng.integers(0, env.n_positions)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated

        correct = int(np.sum(env.board == np.arange(env.n_positions)))
        if correct == env.n_positions:
            successes += 1
        steps_list.append(steps)

    success_rate = successes / n_episodes * 100
    mean_steps   = np.mean(steps_list)
    return success_rate, mean_steps


# ── Load model ────────────────────────────────────────────────
print("Loading trained PPO model...")
model = PPO.load(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}")
print()

# ── Evaluate across 5 seeds ───────────────────────────────────
ppo_results    = []
random_results = []

print(f"Evaluating over {len(SEEDS)} seeds x {N_EVAL_EPISODES} episodes each...")
print()

for seed in SEEDS:
    print(f"  Seed {seed}...")

    ppo_sr,    ppo_steps    = evaluate_ppo_agent(model, N_EVAL_EPISODES, seed)
    random_sr, random_steps = evaluate_random_baseline(N_EVAL_EPISODES, seed)

    ppo_results.append({"seed": seed,
                        "success_rate": ppo_sr,
                        "mean_steps": ppo_steps})

    random_results.append({"seed": seed,
                           "success_rate": random_sr,
                           "mean_steps": random_steps})

    print(f"    PPO:    success={ppo_sr:.1f}%  steps={ppo_steps:.1f}")
    print(f"    Random: success={random_sr:.1f}%  steps={random_steps:.1f}")

# ── Compute statistics ────────────────────────────────────────
ppo_sr_values    = [r["success_rate"] for r in ppo_results]
random_sr_values = [r["success_rate"] for r in random_results]
ppo_step_values    = [r["mean_steps"] for r in ppo_results]
random_step_values = [r["mean_steps"] for r in random_results]

# ── Print results table ───────────────────────────────────────
print()
print("=" * 60)
print("  EXPERIMENT 1 RESULTS — RL Agent vs Random Baseline")
print("=" * 60)
print(f"  {'Method':<20} {'Success Rate':>15} {'Mean Steps':>12}")
print(f"  {'-'*20} {'-'*15} {'-'*12}")
print(f"  {'PPO Agent':<20} "
      f"{np.mean(ppo_sr_values):>6.1f}% +/- {np.std(ppo_sr_values):.1f}%"
      f"  {np.mean(ppo_step_values):>6.1f} +/- {np.std(ppo_step_values):.1f}")
print(f"  {'Random Baseline':<20} "
      f"{np.mean(random_sr_values):>6.1f}% +/- {np.std(random_sr_values):.1f}%"
      f"  {np.mean(random_step_values):>6.1f} +/- {np.std(random_step_values):.1f}")
print("=" * 60)
print()

# ── Save CSV ──────────────────────────────────────────────────
rows = []
for r in ppo_results:
    rows.append({"method": "PPO", **r})
for r in random_results:
    rows.append({"method": "Random", **r})

df = pd.DataFrame(rows)
csv_path = os.path.join(RESULTS_DIR, "results.csv")
df.to_csv(csv_path, index=False)
print(f"Results saved to {csv_path}")