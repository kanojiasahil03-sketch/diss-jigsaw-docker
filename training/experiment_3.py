"""
experiment_3.py
Experiment 3 — Learning Curve (Training Data Size)

Question: How many training episodes does the agent need to reach
stable performance?

Method:
  - Train PPO agents stopping at different timestep checkpoints
  - Evaluate success rate at each checkpoint
  - Plot learning curve: success rate vs training episodes

Usage: python training/experiment_3.py
"""

import sys
import os
import numpy as np
import pandas as pd
sys.path.insert(0, '/workspace')

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from env.jigsaw_env import JigsawEnv

# ── Config ────────────────────────────────────────────────────
GRID_SIZE       = 3
N_EVAL_EPISODES = 100
SEEDS           = [0, 1, 2, 3, 4]
RESULTS_DIR     = "/workspace/results/experiment_3"
CHECKPOINT_DIR  = "/workspace/results/experiment_3/checkpoints"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Evaluate at these timestep milestones
CHECKPOINTS = [5_000, 10_000, 25_000, 50_000,
               100_000, 200_000, 300_000, 500_000]


def evaluate_model(model, n_episodes, seed):
    """Evaluate a model and return success rate."""
    env = JigsawEnv(grid_size=GRID_SIZE)
    successes = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done   = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        correct = int(np.sum(env.board == np.arange(env.n_positions)))
        if correct == env.n_positions:
            successes += 1

    return successes / n_episodes * 100


def train_and_evaluate_seed(seed):
    """
    Train a fresh PPO agent for one seed, evaluating at each checkpoint.
    Returns list of (timesteps, success_rate) tuples.
    """
    print(f"\n  Seed {seed}:")
    env      = JigsawEnv(grid_size=GRID_SIZE)
    model    = PPO("MlpPolicy", env, verbose=0, seed=seed,
                   learning_rate=3e-4, n_steps=2048,
                   batch_size=64, n_epochs=10)

    results  = []
    trained  = 0

    for checkpoint in CHECKPOINTS:
        # Train up to this checkpoint
        steps_needed = checkpoint - trained
        if steps_needed > 0:
            model.learn(total_timesteps=steps_needed, reset_num_timesteps=False)
            trained = checkpoint

        # Evaluate
        sr = evaluate_model(model, N_EVAL_EPISODES, seed)
        results.append({"timesteps": checkpoint,
                        "success_rate": sr,
                        "seed": seed})
        print(f"    {checkpoint:>8,} steps → {sr:.1f}%")

    return results


# ── Run experiment across all seeds ──────────────────────────
print("=" * 55)
print("  EXPERIMENT 3 — Learning Curve")
print("=" * 55)
print(f"  Checkpoints: {CHECKPOINTS}")
print(f"  Seeds: {SEEDS}")
print(f"  Episodes per eval: {N_EVAL_EPISODES}")
print()

all_results = []

for seed in SEEDS:
    seed_results = train_and_evaluate_seed(seed)
    all_results.extend(seed_results)

# ── Print results table ───────────────────────────────────────
print()
print("=" * 55)
print("  LEARNING CURVE RESULTS")
print("=" * 55)
print(f"  {'Timesteps':>10}  {'Mean SR':>10}  {'Std SR':>8}")
print(f"  {'-'*10}  {'-'*10}  {'-'*8}")

for checkpoint in CHECKPOINTS:
    vals = [r["success_rate"] for r in all_results
            if r["timesteps"] == checkpoint]
    print(f"  {checkpoint:>10,}  "
          f"{np.mean(vals):>8.1f}%  "
          f"+/- {np.std(vals):.1f}%")

print("=" * 55)

# ── Save CSV ──────────────────────────────────────────────────
df = pd.DataFrame(all_results)
csv_path = os.path.join(RESULTS_DIR, "results.csv")
df.to_csv(csv_path, index=False)
print(f"\nResults saved to {csv_path}")