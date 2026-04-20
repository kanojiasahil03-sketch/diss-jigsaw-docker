"""
experiment_2.py
Experiment 2 — Irregular Shape Generalisation

Tests whether the PPO agent maintains its advantage when puzzle pieces
have ambiguous or noisy edge classifications.

This is the hardest case for rule-based methods and the most important
finding in the dissertation.

How we simulate irregular/ambiguous edges:
  - Add random noise to edge classifications
  - With probability `noise_level`, flip an edge type randomly
  - This simulates real-world pieces where edges are hard to classify

Usage: python training/experiment_2.py
"""

import sys
import os
import numpy as np
import pandas as pd
sys.path.insert(0, '/workspace')

from stable_baselines3 import PPO
from env.jigsaw_env import JigsawEnv

# ── Config ────────────────────────────────────────────────────
GRID_SIZE       = 3
N_EVAL_EPISODES = 100
SEEDS           = [0, 1, 2, 3, 4]
MODEL_PATH      = "/workspace/models/best_model.zip"
RESULTS_DIR     = "/workspace/results/experiment_2"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Noise levels to test
# 0.0 = clean (baseline)
# 0.1 = 10% of edges randomly flipped (mild noise)
# 0.2 = 20% of edges randomly flipped (moderate noise)
# 0.3 = 30% of edges randomly flipped (heavy noise)
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3]


class NoisyJigsawEnv(JigsawEnv):
    """
    Extends JigsawEnv with edge noise to simulate irregular pieces.
    With probability noise_level, each edge classification is randomly
    changed to a different type — simulating ambiguous or irregular edges.
    """

    def __init__(self, grid_size=3, noise_level=0.0):
        super().__init__(grid_size=grid_size)
        self.noise_level = noise_level

    def _add_noise(self, pieces):
        """Randomly flip edge classifications with probability noise_level."""
        noisy = pieces.copy()
        for i in range(len(noisy)):
            for j in range(4):  # 4 edges per piece
                if np.random.random() < self.noise_level:
                    # Replace with a random different edge type
                    current = noisy[i][j]
                    options = [e for e in [0, 1, 2] if e != current]
                    noisy[i][j] = np.random.choice(options)
        return noisy

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # Add noise to piece edge classifications after reset
        if self.noise_level > 0:
            self.pieces = self._add_noise(self.pieces)
            # Rebuild observation with noisy edges
            obs = self._get_obs()
        return obs, info


def evaluate_agent(model, env_class, noise_level, n_episodes, seed):
    """Evaluate PPO agent on environment with given noise level."""
    env = env_class(grid_size=GRID_SIZE, noise_level=noise_level)
    successes  = 0
    steps_list = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done   = False
        steps  = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated

        correct = int(np.sum(env.board == np.arange(env.n_positions)))
        if correct == env.n_positions:
            successes += 1
        steps_list.append(steps)

    return successes / n_episodes * 100, np.mean(steps_list)


def evaluate_random(noise_level, n_episodes, seed):
    """Random agent baseline for comparison."""
    env = NoisyJigsawEnv(grid_size=GRID_SIZE, noise_level=noise_level)
    successes  = 0
    steps_list = []
    rng        = np.random.default_rng(seed)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done   = False
        steps  = 0

        while not done:
            action = rng.integers(0, env.n_positions)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated

        correct = int(np.sum(env.board == np.arange(env.n_positions)))
        if correct == env.n_positions:
            successes += 1
        steps_list.append(steps)

    return successes / n_episodes * 100, np.mean(steps_list)


# ── Load model ────────────────────────────────────────────────
print("Loading trained PPO model...")
model = PPO.load(MODEL_PATH)
print("Model loaded.\n")

# ── Run experiment ────────────────────────────────────────────
all_results = []

for noise in NOISE_LEVELS:
    print(f"Noise level: {int(noise*100)}%")
    ppo_sr_list    = []
    random_sr_list = []

    for seed in SEEDS:
        ppo_sr, ppo_steps       = evaluate_agent(
            model, NoisyJigsawEnv, noise, N_EVAL_EPISODES, seed)
        random_sr, random_steps = evaluate_random(
            noise, N_EVAL_EPISODES, seed)

        ppo_sr_list.append(ppo_sr)
        random_sr_list.append(random_sr)

        all_results.append({
            "noise_level": noise,
            "method": "PPO",
            "seed": seed,
            "success_rate": ppo_sr,
        })
        all_results.append({
            "noise_level": noise,
            "method": "Random",
            "seed": seed,
            "success_rate": random_sr,
        })

    print(f"  PPO:    {np.mean(ppo_sr_list):.1f}% +/- {np.std(ppo_sr_list):.1f}%")
    print(f"  Random: {np.mean(random_sr_list):.1f}% +/- {np.std(random_sr_list):.1f}%")
    print()

# ── Print results table ───────────────────────────────────────
print("=" * 65)
print("  EXPERIMENT 2 — Irregular Shape Generalisation")
print("=" * 65)
print(f"  {'Noise':>8}  {'PPO Success Rate':>20}  {'Random Success Rate':>20}")
print(f"  {'-'*8}  {'-'*20}  {'-'*20}")

for noise in NOISE_LEVELS:
    ppo_vals = [r["success_rate"] for r in all_results
                if r["noise_level"] == noise and r["method"] == "PPO"]
    rnd_vals = [r["success_rate"] for r in all_results
                if r["noise_level"] == noise and r["method"] == "Random"]

    print(f"  {int(noise*100):>7}%  "
          f"{np.mean(ppo_vals):>6.1f}% +/- {np.std(ppo_vals):.1f}%        "
          f"{np.mean(rnd_vals):>6.1f}% +/- {np.std(rnd_vals):.1f}%")

print("=" * 65)

# ── Save CSV ──────────────────────────────────────────────────
df = pd.DataFrame(all_results)
csv_path = os.path.join(RESULTS_DIR, "results.csv")
df.to_csv(csv_path, index=False)
print(f"\nResults saved to {csv_path}")