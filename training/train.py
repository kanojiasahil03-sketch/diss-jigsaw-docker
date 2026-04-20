"""
train.py
Trains a PPO agent on the JigsawEnv.
Usage: python training/train.py
"""

import sys
sys.path.insert(0, '/workspace')

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from env.jigsaw_env import JigsawEnv
import os

# ── Config ────────────────────────────────────────────────────
GRID_SIZE       = 3          # start with 3x3 as handbook says
TOTAL_TIMESTEPS = 500_000    # Stage 1 target
SEED            = 42         # for reproducibility
MODEL_NAME      = f"ppo_jigsaw_{GRID_SIZE}x{GRID_SIZE}"
LOG_DIR         = "/workspace/tb_logs/"
MODEL_DIR       = "/workspace/models/"

# ── Setup ─────────────────────────────────────────────────────
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Environment ───────────────────────────────────────────────
print(f"Creating JigsawEnv {GRID_SIZE}x{GRID_SIZE}...")
env      = JigsawEnv(grid_size=GRID_SIZE)
eval_env = JigsawEnv(grid_size=GRID_SIZE)

# ── Callback — evaluates agent every 10k steps ────────────────
# Saves best model automatically
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=MODEL_DIR,
    log_path=LOG_DIR,
    eval_freq=10_000,
    n_eval_episodes=20,
    deterministic=True,
    verbose=1
)

# ── Model ─────────────────────────────────────────────────────
print("Creating PPO model...")
model = PPO(
    "MlpPolicy",        # fully connected neural network
    env,
    verbose=1,          # print training progress
    seed=SEED,
    tensorboard_log=LOG_DIR,
    learning_rate=3e-4, # default PPO learning rate
    n_steps=2048,       # steps per update
    batch_size=64,      # minibatch size
    n_epochs=10,        # epochs per update
    gamma=0.99,         # discount factor
)

# ── Train ─────────────────────────────────────────────────────
print(f"Training for {TOTAL_TIMESTEPS:,} timesteps...")
print("Open TensorBoard to monitor: docker-compose up tensorboard")
print()

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=eval_callback,
    tb_log_name=MODEL_NAME,
    progress_bar=True
)

# ── Save ──────────────────────────────────────────────────────
save_path = os.path.join(MODEL_DIR, MODEL_NAME)
model.save(save_path)
print(f"\nModel saved to {save_path}.zip")