"""
verify_env.py
Run this after building the container to confirm everything is installed.
Usage: python scripts/verify_env.py
"""

import sys

# We store results here: (passed=True/False, label, detail)
checks = []

def check(label, fn):
    """Run fn(), store result. If it crashes, store the error."""
    try:
        result = fn()
        checks.append((True, label, result))
    except Exception as e:
        checks.append((False, label, str(e)))

# ── Check 1: Python version ───────────────────────────────────
# Confirms we're running Python 3.11 inside the container
check("Python version",
      lambda: sys.version)

# ── Check 2: numpy ────────────────────────────────────────────
# Required for observation vectors and results statistics
check("numpy",
      lambda: __import__("numpy").__version__)

# ── Check 3: OpenCV ───────────────────────────────────────────
# Required for the Alya0 baseline (piece detection, edge classification)
check("opencv",
      lambda: __import__("cv2").__version__)

# ── Check 4: Gymnasium ────────────────────────────────────────
# The RL environment API your JigsawEnv will inherit from
check("gymnasium",
      lambda: __import__("gymnasium").__version__)

# ── Check 5: Stable-Baselines3 ───────────────────────────────
# Your PPO training library — the core RL tool from the handbook
check("stable-baselines3",
      lambda: __import__("stable_baselines3").__version__)

# ── Check 6: sb3-contrib ─────────────────────────────────────
# Extends SB3 with MaskablePPO (advanced feature for later)
check("sb3-contrib",
      lambda: __import__("sb3_contrib").__version__)

# ── Check 7: TensorBoard ─────────────────────────────────────
# Live training curve monitoring — explicitly required by handbook
check("tensorboard",
      lambda: __import__("tensorboard").__version__)

# ── Check 8: matplotlib ───────────────────────────────────────
# Plotting training curves and results figures for dissertation
check("matplotlib",
      lambda: __import__("matplotlib").__version__)

# ── Check 9: pandas ───────────────────────────────────────────
# Saving experiment results as CSV tables
check("pandas",
      lambda: __import__("pandas").__version__)

# ── Check 10: JupyterLab ──────────────────────────────────────
# Notebook environment for dissertation figures and analysis
check("jupyterlab",
      lambda: __import__("jupyterlab").__version__)

# ── Check 11: PPO smoke test ──────────────────────────────────
# This is the most important check. It actually trains PPO for
# 500 steps on CartPole (a simple balancing game). If this works,
# your entire RL stack is functioning correctly end-to-end.
def check_ppo():
    from stable_baselines3 import PPO
    import gymnasium as gym
    env = gym.make("CartPole-v1")   # simple built-in env for testing
    model = PPO("MlpPolicy", env, verbose=0)  # verbose=0 = no output spam
    model.learn(total_timesteps=500)           # tiny run, just confirm it works
    return "PPO trained 500 steps on CartPole OK"

check("PPO smoke test", check_ppo)

# ── Print report ──────────────────────────────────────────────
print("\n" + "=" * 55)
print("  CONTAINER VERIFICATION REPORT")
print("=" * 55)

all_passed = True
for passed, label, detail in checks:
    icon = "✅" if passed else "❌"
    print(f"  {icon}  {label:<30}  {detail}")
    if not passed:
        all_passed = False

print("=" * 55)
if all_passed:
    print("  All checks passed — container is ready.\n")
else:
    print("  Some checks FAILED — see errors above.\n")
