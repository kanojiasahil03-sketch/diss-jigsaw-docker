"""
visualiser.py
2D real-time visualisation of the PPO agent solving a jigsaw puzzle.
Shows the board updating step by step as the agent places each piece.

Usage: python env/visualiser.py
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
sys.path.insert(0, '/workspace')

from stable_baselines3 import PPO
from env.jigsaw_env import JigsawEnv

# ── Config ────────────────────────────────────────────────────
MODEL_PATH  = "/home/sahil/diss-jigsaw-docker/models/best_model.zip"
GRID_SIZE   = 3
DELAY       = 0.8   # seconds between each placement (slow enough to watch)

# ── Colours ───────────────────────────────────────────────────
COLOURS = [
    "#E63946",  # red
    "#2A9D8F",  # teal
    "#E9C46A",  # yellow
    "#F4A261",  # orange
    "#457B9D",  # blue
    "#A8DADC",  # light blue
    "#6A4C93",  # purple
    "#2DC653",  # green
    "#FF6B6B",  # coral
]
EMPTY_COLOUR    = "#F0F0F0"
CORRECT_COLOUR  = "#2DC653"
WRONG_COLOUR    = "#E63946"
GRID_COLOUR     = "#333333"

EDGE_SYMBOLS = {0: "▲", 1: "▼", 2: "─"}  # head, hole, straight
EDGE_NAMES   = {0: "HEAD", 1: "HOLE", 2: "STRAIGHT"}


def draw_board(ax_board, ax_info, ax_legend, board, pieces,
               current_piece_idx, piece_order, step, reward_total,
               last_action, last_correct, grid_size):
    """Redraw the board and info panel."""

    ax_board.clear()
    ax_info.clear()
    ax_legend.clear()

    n = grid_size
    ax_board.set_xlim(0, n)
    ax_board.set_ylim(0, n)
    ax_board.set_aspect('equal')
    ax_board.axis('off')
    ax_board.set_title("Puzzle Board", fontsize=13, fontweight='bold', pad=10)

    # Draw each cell
    for pos in range(n * n):
        row = pos // n
        col = pos % n
        # Flip row so row 0 is at top
        draw_row = (n - 1) - row

        if board[pos] == -1:
            # Empty cell
            colour = EMPTY_COLOUR
            label  = ""
        else:
            piece_idx = board[pos]
            correct   = (piece_idx == pos)
            colour    = CORRECT_COLOUR if correct else WRONG_COLOUR
            label     = f"P{piece_idx}"

        rect = FancyBboxPatch(
            (col + 0.05, draw_row + 0.05),
            0.90, 0.90,
            boxstyle="round,pad=0.02",
            facecolor=colour,
            edgecolor=GRID_COLOUR,
            linewidth=2
        )
        ax_board.add_patch(rect)

        if label:
            ax_board.text(
                col + 0.5, draw_row + 0.5, label,
                ha='center', va='center',
                fontsize=14, fontweight='bold', color='white'
            )

        # Position label (small, bottom right of cell)
        ax_board.text(
            col + 0.85, draw_row + 0.12, f"{pos}",
            ha='center', va='center',
            fontsize=7, color='#666666'
        )

    # ── Info panel ────────────────────────────────────────────
    ax_info.axis('off')
    ax_info.set_title("Agent Info", fontsize=13, fontweight='bold', pad=10)

    info_lines = [
        f"Step:          {step}",
        f"Total reward:  {reward_total:.2f}",
        f"Pieces placed: {current_piece_idx} / {n*n}",
    ]

    if last_action is not None:
        status = "✓ CORRECT" if last_correct else "✗ WRONG"
        colour = CORRECT_COLOUR if last_correct else WRONG_COLOUR
        info_lines.append("")
        info_lines.append(f"Last action:   Position {last_action}")
        ax_info.text(0.05, 0.38, f"Result:        {status}",
                     transform=ax_info.transAxes,
                     fontsize=11, color=colour, fontweight='bold')

    for i, line in enumerate(info_lines):
        ax_info.text(0.05, 0.85 - i * 0.12, line,
                     transform=ax_info.transAxes,
                     fontsize=11, color='#222222')

    # Current piece info
    if current_piece_idx < n * n:
        current_piece = piece_order[current_piece_idx]
        edges = pieces[current_piece]
        ax_info.text(0.05, 0.20,
                     f"Next piece:    P{current_piece}",
                     transform=ax_info.transAxes,
                     fontsize=11, color='#222222')
        ax_info.text(0.05, 0.08,
                     f"Edges: T={EDGE_NAMES[edges[0]]} "
                     f"R={EDGE_NAMES[edges[1]]}\n"
                     f"       B={EDGE_NAMES[edges[2]]} "
                     f"L={EDGE_NAMES[edges[3]]}",
                     transform=ax_info.transAxes,
                     fontsize=9, color='#444444')

    # ── Legend ────────────────────────────────────────────────
    ax_legend.axis('off')
    ax_legend.set_title("Legend", fontsize=11, fontweight='bold', pad=5)

    legend_items = [
        (CORRECT_COLOUR, "Correct placement"),
        (WRONG_COLOUR,   "Wrong placement"),
        (EMPTY_COLOUR,   "Empty position"),
    ]
    for i, (colour, label) in enumerate(legend_items):
        rect = FancyBboxPatch(
            (0.05, 0.70 - i * 0.28),
            0.15, 0.18,
            boxstyle="round,pad=0.02",
            facecolor=colour,
            edgecolor=GRID_COLOUR,
            linewidth=1.5,
            transform=ax_legend.transAxes
        )
        ax_legend.add_patch(rect)
        ax_legend.text(
            0.28, 0.79 - i * 0.28, label,
            transform=ax_legend.transAxes,
            fontsize=9, va='center', color='#222222'
        )


def run_visualiser():
    # ── Load model and environment ────────────────────────────
    print("Loading PPO model...")
    model = PPO.load(MODEL_PATH)

    env = JigsawEnv(grid_size=GRID_SIZE)
    obs, _ = env.reset(seed=42)

    # ── Setup figure ──────────────────────────────────────────
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(
        "PPO Agent — Jigsaw Puzzle Assembly",
        fontsize=15, fontweight='bold', y=0.98
    )

    # Layout: board (left), info (middle), legend (right)
    ax_board  = fig.add_axes([0.02, 0.05, 0.45, 0.88])
    ax_info   = fig.add_axes([0.52, 0.25, 0.26, 0.65])
    ax_legend = fig.add_axes([0.52, 0.05, 0.26, 0.18])

    plt.ion()
    plt.show()

    # ── Run episode ───────────────────────────────────────────
    done          = False
    step          = 0
    reward_total  = 0.0
    last_action   = None
    last_correct  = None

    print(f"\nWatching PPO agent solve a {GRID_SIZE}x{GRID_SIZE} puzzle...")
    print("Close the window to stop.\n")

    # Draw initial empty board
    draw_board(ax_board, ax_info, ax_legend,
               env.board, env.pieces,
               env.current_piece_idx, env.piece_order,
               step, reward_total, None, None, GRID_SIZE)
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(1.0)

    while not done:
        # Agent decides
        action, _ = model.predict(obs, deterministic=True)
        action     = int(action)

        # Check if correct before stepping
        current_piece = env.piece_order[env.current_piece_idx]
        last_correct  = (action == current_piece)
        last_action   = action

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        reward_total += reward
        step         += 1
        done          = terminated or truncated

        print(f"Step {step}: Place P{current_piece} → Position {action} "
              f"{'✓' if last_correct else '✗'} "
              f"(reward: {reward:+.2f})")

        # Redraw
        draw_board(ax_board, ax_info, ax_legend,
                   env.board, env.pieces,
                   env.current_piece_idx, env.piece_order,
                   step, reward_total,
                   last_action, last_correct, GRID_SIZE)
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(DELAY)

    # ── Final state ───────────────────────────────────────────
    correct = int(np.sum(env.board == np.arange(env.n_positions)))
    success = correct == env.n_positions

    result_text = "SOLVED! ✓" if success else f"Partial: {correct}/{env.n_positions}"
    fig.suptitle(
        f"PPO Agent — {result_text} | "
        f"Total reward: {reward_total:.2f} | Steps: {step}",
        fontsize=13, fontweight='bold',
        color=CORRECT_COLOUR if success else WRONG_COLOUR
    )
    fig.canvas.draw()
    fig.canvas.flush_events()

    print(f"\nResult: {result_text}")
    print(f"Total reward: {reward_total:.2f}")
    print(f"Steps: {step}")
    print("\nClose the window when done.")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    run_visualiser()