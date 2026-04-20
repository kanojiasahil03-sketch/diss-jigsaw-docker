"""
jigsaw_env.py
Custom Gymnasium environment for jigsaw puzzle assembly.
Wraps the puzzle state so PPO can train on it.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class JigsawEnv(gym.Env):
    """
    A Gymnasium environment for jigsaw puzzle assembly.
    
    The agent observes the current board state and the next piece's
    edge classifications, then decides which board position to place it.
    
    Observation: [board_state (n_positions)] + [piece_edges (4)]
    Action: Discrete — which position to place the current piece
    """

    metadata = {"render_modes": []}

    def __init__(self, grid_size=3):
        super().__init__()

        self.grid_size = grid_size
        self.n_positions = grid_size * grid_size  # 9 for 3x3

        # Observation space: board state + current piece edges
        # board: n_positions values (0=empty, 1=filled)
        # piece: 4 edge values (0=head, 1=hole, 2=straight)
        obs_size = self.n_positions + 4
        self.observation_space = spaces.Box(
            low=0,
            high=2,
            shape=(obs_size,),
            dtype=np.float32
        )

        # Action space: choose one of n_positions to place the piece
        self.action_space = spaces.Discrete(self.n_positions)

        # Internal state — initialised in reset()
        self.board = None          # which piece is at each position (-1=empty)
        self.pieces = None         # edge classifications for all pieces
        self.piece_order = None    # shuffled order pieces are presented
        self.current_piece_idx = None  # index into piece_order
        self.steps = 0
        self.max_steps = 3 * self.n_positions  # 27 steps max for 3x3

    def _generate_puzzle(self):
        """
        Generate a synthetic puzzle.
        A grid_size x grid_size grid where each shared edge is randomly
        assigned tab (head=0) or blank (hole=1). Border edges are straight=2.
        Adjacent shared edges are always opposites (head <-> hole).
        """
        pieces = np.full((self.n_positions, 4), 2, dtype=np.int32)
        # Edge order per piece: [top=0, right=1, bottom=2, left=3]

        # Horizontal shared edges (between left-right neighbours)
        for row in range(self.grid_size):
            for col in range(self.grid_size - 1):
                left_piece  = row * self.grid_size + col
                right_piece = row * self.grid_size + col + 1
                edge = np.random.choice([0, 1])  # 0=head, 1=hole
                pieces[left_piece][1]  = edge        # left piece right edge
                pieces[right_piece][3] = 1 - edge    # right piece left edge (opposite)

        # Vertical shared edges (between top-bottom neighbours)
        for row in range(self.grid_size - 1):
            for col in range(self.grid_size):
                top_piece    = row * self.grid_size + col
                bottom_piece = (row + 1) * self.grid_size + col
                edge = np.random.choice([0, 1])
                pieces[top_piece][2]    = edge        # top piece bottom edge
                pieces[bottom_piece][0] = 1 - edge    # bottom piece top edge (opposite)

        return pieces

    def _get_obs(self):
        """Build the observation vector for the current state."""
        # Board state: 1 if position filled, 0 if empty
        board_state = (self.board >= 0).astype(np.float32)

        # If all pieces placed, return zeros for piece edges
        if self.current_piece_idx >= self.n_positions:
            piece_edges = np.zeros(4, dtype=np.float32)
        else:
            current_piece = self.piece_order[self.current_piece_idx]
            piece_edges = self.pieces[current_piece].astype(np.float32)

        return np.concatenate([board_state, piece_edges])



    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode."""
        super().reset(seed=seed)

        # Generate a fresh puzzle
        self.pieces = self._generate_puzzle()

        # Board: -1 means empty, otherwise stores piece index
        self.board = np.full(self.n_positions, -1, dtype=np.int32)

        # Shuffle piece order so agent can't memorise a fixed sequence
        self.piece_order = np.random.permutation(self.n_positions)
        self.current_piece_idx = 0
        self.steps = 0

        return self._get_obs(), {}

    def step(self, action):
        """
        Take one step: place the current piece at the chosen position.
        
        Returns: observation, reward, terminated, truncated, info
        """
        self.steps += 1
        current_piece = self.piece_order[self.current_piece_idx]
        reward = 0.0

        # Check if position already filled
        if self.board[action] >= 0:
            reward = -0.3  # penalty for trying an occupied position
        else:
            # Check if piece belongs at this position
            if action == current_piece:
                reward = 1.0   # correct placement
            else:
                reward = -0.3  # wrong position

            # Place the piece regardless (agent must keep moving)
            self.board[action] = current_piece
            self.current_piece_idx += 1

        # Step penalty — encourages efficiency
        reward -= 0.05

        # Check termination conditions
        terminated = False
        truncated = False

        if self.current_piece_idx >= self.n_positions:
            # All pieces placed
            terminated = True
        elif self.steps >= self.max_steps:
            # Ran out of steps
            truncated = True

        obs = self._get_obs() if not terminated else self._get_obs()
        info = {
            "pieces_placed": self.current_piece_idx,
            "correct_placements": int(np.sum(self.board == np.arange(self.n_positions)))
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        pass