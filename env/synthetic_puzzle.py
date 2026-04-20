"""
synthetic_puzzle.py
Generates synthetic jigsaw puzzles for RL training and evaluation.

Why synthetic data?
Real puzzle photos add noise from lighting, texture, and image quality
that has nothing to do with the planning problem. Synthetic data lets
us isolate the RL contribution cleanly.

Edge types:
  0 = head  (convex bump, tab sticking out)
  1 = hole  (concave slot, tab going in)
  2 = straight (flat border edge)

Edge order per piece: [top, right, bottom, left]

Constraint: adjacent shared edges must always be opposites
  head (0) <-> hole (1)
"""

import numpy as np


class SyntheticPuzzle:
    """
    Generates a grid_size x grid_size jigsaw puzzle with random edges.

    Example usage:
        puzzle = SyntheticPuzzle(grid_size=3)
        print(puzzle.pieces)        # shape: (9, 4)
        print(puzzle.solution)      # shape: (3, 3) — correct position per piece
    """

    def __init__(self, grid_size=3, seed=None):
        self.grid_size = grid_size
        self.n_pieces = grid_size * grid_size
        self.rng = np.random.default_rng(seed)  # reproducible if seed given

        self.pieces = self._generate()

        # solution[i][j] = which piece index belongs at position (i,j)
        # For synthetic puzzles, piece index == correct position index
        self.solution = np.arange(self.n_pieces).reshape(grid_size, grid_size)

    def _generate(self):
        """
        Build edge classifications for all pieces.
        Returns array of shape (n_pieces, 4).
        Each row: [top, right, bottom, left]
        """
        # Start with all edges as straight (border edges stay straight)
        pieces = np.full((self.n_pieces, 4), 2, dtype=np.int32)

        # Assign horizontal shared edges (left piece right <-> right piece left)
        for row in range(self.grid_size):
            for col in range(self.grid_size - 1):
                left_idx  = row * self.grid_size + col
                right_idx = row * self.grid_size + col + 1

                # Randomly assign head or hole to the shared edge
                edge = self.rng.integers(0, 2)  # 0=head or 1=hole
                pieces[left_idx][1]  = edge      # left piece's right edge
                pieces[right_idx][3] = 1 - edge  # right piece's left edge (opposite)

        # Assign vertical shared edges (top piece bottom <-> bottom piece top)
        for row in range(self.grid_size - 1):
            for col in range(self.grid_size):
                top_idx    = row * self.grid_size + col
                bottom_idx = (row + 1) * self.grid_size + col

                edge = self.rng.integers(0, 2)
                pieces[top_idx][2]    = edge      # top piece's bottom edge
                pieces[bottom_idx][0] = 1 - edge  # bottom piece's top edge (opposite)

        return pieces

    def get_piece_edges(self, piece_idx):
        """Return edge classifications for a single piece."""
        return self.pieces[piece_idx]

    def is_valid_placement(self, piece_idx, position, current_board):
        """
        Check if placing piece_idx at position is geometrically valid
        given the current board state.

        current_board: 1D array of length n_positions
                       value = piece index at that position, -1 if empty

        Used by evaluate.py to measure success rate.
        """
        row = position // self.grid_size
        col = position % self.grid_size
        piece = self.pieces[piece_idx]

        # Check top neighbour
        if row > 0:
            top_pos = (row - 1) * self.grid_size + col
            if current_board[top_pos] >= 0:
                top_piece = self.pieces[current_board[top_pos]]
                # My top edge must be opposite of neighbour's bottom edge
                if piece[0] + top_piece[2] != 1:
                    return False

        # Check left neighbour
        if col > 0:
            left_pos = row * self.grid_size + (col - 1)
            if current_board[left_pos] >= 0:
                left_piece = self.pieces[current_board[left_pos]]
                # My left edge must be opposite of neighbour's right edge
                if piece[3] + left_piece[1] != 1:
                    return False

        return True

    def __repr__(self):
        lines = [f"SyntheticPuzzle({self.grid_size}x{self.grid_size})"]
        for i, piece in enumerate(self.pieces):
            row = i // self.grid_size
            col = i % self.grid_size
            lines.append(
                f"  Piece {i} (row={row},col={col}): "
                f"top={piece[0]} right={piece[1]} "
                f"bottom={piece[2]} left={piece[3]}"
            )
        return "\n".join(lines)


def generate_dataset(n_puzzles, grid_size=3, seed=None):
    """
    Generate a dataset of n_puzzles synthetic puzzles.
    Used for training data and held-out test sets.

    Returns list of SyntheticPuzzle objects.
    """
    rng = np.random.default_rng(seed)
    puzzles = []
    for i in range(n_puzzles):
        # Each puzzle gets a unique seed for reproducibility
        puzzle_seed = int(rng.integers(0, 1_000_000))
        puzzles.append(SyntheticPuzzle(grid_size=grid_size, seed=puzzle_seed))
    return puzzles


if __name__ == "__main__":
    # Quick test — run with: python env/synthetic_puzzle.py
    print("Testing SyntheticPuzzle...\n")

    # Test 1: single puzzle
    puzzle = SyntheticPuzzle(grid_size=3, seed=42)
    print(puzzle)
    print()

    # Test 2: verify constraints
    # Adjacent shared edges must always sum to 1 (head+hole=0+1=1)
    print("Verifying edge constraints...")
    errors = 0
    for row in range(puzzle.grid_size):
        for col in range(puzzle.grid_size - 1):
            left  = puzzle.pieces[row * puzzle.grid_size + col]
            right = puzzle.pieces[row * puzzle.grid_size + col + 1]
            if left[1] + right[3] != 1:
                print(f"  ERROR: horizontal mismatch at row={row} col={col}")
                errors += 1

    for row in range(puzzle.grid_size - 1):
        for col in range(puzzle.grid_size):
            top    = puzzle.pieces[row * puzzle.grid_size + col]
            bottom = puzzle.pieces[(row + 1) * puzzle.grid_size + col]
            if top[2] + bottom[0] != 1:
                print(f"  ERROR: vertical mismatch at row={row} col={col}")
                errors += 1

    if errors == 0:
        print("  All edge constraints satisfied ✅")

    # Test 3: generate dataset
    print("\nGenerating dataset of 100 puzzles...")
    dataset = generate_dataset(100, grid_size=3, seed=0)
    print(f"  Generated {len(dataset)} puzzles ✅")

    # Test 4: reproducibility
    p1 = SyntheticPuzzle(grid_size=3, seed=99)
    p2 = SyntheticPuzzle(grid_size=3, seed=99)
    assert np.array_equal(p1.pieces, p2.pieces), "Reproducibility failed!"
    print("  Same seed produces same puzzle ✅")

    print("\nAll tests passed!")