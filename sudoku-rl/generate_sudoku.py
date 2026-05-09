"""Generate Sudoku puzzle/solution pairs.

Saves an .npz with:
  puzzles   (N, 9, 9) uint8, 0 = blank, 1-9 = clue
  solutions (N, 9, 9) uint8, 1-9 = filled
  n_clues   (N,)       uint8, pre-filled cell count per puzzle

The "inpainting" formulation falls out directly from this:
  one-shot       input = puzzle,                       target = solution
  step-by-step   input = puzzle with k of the missing  target = same grid with one
                 cells already revealed,                       additional cell revealed
"""

from __future__ import annotations

import argparse
import random
import time

import numpy as np


# ---- core solver ---------------------------------------------------------

def _candidates(grid: np.ndarray, r: int, c: int) -> list[int]:
    used = set(grid[r, :]) | set(grid[:, c])
    br, bc = 3 * (r // 3), 3 * (c // 3)
    used |= set(grid[br:br + 3, bc:bc + 3].ravel())
    return [v for v in range(1, 10) if v not in used]


def _find_blank(grid: np.ndarray) -> tuple[int, int] | None:
    # MRV (most-constrained-variable) cell choice — big speedup on hard puzzles.
    best, best_n = None, 10
    for r in range(9):
        for c in range(9):
            if grid[r, c] == 0:
                n = len(_candidates(grid, r, c))
                if n < best_n:
                    best, best_n = (r, c), n
                    if n <= 1:
                        return best
    return best


def count_solutions(grid: np.ndarray, cap: int = 2) -> int:
    """Count solutions, bailing once `cap` is reached."""
    grid = grid.copy()

    def rec() -> int:
        cell = _find_blank(grid)
        if cell is None:
            return 1
        r, c = cell
        total = 0
        for v in _candidates(grid, r, c):
            grid[r, c] = v
            total += rec()
            grid[r, c] = 0
            if total >= cap:
                return total
        return total

    return rec()


def _solve_one(grid: np.ndarray, rng: random.Random) -> bool:
    cell = _find_blank(grid)
    if cell is None:
        return True
    r, c = cell
    cands = _candidates(grid, r, c)
    rng.shuffle(cands)
    for v in cands:
        grid[r, c] = v
        if _solve_one(grid, rng):
            return True
        grid[r, c] = 0
    return False


# ---- generation ----------------------------------------------------------

def generate_full_grid(rng: random.Random) -> np.ndarray:
    grid = np.zeros((9, 9), dtype=np.uint8)
    _solve_one(grid, rng)
    return grid


def generate_puzzle(
    target_clues: int,
    require_unique: bool,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (puzzle, solution).

    Removes cells in random order. If require_unique, every removal that would
    introduce a second solution is rolled back. Stops when target_clues reached
    or no further removals are possible.
    """
    solution = generate_full_grid(rng)
    puzzle = solution.copy()

    cells = [(r, c) for r in range(9) for c in range(9)]
    rng.shuffle(cells)

    clues = 81
    for r, c in cells:
        if clues <= target_clues:
            break
        saved = puzzle[r, c]
        puzzle[r, c] = 0
        if require_unique and count_solutions(puzzle, cap=2) > 1:
            puzzle[r, c] = saved
        else:
            clues -= 1
    return puzzle, solution


def generate_dataset(
    n: int,
    min_clues: int,
    max_clues: int,
    require_unique: bool,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = random.Random(seed)
    puzzles = np.zeros((n, 9, 9), dtype=np.uint8)
    solutions = np.zeros((n, 9, 9), dtype=np.uint8)
    n_clues = np.zeros(n, dtype=np.uint8)

    t0 = time.time()
    for i in range(n):
        target = rng.randint(min_clues, max_clues)
        p, s = generate_puzzle(target, require_unique, rng)
        puzzles[i] = p
        solutions[i] = s
        n_clues[i] = int((p != 0).sum())
        if (i + 1) % max(1, n // 20) == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  {i + 1}/{n}  ({rate:.1f} puzzles/s)")
    return {"puzzles": puzzles, "solutions": solutions, "n_clues": n_clues}


# ---- inpainting helpers --------------------------------------------------

def step_by_step_pairs(
    puzzle: np.ndarray,
    solution: np.ndarray,
    rng: random.Random,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Convert one (puzzle, solution) into a list of (state_k, state_{k+1}) grids.

    Reveals the missing cells one at a time in random order. The first state is
    the puzzle; the last is the full solution. Each adjacent pair is one
    inpainting step.
    """
    missing = list(zip(*np.where(puzzle == 0)))
    rng.shuffle(missing)
    states = [puzzle.copy()]
    cur = puzzle.copy()
    for r, c in missing:
        cur = cur.copy()
        cur[r, c] = solution[r, c]
        states.append(cur)
    return list(zip(states[:-1], states[1:]))


# ---- CLI -----------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--min-clues", type=int, default=28)
    p.add_argument("--max-clues", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-unique", action="store_true",
                   help="skip uniqueness check (much faster, but some puzzles "
                        "will have multiple solutions)")
    p.add_argument("--out", default="data/sudoku.npz")
    args = p.parse_args()

    print(f"generating {args.n} puzzles, clues in [{args.min_clues}, {args.max_clues}], "
          f"unique={'no' if args.no_unique else 'yes'}, seed={args.seed}")
    data = generate_dataset(
        n=args.n,
        min_clues=args.min_clues,
        max_clues=args.max_clues,
        require_unique=not args.no_unique,
        seed=args.seed,
    )

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, **data)
    print(f"wrote {args.out}  "
          f"({data['puzzles'].nbytes + data['solutions'].nbytes:,} bytes uncompressed)")
    print(f"  clues:  min={data['n_clues'].min()} "
          f"mean={data['n_clues'].mean():.1f} max={data['n_clues'].max()}")


if __name__ == "__main__":
    main()
