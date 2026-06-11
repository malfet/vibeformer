"""BFS-based teacher policy for QBasic Nibbles.

The teacher inspects the game state directly (no perception step), runs BFS
from the snake head to the active number cell avoiding walls and the snake's
own body, and returns the first move along the shortest safe path.

Plays the role of `tools/heuristic_agent.py` in ../digger-rl: a strong scripted
policy whose action-labels we use for BC pretrain and DAGGER.

Tiebreakers and refinements (kept simple on purpose):

- Of the snake's 3 forward / sideways options (no 180-reverse), pick the
  action that BFS labels with shortest path-to-number.
- If multiple actions tie, prefer the one whose immediate next cell has the
  most empty neighbors (anti-corner bias).
- If no action reaches the number (BFS fails — number blocked by snake body),
  fall back to "longest reach": pick the action whose flood-fill from the
  next cell visits the most empty cells. Buys time for the body to clear.

This already mirrors digger-rl's SmartHeuristic mental model: BFS-distance for
safety, plus a connectivity heuristic when the planner can't find the goal.
"""

from __future__ import annotations

import argparse
import collections
from typing import Optional, Tuple

from nibbles_sim import (
    NibblesGame,
    ARENA_ROWS, ARENA_COLS,
    EMPTY, WALL, SNAKE,
    NOOP, UP, DOWN, LEFT, RIGHT,
)


_ACTION_DIRS = {
    UP:    (-1, 0),
    DOWN:  (+1, 0),
    LEFT:  (0, -1),
    RIGHT: (0, +1),
}
_OPP_ACTION = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

# Map the BAS direction code (1..4) to our action enum.
_DIR_TO_ACTION = {1: UP, 2: DOWN, 3: LEFT, 4: RIGHT}


def _bfs_distance(arena, start: Tuple[int, int],
                  goal: Tuple[int, int]) -> Optional[int]:
    """Shortest path length from start to goal across EMPTY cells.

    Walls (and snake body) block; the goal cell is allowed to be entered even
    if it isn't EMPTY (the number-cell is logically empty in the arena array
    but we don't rely on that). Returns None if unreachable.
    """
    if start == goal:
        return 0
    sr, sc = start
    if not (1 <= sr <= ARENA_ROWS and 1 <= sc <= ARENA_COLS):
        return None
    if arena[sr][sc] != EMPTY:
        return None
    seen = {start}
    q = collections.deque([(sr, sc, 0)])
    while q:
        r, c, d = q.popleft()
        for dr, dc in ((-1, 0), (+1, 0), (0, -1), (0, +1)):
            nr, nc = r + dr, c + dc
            if not (1 <= nr <= ARENA_ROWS and 1 <= nc <= ARENA_COLS):
                continue
            if (nr, nc) == goal:
                return d + 1
            if arena[nr][nc] != EMPTY:
                continue
            if (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            q.append((nr, nc, d + 1))
    return None


def _flood_size(arena, start: Tuple[int, int], cap: int = 800) -> int:
    """Flood-fill size from `start` across EMPTY cells (capped)."""
    if not (1 <= start[0] <= ARENA_ROWS and 1 <= start[1] <= ARENA_COLS):
        return 0
    if arena[start[0]][start[1]] != EMPTY:
        return 0
    seen = {start}
    q = collections.deque([start])
    while q and len(seen) < cap:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (+1, 0), (0, -1), (0, +1)):
            nr, nc = r + dr, c + dc
            if not (1 <= nr <= ARENA_ROWS and 1 <= nc <= ARENA_COLS):
                continue
            if (nr, nc) in seen:
                continue
            if arena[nr][nc] != EMPTY:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return len(seen)


def _number_target_cell(game: NibblesGame) -> Optional[Tuple[int, int]]:
    """Pick which of the number's two sister sub-rows to aim for.

    Per BAS, the eat-check compares `numberRow == INT((sammy.row+1)/2)`,
    i.e. either sub-row of the digit triggers eating. We aim for whichever
    is reachable; default to the lower sub-row (real-row 2R) since it tends
    to be more accessible from below.
    """
    if game.number_row == 0:
        return None
    (r1, c), (r2, _) = game.number_arena_cells()
    return (r1, c) if r1 < r2 else (r2, c)


def heuristic_action(game: NibblesGame) -> int:
    """Choose an action for the current NibblesGame state."""
    head = game.head
    cur_dir = game.snake.direction
    forbidden = _OPP_ACTION[_DIR_TO_ACTION[cur_dir]]

    target = _number_target_cell(game)

    candidates = []  # (-tie_score, action) - smaller is better
    for action, (dr, dc) in _ACTION_DIRS.items():
        if action == forbidden:
            continue
        nr, nc = head[0] + dr, head[1] + dc
        if not (1 <= nr <= ARENA_ROWS and 1 <= nc <= ARENA_COLS):
            continue
        if game.arena[nr][nc] != EMPTY:
            continue

        if target is not None:
            d = _bfs_distance(game.arena, (nr, nc), target)
            if d is None:
                # Unreachable from this neighbor; fall back to flood size.
                tie = (1_000_000, -_flood_size(game.arena, (nr, nc)))
            else:
                # Anti-corner tiebreak: prefer neighbor cells whose flood
                # size is largest.
                tie = (d, -_flood_size(game.arena, (nr, nc)))
        else:
            tie = (0, -_flood_size(game.arena, (nr, nc)))
        candidates.append((tie, action))

    if not candidates:
        # Nowhere safe to go. Just continue forward; we die next step.
        return _DIR_TO_ACTION[cur_dir]

    candidates.sort()
    return candidates[0][1]


# -- standalone harness -------------------------------------------------------

def _run_episodes(num_episodes: int, max_steps: int, seed: int, verbose: bool):
    scores = []
    for ep in range(num_episodes):
        game = NibblesGame(max_steps=max_steps, rng_seed=seed + ep)
        steps = 0
        while True:
            a = heuristic_action(game)
            info = game.step(a)
            steps += 1
            if info.game_over or info.truncated:
                break
        scores.append(game.snake.score)
        if verbose:
            print(f"ep {ep}: score={game.snake.score} "
                  f"level={game.level} lives={game.snake.lives} "
                  f"steps={steps}")
    print(f"\nmean score over {num_episodes} eps: "
          f"{sum(scores) / len(scores):.1f}")
    print(f"max score: {max(scores)}  min: {min(scores)}")


def _live(max_steps: int, seed: int, sleep_s: float):
    import matplotlib.pyplot as plt
    from nibbles_env import NibblesEnv

    env = NibblesEnv(max_steps=max_steps, rng_seed=seed)
    obs = env.reset()
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(obs)
    ax.set_axis_off()
    title = ax.set_title("score=0 lives=5 level=1")

    while True:
        a = heuristic_action(env._game)
        s = env.step(a)
        im.set_data(s.obs)
        title.set_text(
            f"score={s.info['score']} lives={s.info['lives']} "
            f"level={s.info['level']} (next={s.info['number_value']})"
        )
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(sleep_s)
        if s.done:
            print(f"DONE: {s.info}")
            break


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--live", action="store_true",
                   help="Display a matplotlib live view of one episode.")
    p.add_argument("--sleep", type=float, default=0.02,
                   help="Delay between steps in --live mode.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if args.live:
        _live(args.max_steps, args.seed, args.sleep)
    else:
        _run_episodes(args.episodes, args.max_steps, args.seed,
                      verbose=not args.quiet)


if __name__ == "__main__":
    main()
