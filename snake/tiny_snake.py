"""Textbook 12x12 snake — minimal env for BC + MC-credit experiments.

Deliberately simpler than NIBBLES so we can measure what learns at all:
- 12 x 12 grid, border walls only (no level layouts, no progression).
- One food at a time; respawns instantly on a random empty cell.
- Snake starts length 3 at center going right.
- Reward: +1 per food eaten, -1 on death (wall or self).
- Episode ends on death.
- **3-action relative space**: STRAIGHT / TURN_LEFT / TURN_RIGHT. No
  NOOP and no reverse: the agent's action is always valid w.r.t. the
  no-reverse rule, the same label means the same thing regardless of
  current heading, and the teacher labels in the same space.

Same VecEnv API as `nibbles_env.SymbolicVecEnv` so train_bc.py can plug
in with minimal changes.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


GRID_ROWS = 12
GRID_COLS = 12

# Cell type encoding (same scheme as nibbles_env symbolic).
SYM_EMPTY = 0
SYM_WALL = 1
SYM_BODY = 2
SYM_HEAD = 3
SYM_FOOD = 4
SYM_NUM_TYPES = 5

# Relative actions (the only ones the agent ever sees).
STRAIGHT = 0
TURN_LEFT = 1
TURN_RIGHT = 2
NUM_ACTIONS = 3

# Internal absolute headings.
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
_DELTA = {UP: (-1, 0), DOWN: (+1, 0), LEFT: (0, -1), RIGHT: (0, +1)}
# Turning maps (90 degrees CCW for LEFT, CW for RIGHT).
_TURN_LEFT = {UP: LEFT, LEFT: DOWN, DOWN: RIGHT, RIGHT: UP}
_TURN_RIGHT = {UP: RIGHT, RIGHT: DOWN, DOWN: LEFT, LEFT: UP}


@dataclass
class StepResult:
    obs: np.ndarray   # (GRID_ROWS, GRID_COLS) uint8 with cell-type codes
    reward: float
    done: bool
    info: dict


class TinySnake:
    """Single-env textbook snake. Plays one continuous episode at a time."""

    def __init__(self, max_steps: int = 1000,
                 start_length: int = 3,
                 rng_seed: Optional[int] = None):
        self.max_steps = max_steps
        self.start_length = start_length
        self._seeder = random.Random(rng_seed)
        self._game_rng = random.Random()
        self.body: deque = deque()
        self.direction = RIGHT
        self.food: tuple[int, int] = (0, 0)
        self.steps = 0
        self.score = 0
        self.alive = False

    # -- core mechanics -----------------------------------------------------

    def reset(self) -> np.ndarray:
        self._game_rng.seed(self._seeder.randrange(2 ** 31))
        r = GRID_ROWS // 2
        c = GRID_COLS // 2
        # Build the starting snake horizontally so STRAIGHT initially = RIGHT.
        self.body = deque((r, c - i) for i in range(self.start_length - 1, -1, -1))
        self.direction = RIGHT
        self.alive = True
        self.steps = 0
        self.score = 0
        self._spawn_food()
        return self.obs()

    def step(self, action: int) -> StepResult:
        assert self.alive, "call reset() first"
        self.steps += 1
        if action == TURN_LEFT:
            self.direction = _TURN_LEFT[self.direction]
        elif action == TURN_RIGHT:
            self.direction = _TURN_RIGHT[self.direction]
        # else STRAIGHT — keep direction

        dr, dc = _DELTA[self.direction]
        hr, hc = self.body[-1]
        nr, nc = hr + dr, hc + dc

        # Death by wall: 1..GRID_ROWS-2 and 1..GRID_COLS-2 are interior; the
        # border (row 0, GRID_ROWS-1, col 0, GRID_COLS-1) is a kill zone.
        died = (
            nr <= 0 or nr >= GRID_ROWS - 1
            or nc <= 0 or nc >= GRID_COLS - 1
        )
        # Self-collision: head about to enter a body cell that won't move
        # this tick (i.e., everything except the tail when we won't grow).
        will_grow = (nr, nc) == self.food
        body_check = set(self.body) if will_grow else set(list(self.body)[1:])
        if not died and (nr, nc) in body_check:
            died = True

        reward = 0.0
        info = {"ate": False, "died": False, "truncated": False,
                "score": self.score, "length": len(self.body)}

        if died:
            self.alive = False
            reward = -1.0
            info["died"] = True
            return StepResult(self.obs(), reward, True, info)

        self.body.append((nr, nc))
        if will_grow:
            self.score += 1
            reward = 1.0
            info["ate"] = True
            self._spawn_food()
        else:
            self.body.popleft()

        if self.steps >= self.max_steps:
            info["truncated"] = True
            info["score"] = self.score
            info["length"] = len(self.body)
            return StepResult(self.obs(), reward, True, info)

        info["score"] = self.score
        info["length"] = len(self.body)
        return StepResult(self.obs(), reward, False, info)

    # -- helpers -----------------------------------------------------------

    def _spawn_food(self) -> None:
        body_set = set(self.body)
        empties = [
            (r, c)
            for r in range(1, GRID_ROWS - 1)
            for c in range(1, GRID_COLS - 1)
            if (r, c) not in body_set
        ]
        if not empties:
            # Filled the whole interior; degenerate. Park food on the head
            # so eat-check never fires again.
            self.food = self.body[-1]
            return
        self.food = self._game_rng.choice(empties)

    def obs(self) -> np.ndarray:
        arr = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.uint8)
        # Walls = border ring.
        arr[0, :] = SYM_WALL
        arr[GRID_ROWS - 1, :] = SYM_WALL
        arr[:, 0] = SYM_WALL
        arr[:, GRID_COLS - 1] = SYM_WALL
        for r, c in self.body:
            arr[r, c] = SYM_BODY
        hr, hc = self.body[-1]
        arr[hr, hc] = SYM_HEAD
        fr, fc = self.food
        arr[fr, fc] = SYM_FOOD
        return arr

    @property
    def head(self) -> tuple[int, int]:
        return self.body[-1]


# -- heuristic teacher (BFS in absolute coords, relabeled to relative) -------

def _bfs_next_absolute(snake: TinySnake) -> Optional[int]:
    """BFS from head to food, avoiding walls / body. Return absolute next-step
    direction (UP/DOWN/LEFT/RIGHT), or None if unreachable.
    """
    start = snake.head
    goal = snake.food
    if start == goal:
        return None
    # Forbidden cells: walls + body (excluding tail, which moves out).
    body_list = list(snake.body)
    blocked = set([(0, c) for c in range(GRID_COLS)]
                  + [(GRID_ROWS - 1, c) for c in range(GRID_COLS)]
                  + [(r, 0) for r in range(GRID_ROWS)]
                  + [(r, GRID_COLS - 1) for r in range(GRID_ROWS)])
    blocked.update(body_list[1:])  # head's spot will be vacated, tail vacates
    # Parent pointers for path reconstruction.
    from collections import deque as _dq
    seen = {start: None}
    q = _dq([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        cr, cc = cur
        for d, (dr, dc) in _DELTA.items():
            nxt = (cr + dr, cc + dc)
            if nxt in seen or nxt in blocked:
                continue
            seen[nxt] = (cur, d)
            q.append(nxt)
    if goal not in seen:
        return None
    # Trace back to find the first move's direction.
    cur = goal
    while seen[cur] is not None and seen[cur][0] != start:
        cur = seen[cur][0]
    return seen[cur][1]


def _abs_to_relative(snake_dir: int, next_dir: int) -> int:
    """Translate an absolute heading change into our 3-action relative space."""
    if next_dir == snake_dir:
        return STRAIGHT
    if _TURN_LEFT[snake_dir] == next_dir:
        return TURN_LEFT
    if _TURN_RIGHT[snake_dir] == next_dir:
        return TURN_RIGHT
    # The teacher tried to reverse — impossible from BFS unless tail-eat
    # bookkeeping let it through. Default to STRAIGHT and accept death.
    return STRAIGHT


def heuristic_action(snake: TinySnake) -> int:
    """BFS-driven teacher in the 3-action relative space.

    Falls back to "the safest of the three available moves" (max flood-fill
    from the resulting head cell) when BFS finds no path to the food, e.g.
    when the snake's body completely cordons it off.
    """
    nxt = _bfs_next_absolute(snake)
    if nxt is not None:
        return _abs_to_relative(snake.direction, nxt)
    # No path — pick the relative action whose flood-fill reach is largest.
    body_set = set(snake.body)
    walls = set([(0, c) for c in range(GRID_COLS)]
                + [(GRID_ROWS - 1, c) for c in range(GRID_COLS)]
                + [(r, 0) for r in range(GRID_ROWS)]
                + [(r, GRID_COLS - 1) for r in range(GRID_ROWS)])
    blocked = walls | body_set
    best_a, best_size = STRAIGHT, -1
    for a, abs_dir in (
            (STRAIGHT, snake.direction),
            (TURN_LEFT, _TURN_LEFT[snake.direction]),
            (TURN_RIGHT, _TURN_RIGHT[snake.direction])):
        dr, dc = _DELTA[abs_dir]
        hr, hc = snake.head
        cand = (hr + dr, hc + dc)
        if cand in blocked:
            size = -1  # immediate death
        else:
            from collections import deque as _dq
            seen = {cand}
            q = _dq([cand])
            while q and len(seen) < GRID_ROWS * GRID_COLS:
                r, c = q.popleft()
                for ddr, ddc in _DELTA.values():
                    n = (r + ddr, c + ddc)
                    if n not in seen and n not in blocked:
                        seen.add(n)
                        q.append(n)
            size = len(seen)
        if size > best_size:
            best_size = size
            best_a = a
    return best_a


# -- VecEnv wrapper (single env, mirrors SymbolicVecEnv API) ------------------

def compute_distance_map(snake: TinySnake) -> np.ndarray:
    """BFS distance from each cell to the food, treating walls + body as
    blocked. Unreachable cells get GRID_ROWS*GRID_COLS as a sentinel.

    Returns (GRID_ROWS, GRID_COLS) float32. The teacher essentially picks
    the action whose next-head-cell has the smallest value in this map.
    """
    INF = float(GRID_ROWS * GRID_COLS)
    dist = np.full((GRID_ROWS, GRID_COLS), INF, dtype=np.float32)
    fr, fc = snake.food
    walls = {(0, c) for c in range(GRID_COLS)}
    walls |= {(GRID_ROWS - 1, c) for c in range(GRID_COLS)}
    walls |= {(r, 0) for r in range(GRID_ROWS)}
    walls |= {(r, GRID_COLS - 1) for r in range(GRID_ROWS)}
    blocked = walls | set(snake.body)
    if (fr, fc) in blocked:
        return dist
    dist[fr, fc] = 0.0
    from collections import deque as _dq
    q = _dq([(fr, fc)])
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS):
                continue
            if (nr, nc) in blocked:
                continue
            if dist[nr, nc] != INF:
                continue
            dist[nr, nc] = dist[r, c] + 1.0
            q.append((nr, nc))
    return dist


_DIST_SCALE = 4.0  # potential-field bandwidth; tuned so neighbor distances
                   # 3 vs 5 land 0.18 apart instead of 0.01.


def extract_obs_with_dist(snake: TinySnake) -> np.ndarray:
    """Return a (6, GRID_ROWS, GRID_COLS) float32 obs:

        channels 0-4: one-hot of cell types (matches `snake.obs()` codes)
        channel 5:    Potential field exp(-d / _DIST_SCALE), where d is the
                      BFS distance from each cell to the food. Blocked /
                      unreachable cells get 0.0. High values pull the snake
                      toward food; the encoder picks the neighbor cell with
                      the largest value to match teacher behavior.

    Choosing exp(-d/k) over plain d/INF: at the head's neighbors, distances
    differ by O(1) but a divide-by-INF normalisation compresses each unit
    step to 1/144 ~= 0.007 (vs 0.18 here), which is hard for the encoder
    to discriminate. Inspired by potential-field path planning.
    """
    sym = snake.obs()
    onehot = np.eye(SYM_NUM_TYPES, dtype=np.float32)[sym]  # (H, W, 5)
    onehot = onehot.transpose(2, 0, 1)                     # (5, H, W)
    INF = float(GRID_ROWS * GRID_COLS)
    dist = compute_distance_map(snake)
    potential = np.where(dist < INF,
                         np.exp(-dist / _DIST_SCALE), 0.0
                         ).astype(np.float32)
    return np.concatenate([onehot, potential[None]], axis=0)


class TinySnakeVecEnv:
    """1-env wrapper. `add_distance` switches the obs from the bare 1-channel
    int grid (5-class one-hot built at the model boundary) to the 6-channel
    float tensor with the distance map already pre-computed.
    """
    OBS_SHAPE = (GRID_ROWS, GRID_COLS)
    NUM_ACTIONS = NUM_ACTIONS

    def __init__(self, env_kwargs: dict | None = None,
                 add_distance: bool = False):
        self._snake = TinySnake(**(env_kwargs or {}))
        self._snake.reset()
        self.add_distance = add_distance

    def _obs(self) -> np.ndarray:
        if self.add_distance:
            return extract_obs_with_dist(self._snake)
        return self._snake.obs()

    def reset(self) -> np.ndarray:
        self._snake.reset()
        return self._obs()[None]

    def step(self, actions):
        s = self._snake.step(int(actions[0]))
        if s.done:
            self._snake.reset()
        obs = self._obs()
        return (obs[None],
                np.array([s.reward], dtype=np.float32),
                np.array([s.done], dtype=bool),
                [s.info])

    @property
    def _env(self):
        """Expose the inner snake so the heuristic can read state directly."""
        return _Adapter(self._snake)

    def close(self) -> None:
        pass


class _Adapter:
    """Tiny shim so train_bc's `vec._env._game` access pattern keeps working."""
    def __init__(self, snake: TinySnake):
        self._game = snake


# -- smoke test ---------------------------------------------------------------

def _smoke() -> None:
    s = TinySnake(max_steps=500, rng_seed=0)
    s.reset()
    eaten = 0
    deaths = 0
    while True:
        a = heuristic_action(s)
        r = s.step(a)
        if r.info["ate"]:
            eaten += 1
        if r.info["died"]:
            deaths += 1
        if r.done:
            print(f"done in {s.steps} steps, score {s.score}, "
                  f"died={r.info['died']}, ate={eaten}, length={len(s.body)}")
            break


if __name__ == "__main__":
    _smoke()
