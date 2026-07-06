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
                 rng_seed: Optional[int] = None,
                 reward_eat: float = 1.0,
                 reward_die: float = -1.0,
                 reward_step: float = 0.0):
        self.max_steps = max_steps
        self.start_length = start_length
        self.reward_eat = reward_eat
        self.reward_die = reward_die
        # Applied on every step, additive with eat/die. Positive -> survivor
        # incentive (long games score better); negative -> efficiency penalty
        # (bee-line to food). Set to 0 for the classic sparse reward.
        self.reward_step = reward_step
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

        reward = self.reward_step
        info = {"ate": False, "died": False, "truncated": False,
                "score": self.score, "length": len(self.body)}

        if died:
            self.alive = False
            reward = self.reward_die + self.reward_step
            info["died"] = True
            return StepResult(self.obs(), reward, True, info)

        self.body.append((nr, nc))
        if will_grow:
            self.score += 1
            reward = self.reward_eat + self.reward_step
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

_WALLS = frozenset(
    [(0, c) for c in range(GRID_COLS)]
    + [(GRID_ROWS - 1, c) for c in range(GRID_COLS)]
    + [(r, 0) for r in range(GRID_ROWS)]
    + [(r, GRID_COLS - 1) for r in range(GRID_ROWS)])


def _bfs_path(start: tuple[int, int], goal: tuple[int, int],
              blocked: set) -> Optional[list[tuple[int, int]]]:
    """Shortest path from start to goal avoiding `blocked`. Returns the list
    of cells [first_step, ..., goal] (start excluded), or None."""
    if start == goal:
        return None
    from collections import deque as _dq
    seen: dict[tuple[int, int], Optional[tuple[int, int]]] = {start: None}
    q = _dq([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        cr, cc = cur
        for dr, dc in _DELTA.values():
            nxt = (cr + dr, cc + dc)
            if nxt in seen or nxt in blocked:
                continue
            seen[nxt] = cur
            q.append(nxt)
    if goal not in seen:
        return None
    path = [goal]
    while seen[path[-1]] != start:
        path.append(seen[path[-1]])
    path.reverse()
    return path


def _dir_of(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Absolute direction of the single-cell move a -> b."""
    dr, dc = b[0] - a[0], b[1] - a[1]
    for d, delta in _DELTA.items():
        if delta == (dr, dc):
            return d
    raise ValueError(f"cells not adjacent: {a} -> {b}")


def _bfs_next_absolute(snake: TinySnake) -> Optional[int]:
    """BFS from head to food, avoiding walls / body. Return absolute next-step
    direction (UP/DOWN/LEFT/RIGHT), or None if unreachable.
    """
    body_list = list(snake.body)
    blocked = set(_WALLS)
    blocked.update(body_list[1:])  # head's spot will be vacated, tail vacates
    path = _bfs_path(snake.head, snake.food, blocked)
    if path is None:
        return None
    return _dir_of(snake.head, path[0])


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


def _floodfill_fallback(snake: TinySnake) -> int:
    """Pick the relative action whose flood-fill reach from the resulting
    head cell is largest. Last-resort move when no BFS path exists."""
    blocked = _WALLS | set(snake.body)
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


def heuristic_action(snake: TinySnake) -> int:
    """BFS-driven teacher in the 3-action relative space.

    Falls back to "the safest of the three available moves" (max flood-fill
    from the resulting head cell) when BFS finds no path to the food, e.g.
    when the snake's body completely cordons it off.
    """
    nxt = _bfs_next_absolute(snake)
    if nxt is not None:
        return _abs_to_relative(snake.direction, nxt)
    return _floodfill_fallback(snake)


# -- tail-safe teacher --------------------------------------------------------

def _simulate_path(body: list[tuple[int, int]], path: list[tuple[int, int]],
                   food: tuple[int, int]
                   ) -> Optional[deque]:
    """Walk the snake along `path` with real body dynamics (tail vacates each
    step; eating the food cell grows by one). Returns the resulting body
    deque, or None if the path collides with the (moving) body."""
    b = deque(body)
    for cell in path:
        grow = cell == food
        occupied = set(b) if grow else set(list(b)[1:])
        if cell in occupied:
            return None
        b.append(cell)
        if not grow:
            b.popleft()
    return b


def _tail_reachable(body: deque) -> bool:
    """True if the head can reach the tail cell (which vacates next tick)."""
    body_list = list(body)
    head, tail = body_list[-1], body_list[0]
    if head == tail:
        return True
    blocked = set(_WALLS)
    blocked.update(body_list[1:-1])  # tail is the goal, head is the start
    return _bfs_path(head, tail, blocked) is not None


def safe_heuristic_action(snake: TinySnake) -> int:
    """Tail-safe BFS teacher: take the shortest path to food only if, after
    simulating the full path (including growth), the head can still reach
    its own tail. Otherwise chase the tail — following your own vacating
    tail is always survivable — and only then fall back to flood-fill.

    This fixes the plain BFS teacher's dominant failure mode: greedy
    shortest paths that box the snake in right after eating.
    """
    body = list(snake.body)
    head, tail = body[-1], body[0]
    blocked = set(_WALLS)
    blocked.update(body[1:])  # tail vacates
    path = _bfs_path(head, snake.food, blocked)
    if path is not None:
        virt = _simulate_path(body, path, snake.food)
        if virt is not None and _tail_reachable(virt):
            return _abs_to_relative(snake.direction, _dir_of(head, path[0]))
    # Unsafe (or no path) to eat: chase the tail, preferring not to eat
    # accidentally along the way.
    chase_blocked = set(_WALLS)
    chase_blocked.update(body[1:-1])
    for avoid_food in (True, False):
        b = set(chase_blocked)
        if avoid_food:
            b.add(snake.food)
        tail_path = _bfs_path(head, tail, b)
        if tail_path is not None:
            return _abs_to_relative(snake.direction,
                                    _dir_of(head, tail_path[0]))
    return _floodfill_fallback(snake)


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


# 5 one-hot + food potential + body-age + 4 heading planes.
FULL_OBS_CHANNELS = SYM_NUM_TYPES + 1 + 1 + 4


def extract_obs_full(snake: TinySnake) -> np.ndarray:
    """Return an (11, GRID_ROWS, GRID_COLS) float32 obs:

        channels 0-5:  same as `extract_obs_with_dist` (one-hot + potential)
        channel 6:     body-age: for each body cell (tail..head), the value
                       (index_from_tail + 1) / len(body). The tail (smallest
                       value) vacates first; the head is 1.0. Elsewhere 0.
        channels 7-10: constant planes, one-hot of the absolute heading
                       (UP/DOWN/LEFT/RIGHT).

    Motivation: actions are *relative*, but the bare grid neither encodes
    the heading (ambiguous when the body coils next to the head) nor which
    body cells vacate soon — the single most useful fact for late-game
    routing. The teacher reads both straight off game state; this hands
    them to the encoder.
    """
    base = extract_obs_with_dist(snake)
    age = np.zeros((1, GRID_ROWS, GRID_COLS), dtype=np.float32)
    L = len(snake.body)
    for i, (r, c) in enumerate(snake.body):  # i = 0 at tail, L-1 at head
        age[0, r, c] = (i + 1) / L
    heading = np.zeros((4, GRID_ROWS, GRID_COLS), dtype=np.float32)
    heading[snake.direction] = 1.0
    return np.concatenate([base, age, heading], axis=0)


class TinySnakeVecEnv:
    """In-proc vectorized wrapper over `num_envs` TinySnake instances.

    Obs modes: bare 1-channel int grid (default; 5-class one-hot built at
    the model boundary), `add_distance` = 6-channel float with the distance
    map pre-computed, `full_features` = 11-channel float (adds body-age +
    heading planes; implies the distance channel).
    """
    OBS_SHAPE = (GRID_ROWS, GRID_COLS)
    NUM_ACTIONS = NUM_ACTIONS

    def __init__(self, env_kwargs: dict | None = None,
                 add_distance: bool = False,
                 full_features: bool = False,
                 num_envs: int = 1):
        kw = dict(env_kwargs or {})
        seed = kw.pop("rng_seed", None)
        self._snakes = [
            TinySnake(**kw,
                      rng_seed=None if seed is None else seed + 1000 * i)
            for i in range(num_envs)
        ]
        for s in self._snakes:
            s.reset()
        self.num_envs = num_envs
        self.add_distance = add_distance
        self.full_features = full_features

    def _obs_one(self, snake: TinySnake) -> np.ndarray:
        if self.full_features:
            return extract_obs_full(snake)
        if self.add_distance:
            return extract_obs_with_dist(snake)
        return snake.obs()

    def _obs_all(self) -> np.ndarray:
        return np.stack([self._obs_one(s) for s in self._snakes])

    def reset(self) -> np.ndarray:
        for s in self._snakes:
            s.reset()
        return self._obs_all()

    def step(self, actions):
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict] = []
        for i, snake in enumerate(self._snakes):
            s = snake.step(int(actions[i]))
            info = dict(s.info)
            if s.done:
                # On truncation the snake is still alive — capture the obs at
                # the truncation point so the trainer can bootstrap
                # V(s_terminal) instead of zeroing the bootstrap. Real deaths
                # leave the snake in a pre-move state and bootstrap stays 0.
                if info.get("truncated", False):
                    info["terminal_obs"] = self._obs_one(snake)
                snake.reset()
            rewards[i] = s.reward
            dones[i] = s.done
            infos.append(info)
        return self._obs_all(), rewards, dones, infos

    @property
    def games(self) -> list[TinySnake]:
        """Per-env game state, e.g. for teacher labeling during rollouts."""
        return self._snakes

    @property
    def _env(self):
        """Expose env 0 so the single-env heuristic access pattern
        (`vec._env._game`) keeps working."""
        return _Adapter(self._snakes[0])

    def close(self) -> None:
        pass


class _Adapter:
    """Tiny shim so train_bc's `vec._env._game` access pattern keeps working."""
    def __init__(self, snake: TinySnake):
        self._game = snake


# -- teacher benchmark ---------------------------------------------------------

def _benchmark(teacher: str = "bfs", episodes: int = 50,
               max_steps: int = 500, seed: int = 0) -> None:
    fn = safe_heuristic_action if teacher == "safe" else heuristic_action
    scores, lengths, deaths, truncs = [], [], 0, 0
    for ep in range(episodes):
        s = TinySnake(max_steps=max_steps, rng_seed=seed + ep)
        s.reset()
        while True:
            r = s.step(fn(s))
            if r.done:
                scores.append(s.score)
                lengths.append(len(s.body))
                deaths += int(r.info["died"])
                truncs += int(r.info["truncated"])
                break
    arr = np.array(scores)
    print(f"teacher={teacher}  eps={episodes}  max_steps={max_steps}  "
          f"mean {arr.mean():.2f}  median {int(np.median(arr))}  "
          f"min/max {arr.min()}/{arr.max()}  "
          f"deaths {deaths}  truncations {truncs}  "
          f"mean final length {np.mean(lengths):.1f}")


if __name__ == "__main__":
    import argparse
    _p = argparse.ArgumentParser(description="Benchmark a scripted teacher.")
    _p.add_argument("--teacher", choices=["bfs", "safe"], default="bfs")
    _p.add_argument("--episodes", type=int, default=50)
    _p.add_argument("--max-steps", type=int, default=500)
    _p.add_argument("--seed", type=int, default=0)
    _a = _p.parse_args()
    _benchmark(_a.teacher, _a.episodes, _a.max_steps, _a.seed)
