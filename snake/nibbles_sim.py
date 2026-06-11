"""Faithful Python port of QBasic NIBBLES.BAS game logic.

Preserves the original 50-row x 80-col arena (the text-mode sub-pixel trick
in BAS using chr$(220)/chr$(223)/chr$(219)). Rows and cols are 1-indexed to
keep the level wall layouts a direct transcription of the BAS source.

This module is rendering-free. Wrap it with NibblesEnv (in nibbles_env.py)
to get pixel observations.

Coordinate system
-----------------
- Arena: 50 rows x 80 cols. (1,1) top-left in arena coords.
- Rows 1..2: score bar (no collisions).
- Border walls: row 3 (top), row 50 (bottom), col 1 (left), col 80 (right).
- Play area: rows 3..50, cols 1..80 (border inclusive; snake dies on border).
- A "number" digit (1..9) drawn at text-screen row R (1..25) occupies arena
  rows 2*R-1 and 2*R at the same col. In BAS, numbers do NOT update the
  arena collision array, so the snake can move *through* them and the eat
  check is done via numberRow / NumberCol comparison.
"""

from __future__ import annotations

import dataclasses
import random
from collections import deque
from typing import Optional, Tuple

# --- constants (match BAS) ---------------------------------------------------

ARENA_ROWS = 50
ARENA_COLS = 80

# Cell types stored in arena[row][col] for collision purposes.
EMPTY = 0
WALL = 1
SNAKE = 2

# Action space: NOOP keeps the current direction; the rest request a turn.
# A request that would reverse 180 degrees is silently ignored (BAS rule).
NOOP, UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3, 4
NUM_ACTIONS = 5

# Internal direction codes match the BAS:
#   1 = up, 2 = down, 3 = left, 4 = right
_ACTION_TO_DIR = {UP: 1, DOWN: 2, LEFT: 3, RIGHT: 4}
_OPPOSITE = {1: 2, 2: 1, 3: 4, 4: 3}
_DELTA = {1: (-1, 0), 2: (+1, 0), 3: (0, -1), 4: (0, +1)}

STARTING_LIVES = 5
STARTING_LENGTH = 2
DEATH_PENALTY = 10  # BAS: sammy(a).score = sammy(a).score - 10


# --- level wall layouts (direct transcription of `Level` SUB) ----------------

def _border_walls(arena):
    # FOR col = 1 TO 80: Set 3, col, ...: Set 50, col, ...
    for col in range(1, ARENA_COLS + 1):
        arena[3][col] = WALL
        arena[50][col] = WALL
    # FOR row = 4 TO 49: Set row, 1, ...: Set row, 80, ...
    for row in range(4, 50):
        arena[row][1] = WALL
        arena[row][80] = WALL


def _level_walls(level: int, arena):
    """Draw interior walls + return (start_row, start_col, start_dir) for snake 1."""
    if level == 1:
        return 25, 50, 4  # right
    if level == 2:
        for i in range(20, 61):
            arena[25][i] = WALL
        return 7, 60, 3  # left
    if level == 3:
        for i in range(10, 41):
            arena[i][20] = WALL
            arena[i][60] = WALL
        return 25, 50, 1  # up
    if level == 4:
        for i in range(4, 31):
            arena[i][20] = WALL
            arena[53 - i][60] = WALL
        for i in range(2, 41):
            arena[38][i] = WALL
            arena[15][81 - i] = WALL
        return 7, 60, 3
    if level == 5:
        for i in range(13, 40):
            arena[i][21] = WALL
            arena[i][59] = WALL
        for i in range(23, 58):
            arena[11][i] = WALL
            arena[41][i] = WALL
        return 25, 50, 1
    if level == 6:
        for i in range(4, 50):
            if i > 30 or i < 23:
                for col in (10, 20, 30, 40, 50, 60, 70):
                    arena[i][col] = WALL
        return 7, 65, 2  # down
    if level == 7:
        for i in range(4, 50, 2):
            arena[i][40] = WALL
        return 7, 65, 2
    if level == 8:
        for i in range(4, 41):
            arena[i][10] = WALL
            arena[53 - i][20] = WALL
            arena[i][30] = WALL
            arena[53 - i][40] = WALL
            arena[i][50] = WALL
            arena[53 - i][60] = WALL
            arena[i][70] = WALL
        return 7, 65, 2
    if level == 9:
        for i in range(6, 48):
            arena[i][i] = WALL
            arena[i][i + 28] = WALL
        return 40, 75, 1
    # CASE ELSE — level >= 10
    for i in range(4, 50, 2):
        for col in (10, 30, 50, 70):
            arena[i][col] = WALL
        for col in (20, 40, 60):
            arena[i + 1][col] = WALL
    return 7, 65, 2


def _real_to_arena_rows(real_row: int) -> Tuple[int, int]:
    """A text-screen row R covers arena sub-rows 2R-1 and 2R."""
    return 2 * real_row - 1, 2 * real_row


def _arena_to_real_row(arena_row: int) -> int:
    """BAS: arena(row, col).realRow = INT((row + 1) / 2)."""
    return (arena_row + 1) // 2


# --- game state --------------------------------------------------------------

@dataclasses.dataclass
class SnakeState:
    # Head's arena (row, col). The body deque stores (row, col) tuples,
    # most recent (head) on the right.
    body: deque = dataclasses.field(default_factory=deque)
    target_length: int = STARTING_LENGTH
    direction: int = 4  # 1=up 2=down 3=left 4=right (BAS codes)
    pending_dir: int = 4  # direction requested by latest action this tick
    alive: bool = True
    lives: int = STARTING_LIVES
    score: int = 0


@dataclasses.dataclass
class StepInfo:
    score: int
    lives: int
    level: int
    number_value: int           # digit currently on the board (1..9)
    ate: bool                   # number eaten this step
    died: bool                  # snake lost a life this step
    level_complete: bool        # ate '9' and advanced this step
    game_over: bool             # lives == 0
    truncated: bool             # hit max_steps


class NibblesGame:
    """1-player Nibbles, faithful to BAS mechanics."""

    def __init__(self,
                 start_level: int = 1,
                 max_steps: int = 50_000,
                 rng_seed: Optional[int] = None):
        self.start_level = start_level
        self.max_steps = max_steps
        self.rng = random.Random(rng_seed)

        # arena[row][col], 1-indexed; row 0 and col 0 are unused padding.
        self.arena = [[EMPTY] * (ARENA_COLS + 1) for _ in range(ARENA_ROWS + 1)]
        self.snake = SnakeState()
        self.level = start_level
        self.number_value = 1
        self.number_row = 0       # text-screen row 1..25
        self.number_col = 0
        self.steps = 0

        self.reset_full()

    # -- public API -----------------------------------------------------------

    def reset_full(self) -> None:
        """Hard reset: level 1, full lives, score 0."""
        self.level = self.start_level
        self.snake = SnakeState()
        self.number_value = 1
        self.steps = 0
        self._build_level(reset_snake=True)
        self._spawn_number()

    def reset_after_death(self) -> None:
        """Soft reset: same level, snake reset to start, number resets to 1.

        Used when a life was lost but lives > 0. The snake re-spawns at the
        level's start position with length=STARTING_LENGTH. Score is preserved
        (the -10 penalty was already applied in step()).
        """
        self.number_value = 1
        self._build_level(reset_snake=True)
        self._spawn_number()

    def step(self, action: int) -> StepInfo:
        self.steps += 1
        info = StepInfo(
            score=self.snake.score,
            lives=self.snake.lives,
            level=self.level,
            number_value=self.number_value,
            ate=False,
            died=False,
            level_complete=False,
            game_over=False,
            truncated=False,
        )

        # 1. Resolve direction (BAS no-reverse rule).
        if action in _ACTION_TO_DIR:
            requested = _ACTION_TO_DIR[action]
            if requested != _OPPOSITE[self.snake.direction]:
                self.snake.direction = requested

        # 2. Compute new head position.
        dr, dc = _DELTA[self.snake.direction]
        head_r, head_c = self.snake.body[-1]
        new_r, new_c = head_r + dr, head_c + dc

        # 3. Eat check (BAS does this BEFORE the death check, on the new head
        #    position, comparing to (numberRow, NumberCol) where numberRow is
        #    the real-screen row of the digit).
        ate = False
        if (self.number_row == _arena_to_real_row(new_r)
                and self.number_col == new_c
                and self.number_value > 0):
            ate = True
            info.ate = True
            self.snake.score += self.number_value
            # BAS: length += number * 4 (cap at MAXSNAKELENGTH - 30 = 970)
            if self.snake.target_length < 970:
                self.snake.target_length += self.number_value * 4
            if self.number_value == 9:
                # Level complete.
                self.level += 1
                info.level_complete = True
                info.level = self.level
                self.number_value = 1
                self._build_level(reset_snake=True)
                self._spawn_number()
                info.score = self.snake.score
                info.number_value = self.number_value
                return info
            self.number_value += 1
            # Number is consumed; spawn the next one AFTER we resolve the move.
            self.number_row = 0
            self.number_col = 0

        # 4. Death check on the new head cell.
        died = False
        if not (1 <= new_r <= ARENA_ROWS and 1 <= new_c <= ARENA_COLS):
            died = True
        elif self.arena[new_r][new_c] != EMPTY:
            died = True

        if died:
            self.snake.alive = False
            self.snake.lives -= 1
            self.snake.score -= DEATH_PENALTY
            info.died = True
            info.score = self.snake.score
            info.lives = self.snake.lives
            if self.snake.lives <= 0:
                info.game_over = True
                return info
            self.reset_after_death()
            info.level = self.level
            info.number_value = self.number_value
            return info

        # 5. Advance: append new head, conditionally pop tail.
        self.snake.body.append((new_r, new_c))
        self.arena[new_r][new_c] = SNAKE
        if len(self.snake.body) > self.snake.target_length:
            tail_r, tail_c = self.snake.body.popleft()
            # Only clear if the tail cell is still SNAKE (defensive — wall
            # cells shouldn't have been overwritten by the snake anyway).
            if self.arena[tail_r][tail_c] == SNAKE:
                self.arena[tail_r][tail_c] = EMPTY

        # 6. If we just ate, drop the next number.
        if ate and self.number_row == 0:
            self._spawn_number()

        info.score = self.snake.score
        info.number_value = self.number_value

        if self.steps >= self.max_steps:
            info.truncated = True
        return info

    # -- internals ------------------------------------------------------------

    def _build_level(self, reset_snake: bool) -> None:
        # Clear arena, redraw border + interior walls.
        for r in range(1, ARENA_ROWS + 1):
            for c in range(1, ARENA_COLS + 1):
                self.arena[r][c] = EMPTY
        _border_walls(self.arena)
        start_row, start_col, start_dir = _level_walls(self.level, self.arena)

        if reset_snake:
            self.snake.body.clear()
            # BAS starts with length=2; the head is at (row, col), and after
            # 1 step the body will be 2 cells. We seed with a single head cell
            # and let target_length=2 do the work.
            self.snake.body.append((start_row, start_col))
            self.arena[start_row][start_col] = SNAKE
            self.snake.target_length = STARTING_LENGTH
            self.snake.direction = start_dir
            self.snake.pending_dir = start_dir
            self.snake.alive = True

    def _spawn_number(self) -> None:
        """BAS: pick (numberRow, NumberCol) until both arena sub-rows empty."""
        for _ in range(10000):
            arena_row = self.rng.randint(3, 49)  # INT(RND*47 + 3) → 3..49
            col = self.rng.randint(2, 79)        # INT(RND*78 + 2) → 2..79
            # "sister" sub-row: even arena row pairs with row-1, odd with row+1.
            # BAS: sister = (row MOD 2) * 2 - 1, i.e. odd → +1, even → -1.
            sister = +1 if (arena_row % 2 == 1) else -1
            sister_row = arena_row + sister
            if (self.arena[arena_row][col] == EMPTY
                    and 1 <= sister_row <= ARENA_ROWS
                    and self.arena[sister_row][col] == EMPTY):
                self.number_row = _arena_to_real_row(arena_row)
                self.number_col = col
                return
        # Astronomically unlikely fallback: place it somewhere harmless.
        self.number_row = 12
        self.number_col = 40

    # -- introspection helpers (used by env / heuristic) ---------------------

    @property
    def head(self) -> Tuple[int, int]:
        return self.snake.body[-1]

    @property
    def length(self) -> int:
        return len(self.snake.body)

    def is_blocked(self, row: int, col: int) -> bool:
        """True if a step into (row, col) would kill the snake."""
        if not (1 <= row <= ARENA_ROWS and 1 <= col <= ARENA_COLS):
            return True
        return self.arena[row][col] != EMPTY

    def number_arena_cells(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Return ((row, col), (sister_row, col)) for the current number, or
        ((0, 0), (0, 0)) if no number is on the board."""
        if self.number_row == 0:
            return (0, 0), (0, 0)
        r1, r2 = _real_to_arena_rows(self.number_row)
        return (r1, self.number_col), (r2, self.number_col)
