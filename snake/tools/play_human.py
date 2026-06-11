"""Human-play interface for nibbles_sim using ncurses.

Renders the 50-row arena via Unicode half-blocks (`U+2580 U+2584 U+2588`)
so a single terminal row shows two arena sub-rows — the same trick QBASIC
used with `chr$(220) chr$(223) chr$(219)`. Arrow keys steer; q quits.

Run:
    python -m tools.play_human
"""

from __future__ import annotations

import argparse
import curses
import time

from nibbles_sim import (
    NibblesGame,
    ARENA_COLS,
    EMPTY, WALL, SNAKE,
    NOOP, UP, DOWN, LEFT, RIGHT,
)


_PAIR_SNAKE_BG = 1
_PAIR_WALL_BG = 2
_PAIR_SNAKE_WALL = 3
_PAIR_WALL_SNAKE = 4
_PAIR_NUMBER = 5
_PAIR_HEADER = 6


def _init_colors() -> None:
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(_PAIR_SNAKE_BG,   curses.COLOR_YELLOW, curses.COLOR_BLUE)
    curses.init_pair(_PAIR_WALL_BG,    curses.COLOR_RED,    curses.COLOR_BLUE)
    curses.init_pair(_PAIR_SNAKE_WALL, curses.COLOR_YELLOW, curses.COLOR_RED)
    curses.init_pair(_PAIR_WALL_SNAKE, curses.COLOR_RED,    curses.COLOR_YELLOW)
    curses.init_pair(_PAIR_NUMBER,     curses.COLOR_WHITE,  curses.COLOR_BLUE)
    curses.init_pair(_PAIR_HEADER,     curses.COLOR_WHITE,  curses.COLOR_BLACK)


def _cell_pair(t: int) -> int:
    if t == WALL:
        return _PAIR_WALL_BG
    if t == SNAKE:
        return _PAIR_SNAKE_BG
    return 0


def _glyph(top: int, bot: int):
    """Return (char, color_pair_id) for a stacked cell pair."""
    if top == EMPTY and bot == EMPTY:
        return ' ', 0
    if top == EMPTY:
        return '▄', _cell_pair(bot)            # ▄ lower half
    if bot == EMPTY:
        return '▀', _cell_pair(top)            # ▀ upper half
    if top == bot:
        return '█', _cell_pair(top)            # █ full block
    if top == SNAKE and bot == WALL:
        return '▀', _PAIR_SNAKE_WALL
    return '▀', _PAIR_WALL_SNAKE


def _safe_addstr(stdscr, y: int, x: int, s: str, attr: int = 0) -> None:
    try:
        stdscr.addstr(y, x, s, attr)
    except curses.error:
        # Writing to the bottom-right cell auto-advances and raises; ignore.
        pass


def _draw_frame(stdscr, game: NibblesGame, paused: bool) -> bool:
    h, w = stdscr.getmaxyx()
    if h < 27 or w < 80:
        stdscr.erase()
        msg = f"Terminal too small ({h}x{w}); need at least 27x80."
        _safe_addstr(stdscr, 0, 0, msg)
        stdscr.refresh()
        return False

    header = (
        f" Nibbles  level={game.level:<2}  score={game.snake.score:<5}"
        f"  lives={game.snake.lives}  next={game.number_value}"
        f"  len={game.length:<4}  "
        f"{'[PAUSED] ' if paused else ''}"
        f"arrows=move  p=pause  r=reset  q=quit"
    )
    _safe_addstr(stdscr, 0, 0, header.ljust(w - 1)[:w - 1],
                 curses.color_pair(_PAIR_HEADER) | curses.A_BOLD)

    arena = game.arena
    for text_row in range(1, 26):
        ar_top = 2 * text_row - 1
        ar_bot = 2 * text_row
        for col in range(1, ARENA_COLS + 1):
            ch, pair = _glyph(arena[ar_top][col], arena[ar_bot][col])
            attr = curses.color_pair(pair) if pair else 0
            _safe_addstr(stdscr, text_row, col - 1, ch, attr)

    if game.number_row != 0:
        _safe_addstr(stdscr, game.number_row, game.number_col - 1,
                     str(game.number_value),
                     curses.color_pair(_PAIR_NUMBER) | curses.A_BOLD)

    stdscr.refresh()
    return True


_KEY_TO_ACTION = {
    curses.KEY_UP: UP,
    curses.KEY_DOWN: DOWN,
    curses.KEY_LEFT: LEFT,
    curses.KEY_RIGHT: RIGHT,
}


def _drain_input(stdscr):
    """Yield queued keys, returning -1 once exhausted."""
    while True:
        ch = stdscr.getch()
        if ch == -1:
            return
        yield ch


def _loop(stdscr, max_steps: int, tick_ms: int, seed: int):
    curses.curs_set(0)
    _init_colors()
    stdscr.nodelay(True)

    game = NibblesGame(max_steps=max_steps, rng_seed=seed)
    action = NOOP
    paused = False
    last_info = None
    game_over = False

    _draw_frame(stdscr, game, paused)

    while True:
        # Poll input over the tick window so keys feel responsive.
        end_t = time.monotonic() + tick_ms / 1000.0
        while time.monotonic() < end_t:
            for ch in _drain_input(stdscr):
                if ch in (ord('q'), ord('Q')):
                    return last_info
                if ch in (ord('p'), ord('P'), ord(' ')):
                    paused = not paused
                elif ch in (ord('r'), ord('R')):
                    game = NibblesGame(max_steps=max_steps, rng_seed=seed)
                    action = NOOP
                    paused = False
                    game_over = False
                elif ch in _KEY_TO_ACTION and not game_over:
                    action = _KEY_TO_ACTION[ch]
            time.sleep(0.005)

        if game_over or paused:
            _draw_frame(stdscr, game, paused or game_over)
            continue

        info = game.step(action)
        last_info = info
        _draw_frame(stdscr, game, paused)

        if info.game_over or info.truncated:
            game_over = True
            msg = (f"  GAME OVER  final score={game.snake.score} "
                   f"  press r to restart, q to quit")
            _safe_addstr(stdscr, 26, 0, msg,
                         curses.A_BOLD | curses.color_pair(_PAIR_HEADER))
            stdscr.refresh()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Play QBasic Nibbles in your terminal."
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=10 ** 9)
    p.add_argument("--tick-ms", type=int, default=80,
                   help="Frame interval; lower = faster snake.")
    args = p.parse_args()

    last = curses.wrapper(_loop, args.max_steps, args.tick_ms, args.seed)
    if last is not None:
        print(f"Final: score={last.score} lives={last.lives} "
              f"level={last.level}")


if __name__ == "__main__":
    main()
