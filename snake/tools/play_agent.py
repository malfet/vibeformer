"""Watch a trained policy play tiny_snake in your terminal.

Loads a `--ckpt` saved by train_bc.py or train_ppo.py and runs the agent
through one or more episodes, rendering the 12x12 board via ncurses.
The renderer also shows the teacher's recommended action so you can see
where the student disagrees.

Controls (during play):
    q / Q   quit
    p / SPC pause / unpause
    +       slower (more delay)
    -       faster
    g       toggle greedy / stochastic sampling
    r       reset episode

Run:
    python -m tools.play_agent --ckpt checkpoints/ppo01/ppo_tiny.pt
"""

from __future__ import annotations

import argparse
import curses
import time
from pathlib import Path

import numpy as np
import torch

import tiny_snake
from train_bc import Agent, select_device, _to_obs_symbolic


_PAIR_WALL = 1
_PAIR_BODY = 2
_PAIR_HEAD = 3
_PAIR_FOOD = 4
_PAIR_HEADER = 5
_PAIR_BAD = 6


def _init_colors() -> None:
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(_PAIR_WALL,   curses.COLOR_RED,    curses.COLOR_BLUE)
    curses.init_pair(_PAIR_BODY,   curses.COLOR_YELLOW, curses.COLOR_BLUE)
    curses.init_pair(_PAIR_HEAD,   curses.COLOR_BLACK,  curses.COLOR_YELLOW)
    curses.init_pair(_PAIR_FOOD,   curses.COLOR_WHITE,  curses.COLOR_BLUE)
    curses.init_pair(_PAIR_HEADER, curses.COLOR_WHITE,  curses.COLOR_BLACK)
    curses.init_pair(_PAIR_BAD,    curses.COLOR_BLACK,  curses.COLOR_RED)


_CELL_GLYPH = {
    tiny_snake.SYM_EMPTY: ("  ", 0),
    tiny_snake.SYM_WALL:  ("██", _PAIR_WALL),
    tiny_snake.SYM_BODY:  ("██", _PAIR_BODY),
    tiny_snake.SYM_HEAD:  ("()", _PAIR_HEAD),
    tiny_snake.SYM_FOOD:  ("**", _PAIR_FOOD),
}

_ACTION_NAME = {
    tiny_snake.STRAIGHT:   "STR",
    tiny_snake.TURN_LEFT:  "<L<",
    tiny_snake.TURN_RIGHT: ">R>",
}


def _safe_addstr(stdscr, y: int, x: int, s: str, attr: int = 0) -> None:
    try:
        stdscr.addstr(y, x, s, attr)
    except curses.error:
        pass


def _draw(stdscr, snake, last_student, last_teacher, mode: str,
          paused: bool, delay_ms: int, ep_idx: int, total_eps: int,
          probs: np.ndarray) -> bool:
    h, w = stdscr.getmaxyx()
    if h < 18 or w < 50:
        stdscr.erase()
        _safe_addstr(stdscr, 0, 0,
                     f"Terminal too small ({h}x{w}); need at least 18x50.")
        stdscr.refresh()
        return False

    header = (f" tiny-snake  ep {ep_idx}/{total_eps}  "
              f"score={snake.score:<3}  len={len(snake.body):<3}  "
              f"steps={snake.steps:<4}  "
              f"{mode:5s}  delay={delay_ms}ms  "
              f"{'[PAUSED] ' if paused else ''}"
              f"[arrows nope, q quit, p pause, g greedy/stoch, +- speed, r reset]")
    _safe_addstr(stdscr, 0, 0, header.ljust(w - 1)[:w - 1],
                 curses.color_pair(_PAIR_HEADER) | curses.A_BOLD)

    obs = snake.obs()
    # Render board starting at row 2.
    for r in range(tiny_snake.GRID_ROWS):
        for c in range(tiny_snake.GRID_COLS):
            glyph, pair = _CELL_GLYPH[int(obs[r, c])]
            attr = curses.color_pair(pair) if pair else 0
            _safe_addstr(stdscr, 2 + r, 2 + c * 2, glyph, attr)

    # Side panel: student vs teacher actions + probs.
    base_x = 2 + tiny_snake.GRID_COLS * 2 + 3
    _safe_addstr(stdscr, 2, base_x,
                 f"student: {_ACTION_NAME.get(last_student, '?')}",
                 curses.color_pair(_PAIR_HEADER) | curses.A_BOLD)
    _safe_addstr(stdscr, 3, base_x,
                 f"teacher: {_ACTION_NAME.get(last_teacher, '?')}",
                 curses.color_pair(_PAIR_HEADER))
    agree = last_student == last_teacher
    _safe_addstr(stdscr, 4, base_x,
                 "agree" if agree else "DISAGREE",
                 0 if agree else curses.color_pair(_PAIR_BAD) | curses.A_BOLD)
    _safe_addstr(stdscr, 6, base_x, "probs:",
                 curses.color_pair(_PAIR_HEADER))
    for i, name in enumerate(["STR", "<L<", ">R>"]):
        bar_n = int(round(probs[i] * 20))
        bar = "#" * bar_n + "." * (20 - bar_n)
        _safe_addstr(stdscr, 7 + i, base_x,
                     f"  {name} {probs[i]:.2f} {bar}")

    stdscr.refresh()
    return True


def _query_agent(agent, obs, device, greedy: bool):
    obs_t = _to_obs_symbolic(obs[None], device)
    with torch.no_grad():
        logits = agent.actor(agent.encode(obs_t))
        probs = logits.softmax(-1)[0].cpu().numpy()
        if greedy:
            a = int(logits.argmax(-1).item())
        else:
            a = int(torch.distributions.Categorical(logits=logits)
                    .sample().item())
    return a, probs


def _loop(stdscr, ckpt_path: Path, total_eps: int,
          delay_ms: int, greedy: bool, seed: int) -> None:
    curses.curs_set(0)
    _init_colors()
    stdscr.nodelay(True)

    device = select_device(False)
    agent = Agent(tiny_snake.NUM_ACTIONS,
                  in_channels=tiny_snake.SYM_NUM_TYPES,
                  obs_size=tiny_snake.TinySnakeVecEnv.OBS_SHAPE,
                  width=1.0).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device,
                      weights_only=False)
    # The ckpt may have been saved at a different width. Re-build if mismatch.
    state = ckpt["agent"]
    try:
        agent.load_state_dict(state)
    except RuntimeError:
        # Probe the saved actor's input dim to derive the width.
        w_actor = state["actor.weight"]
        fc_dim = w_actor.shape[1]
        guess_width = max(0.0625, fc_dim / 512.0)
        agent = Agent(tiny_snake.NUM_ACTIONS,
                      in_channels=tiny_snake.SYM_NUM_TYPES,
                      obs_size=tiny_snake.TinySnakeVecEnv.OBS_SHAPE,
                      width=guess_width).to(device)
        agent.load_state_dict(state)
    agent.eval()

    for ep_idx in range(1, total_eps + 1):
        snake = tiny_snake.TinySnake(max_steps=1000, rng_seed=seed + ep_idx)
        snake.reset()
        last_student = tiny_snake.STRAIGHT
        last_teacher = tiny_snake.STRAIGHT
        probs = np.array([0.0, 0.0, 0.0])
        paused = False
        mode = "greedy" if greedy else "stoch"

        _draw(stdscr, snake, last_student, last_teacher, mode,
              paused, delay_ms, ep_idx, total_eps, probs)
        while True:
            end_t = time.monotonic() + delay_ms / 1000.0
            while time.monotonic() < end_t:
                ch = stdscr.getch()
                if ch == -1:
                    time.sleep(0.005); continue
                if ch in (ord('q'), ord('Q')):
                    return
                if ch in (ord('p'), ord('P'), ord(' ')):
                    paused = not paused
                elif ch == ord('+'):
                    delay_ms = min(2000, delay_ms + 50)
                elif ch == ord('-'):
                    delay_ms = max(20, delay_ms - 50)
                elif ch in (ord('g'), ord('G')):
                    greedy = not greedy
                    mode = "greedy" if greedy else "stoch"
                elif ch in (ord('r'), ord('R')):
                    snake = tiny_snake.TinySnake(
                        max_steps=1000, rng_seed=seed + ep_idx + 1000)
                    snake.reset()

            if paused:
                _draw(stdscr, snake, last_student, last_teacher, mode,
                      paused, delay_ms, ep_idx, total_eps, probs)
                continue

            last_teacher = tiny_snake.heuristic_action(snake)
            last_student, probs = _query_agent(agent, snake.obs(),
                                               device, greedy)
            r = snake.step(last_student)
            _draw(stdscr, snake, last_student, last_teacher, mode,
                  paused, delay_ms, ep_idx, total_eps, probs)
            if r.done:
                msg = (f"  ep {ep_idx} done: score={snake.score} "
                       f"len={len(snake.body)} "
                       f"{'(died)' if r.info['died'] else '(truncated)'}  "
                       f"press SPACE for next episode, q to quit")
                _safe_addstr(stdscr, tiny_snake.GRID_ROWS + 3, 0, msg,
                             curses.A_BOLD | curses.color_pair(_PAIR_HEADER))
                stdscr.refresh()
                while True:
                    ch = stdscr.getch()
                    if ch in (ord('q'), ord('Q')):
                        return
                    if ch in (ord(' '), ord('\n')):
                        break
                break


def main() -> None:
    p = argparse.ArgumentParser(
        description="Watch a trained tiny-snake policy play in your terminal."
    )
    p.add_argument("--ckpt", type=Path, required=True,
                   help="path to a BC or PPO checkpoint")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--delay-ms", type=int, default=200,
                   help="step interval; smaller = faster")
    p.add_argument("--greedy", action="store_true",
                   help="argmax actions (default: stochastic sampling)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.ckpt.exists():
        raise SystemExit(f"no such checkpoint: {args.ckpt}")
    curses.wrapper(_loop, args.ckpt, args.episodes,
                   args.delay_ms, args.greedy, args.seed)


if __name__ == "__main__":
    main()
