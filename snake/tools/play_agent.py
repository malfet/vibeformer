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
from train_bc import (Agent, IterAgent, select_device, _to_obs_symbolic,
                      _to_obs_dist)


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
    rows, cols = obs.shape
    # Render board starting at row 2.
    for r in range(rows):
        for c in range(cols):
            glyph, pair = _CELL_GLYPH[int(obs[r, c])]
            attr = curses.color_pair(pair) if pair else 0
            _safe_addstr(stdscr, 2 + r, 2 + c * 2, glyph, attr)

    # Side panel: student vs teacher actions + probs.
    base_x = 2 + cols * 2 + 3
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


def _query_agent(agent, obs, device, greedy: bool, to_obs_fn):
    obs_t = to_obs_fn(obs[None], device)
    with torch.no_grad():
        logits = agent.actor(agent.encode(obs_t))
        probs = logits.softmax(-1)[0].cpu().numpy()
        if greedy:
            a = int(logits.argmax(-1).item())
        else:
            a = int(torch.distributions.Categorical(logits=logits)
                    .sample().item())
    return a, probs


def load_agent_for_play(ckpt_path: Path, device, canvas_override: int = 0):
    """Build + load an Agent from a BC/PPO checkpoint, detecting the obs
    mode (bare symbolic / 6-ch dist / 11-ch full / canonical) and the
    teacher it was trained against. `canvas_override` only applies to the
    fully-convolutional iterator arch (canvas-agnostic weights).

    Returns (agent, obs_fn, to_obs_fn, teacher_name).
    """
    ckpt = torch.load(str(ckpt_path), map_location=device,
                      weights_only=False)
    state = ckpt["agent"]
    cfg = ckpt.get("config", {})

    # Detect the obs mode from saved config; if missing, infer from the
    # encoder's first-conv input channels (11 -> full; 6 -> dist; 5 -> bare).
    first_conv_in = None
    if "encoder.0.0.weight" in state:
        first_conv_in = state["encoder.0.0.weight"].shape[1]
    ego = bool(cfg.get("egocentric", False))

    def _wrap(extract):
        def fn(s):
            f = extract(s)
            if ego:
                f = tiny_snake.egocentric_obs(f, s.head)
            return tiny_snake.quantize_obs(f)
        return fn

    if bool(cfg.get("canonical", False)):
        with_dist = not bool(cfg.get("canonical_no_dist", False))
        obs_fn = lambda s: tiny_snake.quantize_obs(
            tiny_snake.extract_obs_canonical(s, with_dist))
        to_obs_fn = _to_obs_dist
        in_ch = tiny_snake.CANONICAL_OBS_CHANNELS - (0 if with_dist else 1)
        canvas = int(cfg.get("canvas", 49))
        if canvas_override and cfg.get("arch", "cnn") == "iter":
            canvas = canvas_override
        if cfg.get("arch", "cnn") == "iter":
            agent = IterAgent(tiny_snake.NUM_ACTIONS, in_channels=in_ch,
                              obs_size=(canvas, canvas),
                              channels=int(cfg.get("iter_channels", 96)),
                              iters=int(cfg.get("iter_steps", 16))
                              ).to(device)
        else:
            agent = Agent(tiny_snake.NUM_ACTIONS, in_channels=in_ch,
                          obs_size=(canvas, canvas),
                          width=float(cfg.get("encoder_width", 1.0)),
                          micro=bool(cfg.get("micro_cnn", False))).to(device)
        agent.load_state_dict(state)
        agent.eval()
        return agent, obs_fn, to_obs_fn, cfg.get("teacher", "bfs")

    if bool(cfg.get("extra_features", False)) \
            or first_conv_in == tiny_snake.FULL_OBS_CHANNELS:
        obs_fn = _wrap(tiny_snake.extract_obs_full)
        to_obs_fn = _to_obs_dist
        in_ch = tiny_snake.FULL_OBS_CHANNELS
    elif bool(cfg.get("dist_feature", False)) or first_conv_in == 6:
        obs_fn = _wrap(tiny_snake.extract_obs_with_dist)
        to_obs_fn = _to_obs_dist
        in_ch = tiny_snake.SYM_NUM_TYPES + 1
    else:
        obs_fn = lambda snake: snake.obs()
        to_obs_fn = _to_obs_symbolic
        in_ch = tiny_snake.SYM_NUM_TYPES

    canvas = int(cfg.get("canvas", 12))

    def _build(micro: bool, width: float):
        return Agent(tiny_snake.NUM_ACTIONS,
                     in_channels=in_ch,
                     obs_size=(canvas, canvas),
                     width=width, micro=micro).to(device)

    # Prefer the saved config's flags when present.
    agent = _build(micro=bool(cfg.get("micro_cnn", False)),
                   width=float(cfg.get("encoder_width", 1.0)))
    try:
        agent.load_state_dict(state)
    except RuntimeError:
        # Fall back to inferring the encoder type from the saved actor's
        # input dim. fc_dim == 32 -> micro CNN; otherwise scale width.
        w_actor = state["actor.weight"]
        fc_dim = w_actor.shape[1]
        if fc_dim == 32:
            agent = _build(micro=True, width=1.0)
        else:
            agent = _build(micro=False,
                           width=max(0.0625, fc_dim / 512.0))
        agent.load_state_dict(state)
    agent.eval()
    return agent, obs_fn, to_obs_fn, cfg.get("teacher", "bfs")


def _teacher_fn(name: str):
    return (tiny_snake.safe_heuristic_action if name == "safe"
            else tiny_snake.heuristic_action)


def _loop(stdscr, ckpt_path: Path | None, total_eps: int,
          delay_ms: int, greedy: bool, seed: int,
          teacher_play: str | None = None, vs: str = "auto",
          env_kwargs: dict | None = None) -> None:
    curses.curs_set(0)
    _init_colors()
    stdscr.nodelay(True)

    env_kwargs = dict(env_kwargs or {})
    device = select_device(False)
    agent = None
    teacher_mode = teacher_play is not None
    # Default obs path: bare (H, W) int grid -> 5-channel one-hot.
    obs_fn = lambda snake: snake.obs()
    to_obs_fn = _to_obs_symbolic
    if teacher_mode:
        compare_fn = _teacher_fn(teacher_play)
    else:
        agent, obs_fn, to_obs_fn, ckpt_teacher = load_agent_for_play(
            ckpt_path, device,
            canvas_override=env_kwargs.get("canvas_rows", 0))
        # Play on the canvas the ckpt was trained for.
        ckpt = torch.load(str(ckpt_path), map_location="cpu",
                          weights_only=False)
        canvas = int(ckpt.get("config", {}).get("canvas", 12))
        env_kwargs.setdefault("canvas_rows", canvas)
        env_kwargs.setdefault("canvas_cols", canvas)
        # Comparison panel: the teacher the ckpt was trained against,
        # unless overridden with --vs.
        compare_fn = _teacher_fn(vs if vs != "auto" else ckpt_teacher)

    for ep_idx in range(1, total_eps + 1):
        snake = tiny_snake.TinySnake(max_steps=1000, rng_seed=seed + ep_idx,
                                     **env_kwargs)
        snake.reset()
        last_student = tiny_snake.STRAIGHT
        last_teacher = tiny_snake.STRAIGHT
        probs = np.array([0.0, 0.0, 0.0])
        paused = False
        if teacher_mode:
            mode = teacher_play.upper()
        else:
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
                        max_steps=1000, rng_seed=seed + ep_idx + 1000,
                        **env_kwargs)
                    snake.reset()

            if paused:
                _draw(stdscr, snake, last_student, last_teacher, mode,
                      paused, delay_ms, ep_idx, total_eps, probs)
                continue

            last_teacher = compare_fn(snake)
            if teacher_mode:
                last_student = last_teacher
                probs = np.zeros(3, dtype=np.float32)
                probs[last_teacher] = 1.0
            else:
                last_student, probs = _query_agent(
                    agent, obs_fn(snake), device, greedy, to_obs_fn)
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
    p.add_argument("--ckpt", type=Path, default=None,
                   help="path to a BC or PPO checkpoint (omit when --teacher)")
    p.add_argument("--teacher", nargs="?", const="bfs",
                   choices=["bfs", "safe"], default=None,
                   help="play a scripted teacher instead of a model (no "
                        "ckpt needed). Bare --teacher = the BFS teacher; "
                        "--teacher safe = the tail-safe teacher.")
    p.add_argument("--vs", choices=["auto", "bfs", "safe"], default="auto",
                   help="which teacher the side panel compares the student "
                        "against. auto (default) = the teacher recorded in "
                        "the checkpoint's config.")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--apples", type=int, default=1,
                   help="simultaneous apples (0 = survival-only)")
    p.add_argument("--canvas", type=int, default=0,
                   help="canvas size; default = ckpt's canvas (or 12)")
    p.add_argument("--field-min", type=int, default=0,
                   help="with --field-max, random field size per episode")
    p.add_argument("--field-max", type=int, default=0)
    p.add_argument("--delay-ms", type=int, default=200,
                   help="step interval; smaller = faster")
    p.add_argument("--greedy", action="store_true",
                   help="argmax actions (default: stochastic sampling)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.teacher is None:
        if args.ckpt is None or not args.ckpt.exists():
            raise SystemExit(
                "need either --teacher or --ckpt <existing-path>")
    env_kwargs: dict = {"num_apples": args.apples}
    if args.canvas > 0:
        if args.ckpt is not None and args.ckpt.exists():
            _cfg = torch.load(str(args.ckpt), map_location="cpu",
                              weights_only=False).get("config", {})
            _ck_canvas = int(_cfg.get("canvas", 12))
            if args.canvas != _ck_canvas \
                    and _cfg.get("arch", "cnn") != "iter":
                raise SystemExit(
                    f"--canvas {args.canvas} but the checkpoint was trained "
                    f"at canvas {_ck_canvas}; the FC layer bakes the canvas "
                    f"into the weights (iterator-arch ckpts are exempt). "
                    f"Omit --canvas, or use --field-min/--field-max to "
                    f"vary the playable area inside the canvas instead.")
        env_kwargs["canvas_rows"] = args.canvas
        env_kwargs["canvas_cols"] = args.canvas
    if args.field_min > 0:
        env_kwargs["field_range"] = (args.field_min, args.field_max)
    curses.wrapper(_loop, args.ckpt, args.episodes,
                   args.delay_ms, args.greedy, args.seed,
                   teacher_play=args.teacher, vs=args.vs,
                   env_kwargs=env_kwargs)


if __name__ == "__main__":
    main()
