"""Hand-coded baseline policy + optional matplotlib viewer.

Digger digs through everything but bags, so no A* needed -- the agent
greedily takes the action that moves it one Manhattan step closer to
the nearest emerald, with a small overlay of threat-aware behaviour:

  - Dodge:  if a monster is in the same row/col within `dodge_range`
            tiles, escape perpendicularly (toward an emerald if one is
            available along the escape axis, otherwise just away).
  - Shoot:  if a monster is collinear within `fire_range` AND the
            digger is already facing toward it (tracked across steps
            via `prev_dir`), press FIRE. The bullet travels along the
            facing direction, so this hits the threat.

Headless usage (compares to the previous baseline):
    python -m tools.heuristic_agent --episodes 5 --no-episodic-life

Interactive viewer (matplotlib window showing live gameplay with
detected GameState overlaid as markers, plus title with current
action / score / lives):
    python -m tools.heuristic_agent --live
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running this file directly (`python tools/heuristic_agent.py`)
# by ensuring the project root is on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from digger_env import DiggerEnv
from tools.game_state import MHEIGHT, MWIDTH, render_overlay, tile_center
from tools.symbolic_env import SymbolicDiggerEnv

ACTION_NAMES = ["NOOP", "LEFT", "RIGHT", "UP", "DOWN", "FIRE"]


def _direction_toward(target_row: int, target_col: int,
                       dr: int, dc: int, prefer_col_first: bool) -> int:
    """Return the move action that reduces (dr, dc) -> (target_row, target_col)."""
    if prefer_col_first:
        if target_col < dc: return DiggerEnv.LEFT
        if target_col > dc: return DiggerEnv.RIGHT
        if target_row < dr: return DiggerEnv.UP
        if target_row > dr: return DiggerEnv.DOWN
    else:
        if target_row < dr: return DiggerEnv.UP
        if target_row > dr: return DiggerEnv.DOWN
        if target_col < dc: return DiggerEnv.LEFT
        if target_col > dc: return DiggerEnv.RIGHT
    return DiggerEnv.NOOP


def _greedy_step(state) -> int:
    """One stateless greedy Manhattan step toward the nearest emerald."""
    if state.digger is None or not state.digger.present:
        return DiggerEnv.NOOP
    em_rows, em_cols = np.where(state.emeralds)
    if em_rows.size == 0:
        return DiggerEnv.NOOP
    dr, dc = state.digger.row, state.digger.col
    distances = np.abs(em_rows - dr) + np.abs(em_cols - dc)
    i = int(np.argmin(distances))
    er, ec = int(em_rows[i]), int(em_cols[i])
    return _direction_toward(er, ec, dr, dc,
                              prefer_col_first=abs(ec - dc) >= abs(er - dr))


class GreedyEmerald:
    """Per-step greedy nearest-emerald chaser with anti-jitter stickiness.

    Each step we pick the action that takes us one Manhattan step closer
    to the currently-nearest emerald, so we adapt naturally when the
    digger crosses a tile boundary and a different emerald becomes
    closest.

    But when the digger's CV-detected position briefly snaps to the tile
    it's actively moving toward (mid-frame straddle), `_greedy_step`
    sees "we're on the target" and returns NOOP -- even though the
    emerald hasn't actually been consumed. Without correction the agent
    appears to hesitate at the edge of every emerald tile. So: if the
    one-shot decision is NOOP but there are still emeralds in the grid,
    fall back to the last directional action we took. The fallback
    naturally clears once we've actually consumed the emerald (no more
    near-tile jitter to trigger the NOOP path).

    Call .reset() between episodes.
    """

    MOVE_ACTIONS = (DiggerEnv.LEFT, DiggerEnv.RIGHT,
                    DiggerEnv.UP, DiggerEnv.DOWN)

    def __init__(self):
        self.last_dir: int = DiggerEnv.NOOP

    def reset(self) -> None:
        self.last_dir = DiggerEnv.NOOP

    def __call__(self, state) -> int:
        action = _greedy_step(state)
        if action == DiggerEnv.NOOP and state.emeralds.any() \
                and self.last_dir != DiggerEnv.NOOP:
            return self.last_dir
        if action in self.MOVE_ACTIONS:
            self.last_dir = action
        return action


def greedy_emerald(state) -> int:
    """Stateless one-shot wrapper. For repeated use prefer GreedyEmerald()."""
    return _greedy_step(state)


class SmartHeuristic:
    """Greedy emerald-chaser with monster dodging + opportunistic firing.

    Keeps `prev_dir` so we know which way the digger is facing (needed
    for firing -- the bullet travels in the facing direction). Reset
    via .reset() between episodes.
    """

    def __init__(self, dodge_range: int = 2, fire_range: int = 5):
        self.dodge_range = dodge_range
        self.fire_range = fire_range
        self.prev_dir: int = DiggerEnv.NOOP
        self._chaser = GreedyEmerald()

    def reset(self) -> None:
        self.prev_dir = DiggerEnv.NOOP
        self._chaser.reset()

    def __call__(self, state) -> int:
        action = self._decide(state)
        if action in (DiggerEnv.LEFT, DiggerEnv.RIGHT,
                      DiggerEnv.UP, DiggerEnv.DOWN):
            self.prev_dir = action
        return action

    def _decide(self, state) -> int:
        if state.digger is None or not state.digger.present:
            return DiggerEnv.NOOP
        dr, dc = state.digger.row, state.digger.col

        # ---- Threat assessment ----
        # A threat is a monster collinear with the digger within fire_range.
        for m in state.monsters:
            mr, mc = m.row, m.col
            if mr == dr and 0 < abs(mc - dc) <= self.fire_range:
                # Monster in same row. Are we already aimed at it?
                facing_toward = (mc < dc and self.prev_dir == DiggerEnv.LEFT) \
                              or (mc > dc and self.prev_dir == DiggerEnv.RIGHT)
                if facing_toward:
                    return DiggerEnv.FIRE
                if abs(mc - dc) <= self.dodge_range:
                    # Too close to safely turn-and-fire; dodge vertically.
                    return self._escape_axis(state, dr, dc, axis="row")
            if mc == dc and 0 < abs(mr - dr) <= self.fire_range:
                facing_toward = (mr < dr and self.prev_dir == DiggerEnv.UP) \
                              or (mr > dr and self.prev_dir == DiggerEnv.DOWN)
                if facing_toward:
                    return DiggerEnv.FIRE
                if abs(mr - dr) <= self.dodge_range:
                    return self._escape_axis(state, dr, dc, axis="col")

        return self._chaser(state)

    def _escape_axis(self, state, dr: int, dc: int, axis: str) -> int:
        """Move perpendicular to the threat axis, preferring toward an emerald.

        axis="row": threat is in the same row as us, escape vertically.
        axis="col": threat is in the same col as us, escape horizontally.
        """
        em_rows, em_cols = np.where(state.emeralds)
        if axis == "row":
            # Pick UP or DOWN. Prefer the side with an emerald nearby.
            up_em = ((em_rows < dr).sum() if em_rows.size else 0)
            dn_em = ((em_rows > dr).sum() if em_rows.size else 0)
            if up_em > 0 and dr > 0:
                return DiggerEnv.UP
            if dn_em > 0 and dr < MHEIGHT - 1:
                return DiggerEnv.DOWN
            return DiggerEnv.UP if dr > 0 else DiggerEnv.DOWN
        else:
            lf_em = ((em_cols < dc).sum() if em_cols.size else 0)
            rt_em = ((em_cols > dc).sum() if em_cols.size else 0)
            if lf_em > 0 and dc > 0:
                return DiggerEnv.LEFT
            if rt_em > 0 and dc < MWIDTH - 1:
                return DiggerEnv.RIGHT
            return DiggerEnv.LEFT if dc > 0 else DiggerEnv.RIGHT


# ---- Runners --------------------------------------------------------------

def run_headless(args) -> None:
    env = SymbolicDiggerEnv(max_steps=10**9,
                            episodic_life=not args.no_episodic_life)
    if args.smart:
        policy = SmartHeuristic(args.dodge_range, args.fire_range)
        pol_name = f"smart(dodge={args.dodge_range}, fire={args.fire_range})"
    else:
        policy = GreedyEmerald()
        pol_name = "greedy(anti-jitter)"

    scores: list[int] = []
    lengths: list[int] = []
    t0 = time.monotonic()

    for ep in range(args.episodes):
        env.reset()
        policy.reset()
        ep_score = 0
        ep_len = 0
        for step in range(args.max_steps):
            action = policy(env._last_state)
            done = False
            for _ in range(args.frame_skip):
                obs, r, done_, info = env.step(action)
                if done_:
                    done = True
                    break
            ep_score = info.get("score", 0)
            ep_len = step + 1
            if done:
                break
        scores.append(ep_score)
        lengths.append(ep_len)
        print(f"  ep {ep + 1}/{args.episodes}: score={ep_score}  steps={ep_len}",
              flush=True)

    elapsed = time.monotonic() - t0
    arr = np.array(scores)
    print()
    print(f"=== heuristic baseline: {pol_name} ===")
    print(f"  episodes: {args.episodes}")
    print(f"  mean score:  {arr.mean():.1f}")
    print(f"  median:      {np.median(arr):.0f}")
    print(f"  min/max:     {arr.min()} / {arr.max()}")
    print(f"  mean length: {np.mean(lengths):.0f} steps")
    print(f"  wall time:   {elapsed:.1f}s")
    env.close()


def run_live(args) -> None:
    """Open a matplotlib window, render gameplay with GameState overlay."""
    import matplotlib.pyplot as plt

    env = SymbolicDiggerEnv(max_steps=10**9,
                            episodic_life=not args.no_episodic_life)
    policy = (SmartHeuristic(args.dodge_range, args.fire_range)
              if args.smart else GreedyEmerald())

    obs = env.reset()
    raw = env._env._core.get_frame()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_axis_off()
    if args.overlay:
        img = ax.imshow(render_overlay(
            raw, env._last_state,
            show_digger=args.show_digger,
            show_emeralds=args.show_emeralds))
    else:
        img = ax.imshow(raw[..., :3])
    fig.tight_layout(pad=0)
    fig.canvas.manager.set_window_title("DIGGER heuristic (live)")
    plt.ion()
    plt.show()

    target_fps = 70.087
    target_dt = (1.0 / target_fps) * args.frame_skip
    last_wall = time.monotonic()
    fps_ema = 1.0 / target_dt
    step_no = 0
    seen_alive = False
    policy.reset()

    while plt.fignum_exists(fig.number):
        now = time.monotonic()
        elapsed = now - last_wall
        last_wall = now

        action = policy(env._last_state)

        done = False
        info = {}
        for _ in range(args.frame_skip):
            obs, r, done_, info = env.step(action)
            if done_: done = True; break
        step_no += args.frame_skip

        raw = env._env._core.get_frame()
        if args.overlay:
            img.set_data(render_overlay(
                raw, env._last_state,
                show_digger=args.show_digger,
                show_emeralds=args.show_emeralds))
        else:
            img.set_data(raw[..., :3])
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        score = info.get("score", 0)
        lives = info.get("lives", 0)
        if lives > 0: seen_alive = True
        if elapsed > 0:
            fps_ema = 0.9 * fps_ema + 0.1 * (args.frame_skip / elapsed)
        fig.canvas.manager.set_window_title(
            f"DIGGER heuristic -- score {score:>6d} -- lives {lives} -- "
            f"a={ACTION_NAMES[action]:<5s} -- {fps_ema:5.1f} fps -- step {step_no}"
        )

        if done and seen_alive and lives == 0:
            print(f"game over: score {score} steps {step_no}")
            break
        slack = target_dt - (time.monotonic() - now)
        if slack > 0: time.sleep(slack)

    plt.close("all")
    env.close()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--no-episodic-life", action="store_true")
    p.add_argument("--smart", action="store_true",
                   help="enable monster dodge + opportunistic FIRE")
    p.add_argument("--dodge-range", type=int, default=2)
    p.add_argument("--fire-range", type=int, default=5)
    p.add_argument("--live", action="store_true",
                   help="open a matplotlib window and watch the policy play")
    p.add_argument("--overlay", action="store_true",
                   help="draw small markers for detected monsters/bags "
                        "(off by default; the digger sprite is already visible)")
    p.add_argument("--show-digger", action="store_true",
                   help="with --overlay, also mark the digger position")
    p.add_argument("--show-emeralds", action="store_true",
                   help="with --overlay, mark every detected emerald")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.live:
        run_live(args)
    else:
        run_headless(args)


if __name__ == "__main__":
    main()
