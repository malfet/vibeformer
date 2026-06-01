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
import collections
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


# Per-action (drow, dcol) tile offsets.
_ACTION_DELTA: dict[int, tuple[int, int]] = {
    DiggerEnv.LEFT:  (0, -1),
    DiggerEnv.RIGHT: (0,  1),
    DiggerEnv.UP:    (-1, 0),
    DiggerEnv.DOWN:  ( 1, 0),
}


def _greedy_step(state) -> int:
    """One stateless greedy Manhattan step toward the nearest emerald.

    Bag-aware: intact bags are obstacles. The digger CAN push a bag
    horizontally if there's empty space behind it, but the push wastes
    several frames and risks shoving the bag onto a ledge where it
    falls and breaks (and can land on the digger on the way back).
    So we route around: when the preferred axis step would land on a
    bag tile, try the alternate axis. Only if BOTH axes are blocked
    do we accept the push, since otherwise the digger could be stuck
    forever facing a single bag.
    """
    if state.digger is None or not state.digger.present:
        return DiggerEnv.NOOP
    em_rows, em_cols = np.where(state.emeralds)
    if em_rows.size == 0:
        return DiggerEnv.NOOP
    dr, dc = state.digger.row, state.digger.col
    distances = np.abs(em_rows - dr) + np.abs(em_cols - dc)
    i = int(np.argmin(distances))
    er, ec = int(em_rows[i]), int(em_cols[i])

    bag_tiles = {(b.row, b.col) for b in state.bags}

    # Per-axis candidate actions (None if already aligned on that axis).
    col_act = (DiggerEnv.LEFT if ec < dc else
               DiggerEnv.RIGHT if ec > dc else None)
    row_act = (DiggerEnv.UP if er < dr else
               DiggerEnv.DOWN if er > dr else None)
    # Step on the longer-distance axis first; ties favour columns to match
    # the historical behaviour of _direction_toward(prefer_col_first=True).
    order = ([col_act, row_act] if abs(ec - dc) >= abs(er - dr)
             else [row_act, col_act])
    order = [a for a in order if a is not None]
    if not order:
        return DiggerEnv.NOOP

    for action in order:
        ddr, ddc = _ACTION_DELTA[action]
        if (dr + ddr, dc + ddc) not in bag_tiles:
            return action
    # Both axes hit bags -- accept the push on the preferred axis.
    return order[0]


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
    for firing -- the bullet travels in the facing direction). Also
    tracks an explicit fire cooldown: after FIRE the in-game turret
    is "down" for ~200 game frames (visible as the lowered turret on
    the digger-life icon in the score bar), during which further FIRE
    presses are no-ops. Without this counter the agent mashes FIRE on
    every step it sees a line-of-sight monster, wasting most of those
    steps as no-ops while a monster closes in. Reset via .reset()
    between episodes.
    """

    # Default in *agent-steps* (== game-frames / frame_skip). With the
    # default frame_skip=4, 50 agent-steps ≈ 200 game frames, matching
    # the observed turret-recharge duration on the score-bar icon.
    DEFAULT_FIRE_COOLDOWN_STEPS = 50

    def __init__(self, dodge_range: int = 2, fire_range: int = 5,
                 fire_cooldown_steps: int = DEFAULT_FIRE_COOLDOWN_STEPS):
        # `dodge_range` is retained for CLI backward-compat but no longer
        # used: the new lex-scored selection caps safety at distance 3 and
        # blends emerald-chase in continuously.
        self.dodge_range = dodge_range
        self.fire_range = fire_range
        self.fire_cooldown_steps = fire_cooldown_steps
        self.prev_dir: int = DiggerEnv.NOOP
        self._fire_cd: int = 0

    def reset(self) -> None:
        self.prev_dir = DiggerEnv.NOOP
        self._fire_cd = 0

    def __call__(self, state) -> int:
        action = self._decide(state)
        if action == DiggerEnv.FIRE:
            self._fire_cd = self.fire_cooldown_steps
        elif self._fire_cd > 0:
            self._fire_cd -= 1
        if action in (DiggerEnv.LEFT, DiggerEnv.RIGHT,
                      DiggerEnv.UP, DiggerEnv.DOWN):
            self.prev_dir = action
        return action

    # Movement candidates evaluated by the score function. FIRE is handled
    # separately via the line-of-sight check below.
    _MOVE_CANDIDATES: tuple[tuple[int, int, int], ...] = (
        (DiggerEnv.NOOP,  0,  0),
        (DiggerEnv.LEFT,  0, -1),
        (DiggerEnv.RIGHT, 0,  1),
        (DiggerEnv.UP,   -1,  0),
        (DiggerEnv.DOWN,  1,  0),
    )

    _MAX_TUNNEL_DIST: int = MWIDTH * MHEIGHT + 1  # sentinel for "unreachable"

    def _decide(self, state) -> int:
        if state.digger is None or not state.digger.present:
            return DiggerEnv.NOOP
        dr, dc = state.digger.row, state.digger.col

        # ---- FIRE opportunity ------------------------------------------
        # Replaces the old "any threat in fire_range" check with a real
        # line-of-sight scan: bullets stop at dirt/bag tiles, so a monster
        # collinear with us but behind dirt is NOT actually fireable.
        # Without this check the old SmartHeuristic burned its 50-step
        # cooldown on shots that visibly hit a dirt wall.
        if self._fire_cd == 0:
            if self._line_of_sight_monster(state, dr, dc, self.prev_dir):
                return DiggerEnv.FIRE

        # ---- Move selection: lex-scored single pass --------------------
        # `tunnel_dist[r, c]` is shortest path from any monster to (r, c)
        # through *non-dirt* tiles. Diagonal monsters separated by dirt
        # walls register as unreachable, so the safety term no longer
        # over-penalises moves toward emeralds along the digger's own
        # tunnel. The v2 used Manhattan, which conflated reachable and
        # blocked threats and made the agent reroute pointlessly.
        tunnel_dist = self._compute_monster_tunnel_distance(state)
        bag_tiles = {(b.row, b.col) for b in state.bags}
        monster_tiles = {(m.row, m.col) for m in state.monsters}
        em_rows, em_cols = np.where(state.emeralds)
        em_targets = (list(zip(em_rows.tolist(), em_cols.tolist()))
                       if em_rows.size else [])

        def emerald_dist(r: int, c: int) -> int:
            if not em_targets:
                return MWIDTH + MHEIGHT
            return min(abs(er - r) + abs(ec - c) for er, ec in em_targets)

        best_score: tuple | None = None
        best_action = DiggerEnv.NOOP
        for action, ddr, ddc in self._MOVE_CANDIDATES:
            nr, nc = dr + ddr, dc + ddc
            if nr < 0 or nr >= MHEIGHT or nc < 0 or nc >= MWIDTH:
                continue
            if (nr, nc) in bag_tiles:
                continue
            if (nr, nc) in monster_tiles:
                # Stepping onto a monster tile == instant death.
                continue
            m_dist = int(tunnel_dist[nr, nc])
            e_dist = emerald_dist(nr, nc)
            # `safety` is capped at 2: only tiles where a monster could
            # collide next step (m_dist=1) get penalised. m_dist>=2 buckets
            # together and emerald-chase takes over. Unreachable monsters
            # produce m_dist=_MAX_TUNNEL_DIST which trivially clears the
            # cap.
            safety = min(m_dist, 2)
            sticky = (1 if action == self.prev_dir
                       and action != DiggerEnv.NOOP else 0)
            # `-is_noop` sits above `-e_dist` in the lex order so a NOOP
            # never wins on emerald-distance alone: standing still has
            # the same e_dist as the current tile, which trivially ties
            # any move that walks AWAY from an emerald (e.g. to dodge),
            # and the old ordering had NOOP win those ties.
            is_noop = 1 if action == DiggerEnv.NOOP else 0
            score = (safety, -is_noop, -e_dist, sticky)
            if best_score is None or score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _compute_monster_tunnel_distance(self, state) -> np.ndarray:
        """(MHEIGHT, MWIDTH) int array of shortest tunnel-distance from
        any monster to that tile. Unreachable tiles get _MAX_TUNNEL_DIST.

        Nobbins traverse only cleared (non-dirt) tiles, so this BFS skips
        any tile classified as dirt. A monster sitting in a different
        tunnel separated by an unbroken dirt wall registers as unreachable
        and stops contaminating the safety score.
        """
        H, W = state.dirt.shape
        dist = np.full((H, W), self._MAX_TUNNEL_DIST, dtype=np.int32)
        if not state.monsters:
            return dist
        queue: collections.deque[tuple[int, int]] = collections.deque()
        for m in state.monsters:
            if 0 <= m.row < H and 0 <= m.col < W:
                dist[m.row, m.col] = 0
                queue.append((m.row, m.col))
        # 4-connected BFS over non-dirt tiles.
        while queue:
            r, c = queue.popleft()
            d = dist[r, c]
            for dr_, dc_ in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr_, c + dc_
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if state.dirt[nr, nc]:
                    continue  # dirt blocks nobbin traversal
                if dist[nr, nc] > d + 1:
                    dist[nr, nc] = d + 1
                    queue.append((nr, nc))
        return dist

    def _line_of_sight_monster(self, state, dr: int, dc: int, facing: int):
        """Return the closest fireable monster, or None.

        "Fireable" means: collinear with the digger along `facing`, within
        `fire_range` tiles, with no intact bag in between (bullet stops
        at a bag).

        We deliberately do NOT check the dirt mask: dirt[r, c] is True
        whenever the tile's brown-pixel count exceeds a threshold, but
        tiles that have just been dug through still register as dirt-ish
        for a frame or two while the texture transitions. A bullet in a
        cleared tunnel would be flagged as blocked. Empirically, almost
        every monster the agent could see ended up behind a "dirt" tile
        by that test, and the FIRE action rate dropped to 0%.
        """
        if facing == DiggerEnv.NOOP:
            return None
        if facing == DiggerEnv.LEFT:
            line = ((dr, dc - i) for i in range(1, dc + 1))
        elif facing == DiggerEnv.RIGHT:
            line = ((dr, dc + i) for i in range(1, MWIDTH - dc))
        elif facing == DiggerEnv.UP:
            line = ((dr - i, dc) for i in range(1, dr + 1))
        else:  # DOWN
            line = ((dr + i, dc) for i in range(1, MHEIGHT - dr))
        monsters_by_tile = {(m.row, m.col): m for m in state.monsters}
        bag_tiles = {(b.row, b.col) for b in state.bags if not b.broken}
        for i, (r, c) in enumerate(line, start=1):
            if i > self.fire_range:
                return None
            if (r, c) in bag_tiles:
                return None  # bullet absorbed by bag
            if (r, c) in monsters_by_tile:
                return monsters_by_tile[(r, c)]
        return None


class DodgeMonsters:
    """Maximize time alive; ignore emeralds entirely.

    Each step, pick the move whose resulting tile keeps the minimum
    Manhattan distance to any monster as large as possible. Ties broken
    by (1) staying farther from walls (avoid corner traps), then (2)
    continuing in the previous direction to suppress one-step
    oscillations between two equal-distance moves.

    Bags are treated as obstacles (won't step onto a bag tile). Walls
    clamp the candidate set. FIRE is not in the candidate set: a 50-
    step cooldown plus a directional bullet makes it useless as the
    primary survival action -- we'd rather spend the step actually
    moving away.

    Designed as a DAGGER teacher whose label distribution covers every
    near-monster state (since chasing emeralds tends to walk INTO
    monsters, the greedy teacher's coverage of "monster nearby" is
    biased toward the actions that got the digger killed). Use this
    teacher to bias the BC dataset toward survival, then mix with
    GreedyEmerald to recover scoring behaviour.
    """

    # Movement candidates (excludes FIRE; see class docstring).
    _CANDIDATES: tuple[tuple[int, int, int], ...] = (
        (DiggerEnv.NOOP,  0,  0),
        (DiggerEnv.LEFT,  0, -1),
        (DiggerEnv.RIGHT, 0,  1),
        (DiggerEnv.UP,   -1,  0),
        (DiggerEnv.DOWN,  1,  0),
    )

    MOVE_ACTIONS = (DiggerEnv.LEFT, DiggerEnv.RIGHT,
                    DiggerEnv.UP, DiggerEnv.DOWN)

    def __init__(self):
        self.last_dir: int = DiggerEnv.NOOP

    def reset(self) -> None:
        self.last_dir = DiggerEnv.NOOP

    def __call__(self, state) -> int:
        if state.digger is None or not state.digger.present:
            return DiggerEnv.NOOP
        if not state.monsters:
            # No threat: sit still. Wandering invites trouble (e.g.
            # walking into a tile a bag is about to fall onto).
            return DiggerEnv.NOOP

        dr, dc = state.digger.row, state.digger.col
        bag_tiles = {(b.row, b.col) for b in state.bags}

        best_key: tuple[int, int, int] | None = None
        best_action = DiggerEnv.NOOP
        for action, ddr, ddc in self._CANDIDATES:
            nr, nc = dr + ddr, dc + ddc
            if nr < 0 or nr >= MHEIGHT or nc < 0 or nc >= MWIDTH:
                continue
            if (nr, nc) in bag_tiles:
                continue
            min_dist = min(abs(m.row - nr) + abs(m.col - nc)
                           for m in state.monsters)
            wall_dist = min(nr, MHEIGHT - 1 - nr, nc, MWIDTH - 1 - nc)
            sticky = 1 if action == self.last_dir \
                and action != DiggerEnv.NOOP else 0
            key = (min_dist, wall_dist, sticky)
            if best_key is None or key > best_key:
                best_key = key
                best_action = action

        if best_action in self.MOVE_ACTIONS:
            self.last_dir = best_action
        return best_action


# ---- Runners --------------------------------------------------------------

def run_headless(args) -> None:
    env = SymbolicDiggerEnv(max_steps=10**9,
                            episodic_life=not args.no_episodic_life)
    if args.dodge:
        policy = DodgeMonsters()
        pol_name = "dodge(survival)"
    elif args.smart:
        policy = SmartHeuristic(args.dodge_range, args.fire_range,
                                args.fire_cooldown)
        pol_name = (f"smart(dodge={args.dodge_range}, fire={args.fire_range}, "
                    f"cd={args.fire_cooldown})")
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
    if args.dodge:
        policy = DodgeMonsters()
    elif args.smart:
        policy = SmartHeuristic(args.dodge_range, args.fire_range,
                                args.fire_cooldown)
    else:
        policy = GreedyEmerald()

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
    p.add_argument("--dodge", action="store_true",
                   help="survival-only policy: ignore emeralds, "
                        "maximise distance from monsters. Mutually "
                        "exclusive with --smart.")
    p.add_argument("--dodge-range", type=int, default=2)
    p.add_argument("--fire-range", type=int, default=5)
    p.add_argument("--fire-cooldown", type=int,
                   default=SmartHeuristic.DEFAULT_FIRE_COOLDOWN_STEPS,
                   help="agent-steps to wait between FIREs "
                        "(~200 game frames / frame_skip)")
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
    if args.dodge and args.smart:
        raise SystemExit("--dodge and --smart are mutually exclusive")
    if args.live:
        run_live(args)
    else:
        run_headless(args)


if __name__ == "__main__":
    main()
