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
import heapq
import sys
import time
from pathlib import Path

# Allow running this file directly (`python tools/heuristic_agent.py`)
# by ensuring the project root is on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from digger_env import DiggerEnv
from tools.game_state import (DIR_DOWN, DIR_LEFT, DIR_NONE, DIR_RIGHT, DIR_UP,
                                MHEIGHT, MWIDTH, render_overlay, tile_center)
from tools.symbolic_env import SymbolicDiggerEnv

_DIR_TO_ACTION = {
    DIR_LEFT:  DiggerEnv.LEFT,
    DIR_RIGHT: DiggerEnv.RIGHT,
    DIR_UP:    DiggerEnv.UP,
    DIR_DOWN:  DiggerEnv.DOWN,
}
_ACTION_TO_FACING = {
    DiggerEnv.LEFT:  DIR_LEFT,
    DiggerEnv.RIGHT: DIR_RIGHT,
    DiggerEnv.UP:    DIR_UP,
    DiggerEnv.DOWN:  DIR_DOWN,
}

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
    """Emerald-chaser with monster dodging + opportunistic firing (v6.2).

    v6 changes (the "better teacher" pass, applying the snake-project
    lesson that teacher quality is the biggest lever downstream):
      - phantom-monster filter: detections on dirt tiles are CV
        artifacts and are ignored (no fake dodges / wasted FIREs);
      - under-bag transit rule: dirt below an intact bag may be dug
        through horizontally but never entered from below or lingered
        in (digging it releases the bag onto us);
      - BFS emerald routing: moves are scored against a multi-source
        Dijkstra distance field (dirt costs 2, bags block) instead of
        straight-line Manhattan distance, so the agent routes *around*
        bags and prefers existing tunnels.

    v6.2 adds the dynamic fall-column hazard model, grounded in the
    Digger Remastered source (bags.c): an unsupported bag wobbles for
    ~15 ticks then falls at 3 ticks/tile, and the digger only pauses
    the fuse while directly beneath it moving vertically. Every
    unsupported or falling bag projects a hazard column (contiguous
    not-fully-dirt tiles below it): never entered from outside, never
    FIREd from, never NOOPed in; when inside, sideways exits through
    open tunnel are preferred (`fast_exit`) and DOWN is forbidden --
    a falling bag is ~1.7x faster than the digger. Hazard avoidance
    leads the lex score, above monster distance. Also fixes the
    fire-at-nothing bug (FIRE when target_dir == facing == NOOP at
    episode start).

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
                 fire_cooldown_steps: int = DEFAULT_FIRE_COOLDOWN_STEPS,
                 phantom_filter: bool = True,
                 underbag: str = "transit",
                 routing: str = "dijkstra"):
        # `dodge_range` is retained for CLI backward-compat but no longer
        # used: the new lex-scored selection caps safety at distance 3 and
        # blends emerald-chase in continuously.
        #
        # The v6 knobs are independently toggleable for ablation:
        #   phantom_filter -- drop monster detections on dirt tiles.
        #     Caveat: hobbins DO dig through dirt, so this also blinds the
        #     dodge logic to a hobbin mid-dirt; level 1 is nobbin-dominated
        #     so the trade is usually worth it, but it's a knob.
        #   underbag -- "off": v5 behaviour (under-bag dirt is diggable);
        #     "strict": under-bag dirt is an obstacle (makes emeralds
        #     directly beneath bags unreachable -- measured 622 vs 1010,
        #     don't use); "transit": horizontal dig-through is allowed but
        #     UP-entry from below and lingering (NOOP) under a bag are
        #     forbidden, and routing pays a surcharge to prefer going
        #     around.
        #   routing -- "dijkstra": multi-source distance field with dirt
        #     cost 2 and bag obstacles; "manhattan": v5 straight-line.
        self.dodge_range = dodge_range
        self.fire_range = fire_range
        self.fire_cooldown_steps = fire_cooldown_steps
        assert underbag in ("off", "strict", "transit")
        assert routing in ("dijkstra", "manhattan")
        self.phantom_filter = phantom_filter
        self.underbag = underbag
        self.routing = routing
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
        # Prefer the CV-detected facing; fall back to the action-history
        # `prev_dir` when CV reports DIR_NONE (sprite mid-animation).
        cv_dir = getattr(state.digger, "dir", DIR_NONE)
        facing = (_DIR_TO_ACTION[cv_dir] if cv_dir in _DIR_TO_ACTION
                   else self.prev_dir)

        # Phantom filter: nobbins traverse only cleared tiles, so a
        # "monster" detected on a fully-dirt tile is a CV artifact (the
        # border-tile phantoms from the known-problems list). Dropping
        # them here matters twice over: a phantom triggers pointless
        # dodges, and FIREing at one wastes the ~50-step turret cooldown
        # while a real nobbin closes in.
        monsters = state.monsters
        if self.phantom_filter:
            monsters = [m for m in monsters
                        if not (0 <= m.row < MHEIGHT and 0 <= m.col < MWIDTH
                                and state.dirt[m.row, m.col])]

        # ---- Bag geometry: obstacles + dynamic fall-column hazard ------
        # Ground truth (bags.c, Digger Remastered source): a bag whose
        # support tile is no longer fully solid starts a ~15-tick wobble
        # fuse, then falls at 3 ticks/tile -- ~1.7x faster than the
        # digger walks, killing anything at-or-below it in the column.
        # The digger only pauses the fuse while directly beneath AND
        # moving vertically; standing still or passing horizontally does
        # not. CV can't see the wobble animation, but "support tile is
        # not dirt" is exactly the fuse-lit condition, so we derive the
        # hazard from the dirt grid: every unsupported or falling bag
        # projects a fall column (contiguous not-fully-dirt tiles below
        # it) that we must never enter, never linger in, and exit
        # sideways if we're inside (digging DOWN can't outrun a bag).
        bag_tiles = {(b.row, b.col) for b in state.bags}
        underbag_tiles: set[tuple[int, int]] = set()
        hazard_tiles: set[tuple[int, int]] = set()
        for b in state.bags:
            below = (b.row + 1, b.col)
            supported = (b.row + 1 >= MHEIGHT or state.dirt[below]
                         or below in bag_tiles)
            if not b.moving and supported:
                underbag_tiles.add(below)
                continue
            r = b.row + 1
            while r < MHEIGHT and not state.dirt[r, b.col] \
                    and (r, b.col) not in bag_tiles:
                hazard_tiles.add((r, b.col))
                r += 1
        if self.underbag == "strict":
            # Hard obstacle. Measured to cost ~390 mean score: emeralds
            # directly beneath bags become permanently unreachable.
            bag_tiles |= {t for t in underbag_tiles if state.dirt[t]}
        in_hazard = (dr, dc) in hazard_tiles

        # ---- FIRE / turn-to-fire opportunity ----------------------------
        # Scan every direction for a line-of-sight monster, not just the
        # one we're facing. If a fireable target is in *some* direction,
        # turn toward it: today we move toward it (and face it as a side
        # effect), then next step the facing matches and we FIRE.
        # Never while inside a fall column: a lit fuse outranks any
        # monster; keep moving.
        if self._fire_cd == 0 and not in_hazard:
            target_dir = self._best_fire_direction(state, dr, dc, monsters)
            # target_dir must be a real direction: at episode start both
            # target_dir and facing are NOOP, and a bare `==` makes the
            # agent FIRE at nothing, wasting the opening cooldown. (A
            # cross-session comparison once suggested the wasted shot
            # helped by suppressing early monster-hunting; a same-day
            # control run showed that was benchmark drift -- 20-ep means
            # move +/-400 between sessions on identical code. Same-
            # session A/B only.)
            if target_dir != DiggerEnv.NOOP and target_dir == facing:
                return DiggerEnv.FIRE
            if target_dir != DiggerEnv.NOOP:
                # Only turn-to-fire when (a) the move is legal and (b) the
                # post-turn tile is at least 2 tiles from the nearest
                # monster. Without (b) the agent walks directly into a
                # nearby monster trying to align its facing: at the right
                # edge with a monster closing along the same row, this
                # caused an oscillating LEFT-turn → FIRE → walk back RIGHT
                # → repeat loop while the next nobbin closed in, with the
                # digger never digging UP/DOWN to escape.
                ddr, ddc = _ACTION_DELTA[target_dir]
                if self._move_is_legal(state, dr, dc, target_dir, monsters) \
                        and self._turn_is_safe(dr, dc, target_dir, monsters) \
                        and (dr + ddr, dc + ddc) not in hazard_tiles:
                    return target_dir

        # ---- Move selection: lex-scored single pass --------------------
        # `tunnel_dist[r, c]` is shortest path from any monster to (r, c)
        # through *non-dirt* tiles. Diagonal monsters separated by dirt
        # walls register as unreachable, so the safety term no longer
        # over-penalises moves toward emeralds along the digger's own
        # tunnel. The v2 used Manhattan, which conflated reachable and
        # blocked threats and made the agent reroute pointlessly.
        tunnel_dist = self._compute_monster_tunnel_distance(state, monsters)
        monster_tiles = {(m.row, m.col) for m in monsters}
        # Distance-to-nearest-emerald over the real board replaces the
        # v5 straight-line Manhattan distance when routing="dijkstra":
        # bags and active fall columns block, dirt costs 2, and in
        # "transit" mode under-bag dirt pays a surcharge so routes
        # prefer going around a bag but can still dig through to an
        # emerald beneath it.
        surcharge = (underbag_tiles if self.underbag == "transit" else set())
        em_dist = (self._emerald_dist_field(state, bag_tiles | hazard_tiles,
                                            surcharge)
                   if self.routing == "dijkstra" else None)
        no_emeralds = not bool(state.emeralds.any())
        if em_dist is None:
            em_rows, em_cols = np.where(state.emeralds)
            em_targets = list(zip(em_rows.tolist(), em_cols.tolist()))

        def emerald_dist(r: int, c: int) -> int:
            if no_emeralds:
                return MWIDTH + MHEIGHT
            if em_dist is not None:
                return int(em_dist[r, c])
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
            tile_hazard = (nr, nc) in hazard_tiles
            if tile_hazard and not in_hazard:
                # Never step into an active fall column from outside.
                continue
            if in_hazard:
                # Already inside one (e.g. mid dig-through when the
                # support broke): keep moving and get out sideways.
                # DOWN stays inside the column and a falling bag is
                # ~1.7x faster than we are; NOOP burns fuse ticks.
                if action == DiggerEnv.NOOP:
                    continue
                if action == DiggerEnv.DOWN and tile_hazard:
                    continue
            if self.underbag == "transit":
                # Horizontal dig-through under a bag is allowed (the
                # wobble fuse gives us ~15 ticks; the hazard rules above
                # take over the moment the support actually breaks), but
                # digging UP into the under-bag tile drops the bag into
                # our own column, and standing still beneath one invites
                # a crush.
                if action == DiggerEnv.UP and (nr, nc) in underbag_tiles:
                    continue
                if action == DiggerEnv.NOOP and (dr, dc) in underbag_tiles:
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
            # `not_hazard` leads the lex order: leaving the fall column
            # outranks monster distance -- a bag drop is near-certain
            # death on its timer, a monster one tile away still has to
            # choose to walk into us. While inside a column, `fast_exit`
            # prefers open-tunnel exits over digging through dirt
            # (digging is ~1.7x slower than the bag falls).
            not_hazard = 0 if tile_hazard else 1
            fast_exit = (1 if in_hazard and not state.dirt[nr, nc] else 0)
            score = (not_hazard, fast_exit, safety, -is_noop, -e_dist,
                     sticky)
            if best_score is None or score > best_score:
                best_score = score
                best_action = action
        return best_action

    # Extra path cost for entering an under-bag tile in "transit" mode:
    # high enough to route around a bag when a detour exists, low enough
    # that an emerald directly beneath a bag is still worth digging to.
    _UNDERBAG_SURCHARGE: int = 6

    def _emerald_dist_field(self, state, blocked: set,
                             surcharge: set = frozenset()) -> np.ndarray:
        """(MHEIGHT, MWIDTH) int array: cheapest-path distance from each
        tile to the nearest emerald. Multi-source Dijkstra from all
        emerald tiles; `blocked` tiles (bags, falling-bag paths) are
        impassable; entering a dirt tile costs 2 (digging is slower than
        walking a tunnel), cleared tiles cost 1; `surcharge` tiles
        (under-bag) add _UNDERBAG_SURCHARGE. Unreachable tiles get a
        large sentinel so any reachable move dominates them.
        """
        H, W = state.dirt.shape
        dist = np.full((H, W), 10 ** 6, dtype=np.int32)
        er, ec = np.where(state.emeralds)
        if er.size == 0:
            return dist
        heap: list[tuple[int, int, int]] = []
        for r, c in zip(er.tolist(), ec.tolist()):
            dist[r, c] = 0
            heap.append((0, r, c))
        heapq.heapify(heap)
        while heap:
            d, r, c = heapq.heappop(heap)
            if d > dist[r, c]:
                continue
            for dr_, dc_ in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr_, c + dc_
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if (nr, nc) in blocked:
                    continue
                nd = d + (2 if state.dirt[nr, nc] else 1)
                if (nr, nc) in surcharge:
                    nd += self._UNDERBAG_SURCHARGE
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    heapq.heappush(heap, (nd, nr, nc))
        return dist

    def _compute_monster_tunnel_distance(self, state, monsters) -> np.ndarray:
        """(MHEIGHT, MWIDTH) int array of shortest tunnel-distance from
        any monster to that tile. Unreachable tiles get _MAX_TUNNEL_DIST.

        Nobbins traverse only cleared (non-dirt) tiles, so this BFS skips
        any tile classified as dirt. A monster sitting in a different
        tunnel separated by an unbroken dirt wall registers as unreachable
        and stops contaminating the safety score.
        """
        H, W = state.dirt.shape
        dist = np.full((H, W), self._MAX_TUNNEL_DIST, dtype=np.int32)
        if not monsters:
            return dist
        queue: collections.deque[tuple[int, int]] = collections.deque()
        for m in monsters:
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

    def _best_fire_direction(self, state, dr: int, dc: int,
                              monsters) -> int:
        """Return the action (LEFT/RIGHT/UP/DOWN) that points at the
        closest line-of-sight monster, or NOOP if none reachable.

        Used by the turn-to-fire logic: when there's a fireable monster
        not aligned with the current facing, we move toward it so next
        step the facing matches and we can FIRE.
        """
        best_dir = DiggerEnv.NOOP
        best_dist: int | float = float("inf")
        for action in (DiggerEnv.LEFT, DiggerEnv.RIGHT,
                        DiggerEnv.UP, DiggerEnv.DOWN):
            m = self._line_of_sight_monster(state, dr, dc, action, monsters)
            if m is None:
                continue
            d = abs(m.row - dr) + abs(m.col - dc)
            if d < best_dist:
                best_dist = d
                best_dir = action
        return best_dir

    def _move_is_legal(self, state, dr: int, dc: int, action: int,
                        monsters) -> bool:
        deltas = {DiggerEnv.LEFT: (0, -1), DiggerEnv.RIGHT: (0, 1),
                   DiggerEnv.UP: (-1, 0), DiggerEnv.DOWN: (1, 0)}
        ddr, ddc = deltas.get(action, (0, 0))
        nr, nc = dr + ddr, dc + ddc
        if not (0 <= nr < MHEIGHT and 0 <= nc < MWIDTH):
            return False
        if any(b.row == nr and b.col == nc for b in state.bags):
            return False
        if any(m.row == nr and m.col == nc for m in monsters):
            return False
        return True

    def _turn_is_safe(self, dr: int, dc: int, action: int,
                       monsters) -> bool:
        """True if the tile we'd step onto for turn-to-fire is at least
        2 Manhattan tiles from every monster. Prevents walking directly
        into a nobbin while trying to face it.
        """
        deltas = {DiggerEnv.LEFT: (0, -1), DiggerEnv.RIGHT: (0, 1),
                   DiggerEnv.UP: (-1, 0), DiggerEnv.DOWN: (1, 0)}
        ddr, ddc = deltas.get(action, (0, 0))
        nr, nc = dr + ddr, dc + ddc
        for m in monsters:
            if abs(m.row - nr) + abs(m.col - nc) < 2:
                return False
        return True

    def _line_of_sight_monster(self, state, dr: int, dc: int, facing: int,
                                monsters):
        """Return the closest fireable monster, or None.

        "Fireable" means: collinear with the digger along `facing`, within
        `fire_range` tiles, with no intact bag and no unbroken dirt wall
        in between.

        Dirt check: a previous version of this code ignored `state.dirt`
        because freshly-dug tiles briefly register as dirt-ish, but
        skipping the check entirely caused FIREs through real walls
        ("fires in the wall" from a parallel tunnel). The current
        compromise is to tolerate dirt on the tile *directly in front of
        the digger* only (i==1) -- this is the tile most likely to be
        mid-transition after the digger just dug into it -- and treat
        any dirt further along the line as a real wall that stops the
        bullet.
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
        monsters_by_tile = {(m.row, m.col): m for m in monsters}
        bag_tiles = {(b.row, b.col) for b in state.bags if not b.broken}
        for i, (r, c) in enumerate(line, start=1):
            if i > self.fire_range:
                return None
            if (r, c) in bag_tiles:
                return None  # bullet absorbed by bag
            if i >= 2 and state.dirt[r, c]:
                return None  # real dirt wall stops the bullet
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


# ---- Death / bag-timing instrumentation -----------------------------------

class DeathBagLogger:
    """JSONL event logger for headless runs (--death-log PATH).

    Two jobs:
      1. Death forensics: on every life loss, dump the pre-death state
         (digger tile, bags with support/moving flags, nearest-monster
         distance) plus a classification: was the digger inside some
         bag's *fall column* (contiguous not-fully-dirt tiles below an
         unsupported or moving bag)?
      2. Passive timing calibration: per bag, log the step when its
         support tile stops being dirt ("unsupported"), the step its
         sprite first goes vertically off-centre ("bag_moving"), and
         when it stops ("bag_landed"). The unsupported->moving gap
         measures the game's 15-tick wobble fuse in agent-steps; the
         landing row delta measures fall speed. (Ground truth: bags.c
         in the Digger Remastered source -- wt=15, 6px/tick, breaks
         into gold on fallh>1.)
    """

    def __init__(self, path: str):
        import json
        self._json = json
        self._f = open(path, "a")
        self.deaths = 0
        # col -> (row_when_support_lost, step)
        self._unsupported: dict[int, tuple[int, int]] = {}
        # col -> (row_first_moving, step)
        self._moving: dict[int, tuple[int, int]] = {}
        self.start_episode(0)

    def _emit(self, obj: dict) -> None:
        self._f.write(self._json.dumps(obj) + "\n")
        self._f.flush()

    def start_episode(self, ep: int) -> None:
        self.ep = ep
        self.step = 0
        self.prev_lives = None
        self._unsupported.clear()
        self._moving.clear()
        self._last_digger: tuple[int, int, int] | None = None
        # (step, "unsupported"|"moving"|"landed", col, row) ring of
        # recent bag activity for post-hoc death attribution.
        self._recent: collections.deque = collections.deque(maxlen=40)

    @staticmethod
    def _bag_supported(state, b) -> bool:
        if b.row + 1 >= MHEIGHT:
            return True  # resting on the floor
        return bool(state.dirt[b.row + 1, b.col])

    @staticmethod
    def _fall_columns(state) -> set[tuple[int, int]]:
        """Tiles inside the fall column of any unsupported or moving
        bag: contiguous run of not-fully-dirt tiles directly below it.
        """
        out: set[tuple[int, int]] = set()
        for b in state.bags:
            if not b.moving and DeathBagLogger._bag_supported(state, b):
                continue
            r = b.row + 1
            while r < MHEIGHT and not state.dirt[r, b.col]:
                out.add((r, b.col))
                r += 1
        return out

    def on_frame(self, state) -> None:
        """Call once per emulator frame (the env extracts a fresh state
        every frame anyway). Bag falls are FAST -- a 1-tile drop takes
        ~3 game ticks, less than one agent step at frame_skip=4 -- so
        per-step sampling misses short falls entirely; per-frame
        tracking is the only way to attribute those deaths.
        """
        self._track_bags(state)

    def _track_bags(self, state) -> None:
        for b in state.bags:
            col = int(b.col)
            if b.moving:
                if col not in self._moving:
                    self._moving[col] = (int(b.row), self.step)
                    ev = {"ev": "bag_moving", "ep": self.ep,
                          "step": self.step, "col": col, "row": int(b.row)}
                    if col in self._unsupported:
                        r0, s0 = self._unsupported[col]
                        ev["fuse_steps"] = self.step - s0
                        ev["row_at_support_loss"] = r0
                    self._recent.append((self.step, "moving", col, int(b.row)))
                    self._emit(ev)
            else:
                if col in self._moving:
                    r0, s0 = self._moving.pop(col)
                    self._recent.append((self.step, "landed", col, int(b.row)))
                    self._emit({"ev": "bag_landed", "ep": self.ep,
                                "step": self.step, "col": col,
                                "row_from": r0, "row_to": int(b.row),
                                "fall_steps": self.step - s0})
                    self._unsupported.pop(col, None)
                if not self._bag_supported(state, b):
                    if col not in self._unsupported:
                        self._unsupported[col] = (int(b.row), self.step)
                        self._recent.append(
                            (self.step, "unsupported", col, int(b.row)))
                        self._emit({"ev": "unsupported", "ep": self.ep,
                                    "step": self.step, "col": col,
                                    "row": int(b.row)})
                else:
                    self._unsupported.pop(col, None)

    def on_step(self, state, info: dict) -> None:
        """Call once per agent step with the state the policy just saw
        and the info dict returned by the env for that step."""
        self.step += 1
        self._track_bags(state)
        if state.digger is not None:
            self._last_digger = (int(state.digger.row),
                                 int(state.digger.col), self.step)
        # `state` is what the policy saw when it chose the action that
        # got it killed this step -- the most useful forensic snapshot.
        lives = info.get("lives")
        if lives is not None:
            if self.prev_lives is not None and lives < self.prev_lives:
                self._log_death(state, info)
            self.prev_lives = lives

    def _log_death(self, state, info: dict) -> None:
        self.deaths += 1
        rec: dict = {"ev": "death", "ep": self.ep, "step": self.step,
                     "score": info.get("score", 0)}
        if state.digger is not None:
            dr, dc = int(state.digger.row), int(state.digger.col)
            rec["digger"] = [dr, dc]
            cols = self._fall_columns(state)
            rec["in_fall_column"] = (dr, dc) in cols
            rec["moving_bag_above"] = any(
                b.moving and b.col == dc and b.row < dr for b in state.bags)
            if state.monsters:
                rec["monster_min_dist"] = min(
                    abs(m.row - dr) + abs(m.col - dc) for m in state.monsters)
        if state.digger is None and self._last_digger is not None:
            # CV loses the digger when a bag/gold sprite lands on top of
            # it -- the signature of a crush. Fall back to the last seen
            # tile for attribution.
            lr, lc, ls = self._last_digger
            rec["last_digger"] = [lr, lc]
            rec["last_digger_age"] = self.step - ls
            rec["last_in_fall_column"] = (lr, lc) in self._fall_columns(state)
        ref = rec.get("digger") or rec.get("last_digger")
        if ref is not None:
            rec["recent_bag_activity"] = [
                {"step": s, "ev": ev, "col": c, "row": r}
                for (s, ev, c, r) in self._recent
                if self.step - s <= 15 and abs(c - ref[1]) <= 1]
        rec["bags"] = [{"r": int(b.row), "c": int(b.col),
                        "moving": bool(b.moving),
                        "supported": self._bag_supported(state, b)}
                       for b in state.bags]
        self._emit(rec)


# ---- Runners --------------------------------------------------------------

def run_headless(args) -> None:
    env = SymbolicDiggerEnv(max_steps=10**9,
                            episodic_life=not args.no_episodic_life)
    if args.dodge:
        policy = DodgeMonsters()
        pol_name = "dodge(survival)"
    elif args.smart:
        policy = SmartHeuristic(args.dodge_range, args.fire_range,
                                args.fire_cooldown,
                                phantom_filter=not args.no_phantom_filter,
                                underbag=args.underbag,
                                routing=args.routing)
        pol_name = (f"smart(dodge={args.dodge_range}, fire={args.fire_range}, "
                    f"cd={args.fire_cooldown}, "
                    f"phantom={not args.no_phantom_filter}, "
                    f"underbag={args.underbag}, routing={args.routing})")
    else:
        policy = GreedyEmerald()
        pol_name = "greedy(anti-jitter)"

    scores: list[int] = []
    lengths: list[int] = []
    dlog = DeathBagLogger(args.death_log) if args.death_log else None
    t0 = time.monotonic()

    for ep in range(args.episodes):
        env.reset()
        policy.reset()
        if dlog is not None:
            dlog.start_episode(ep)
        ep_score = 0
        ep_len = 0
        for step in range(args.max_steps):
            state = env._last_state
            action = policy(state)
            done = False
            for _ in range(args.frame_skip):
                obs, r, done_, info = env.step(action)
                if dlog is not None:
                    dlog.on_frame(env._last_state)
                if done_:
                    done = True
                    break
            if dlog is not None:
                dlog.on_step(state, info)
            ep_score = info.get("score", 0)
            ep_len = step + 1
            if done:
                break
        scores.append(ep_score)
        lengths.append(ep_len)
        print(f"  ep {ep + 1}/{args.episodes}: score={ep_score}  steps={ep_len}",
              flush=True)
    if dlog is not None:
        print(f"  death-log: {dlog.deaths} deaths logged to {args.death_log}")

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
                                args.fire_cooldown,
                                phantom_filter=not args.no_phantom_filter,
                                underbag=args.underbag,
                                routing=args.routing)
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
    p.add_argument("--no-phantom-filter", action="store_true",
                   help="keep monster detections on dirt tiles (v5 behaviour)")
    p.add_argument("--underbag", choices=("off", "strict", "transit"),
                   default="transit",
                   help="handling of dirt directly below a bag: off = v5 "
                        "(freely diggable), strict = hard obstacle, "
                        "transit = horizontal dig-through allowed, UP-entry "
                        "and lingering forbidden (default)")
    p.add_argument("--routing", choices=("manhattan", "dijkstra"),
                   default="dijkstra",
                   help="emerald distance metric for move scoring")
    p.add_argument("--death-log", type=str, default="",
                   help="append JSONL death-forensics + bag-timing events "
                        "to this file (headless mode only)")
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
