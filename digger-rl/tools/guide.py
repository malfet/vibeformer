"""Scripted guides for GRPO's guided rollouts.

A guide maps a RAM-decoded GameState to an action. It plans a shortest
path on the 15x10 tile grid (Dijkstra: open tunnel 1, dirt --dig-cost,
bags impassable, tiles at and next to a monster expensive) from the tile
the digger is heading into, and presses the first step's direction. A
perpendicular key held mid-tile keeps the digger moving and turns it at
the next boundary (digger.c updatedigger), so planning from the heading
tile is enough; no explicit alignment logic is needed.

    cherry   walk to the cherry; in bonus mode chase the nearest monster
    hunt     fire when a shot is lined up, otherwise walk toward the
             nearest monster, keeping a tile away from it while unloaded

These are not good players. They exist so that some rollouts in a group
reach an event (the cherry, a kill) the policy never reaches by itself,
which gives GRPO a return difference to learn from.
"""

from __future__ import annotations

import heapq

from digger_env import DiggerEnv
from tools.game_state import (DIR_DOWN, DIR_LEFT, DIR_RIGHT, DIR_UP, MHEIGHT,
                              MWIDTH, heading_tile)
from tools.play_audit import STEP, line_of_fire

MOVES = ((DIR_RIGHT, DiggerEnv.RIGHT), (DIR_LEFT, DiggerEnv.LEFT),
         (DIR_UP, DiggerEnv.UP), (DIR_DOWN, DiggerEnv.DOWN))
ACT_DIR = {act: d for d, act in MOVES}


def first_step(s, goals: set, dig_cost: float = 2.0,
               near_cost: float = 8.0, avoid: bool = True) -> int | None:
    """The move toward the cheapest goal tile, or None if none is
    reachable (or the digger is already on one)."""
    d = s.digger
    r0, c0, _ = heading_tile(d)
    if (r0, c0) in goals:
        return None
    bags = {(b.row, b.col) for b in s.bags if not b.broken}
    mons = {(m.row, m.col) for m in s.monsters}
    danger = set()
    if avoid:
        for r, c in mons:
            danger |= {(r, c), (r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)}
    dist = {(r0, c0): 0.0}
    first = {(r0, c0): None}
    pq = [(0.0, r0, c0)]
    while pq:
        g, r, c = heapq.heappop(pq)
        if g > dist[(r, c)]:
            continue
        if (r, c) in goals:
            return first[(r, c)]
        for dr_, act in MOVES:
            dr, dc = STEP[dr_]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < MHEIGHT and 0 <= nc < MWIDTH):
                continue
            if (nr, nc) in bags and (nr, nc) not in goals:
                continue
            w = dig_cost if s.dirt[nr, nc] else 1.0
            if (nr, nc) in danger and (nr, nc) not in goals:
                w += near_cost
            ng = g + w
            if ng < dist.get((nr, nc), 1e9):
                dist[(nr, nc)] = ng
                first[(nr, nc)] = act if first[(r, c)] is None else first[(r, c)]
                heapq.heappush(pq, (ng, nr, nc))
    return None


def cherry_guide(s) -> int:
    d = s.digger
    if d is None or s.dying:
        return DiggerEnv.NOOP
    if s.cherry is not None:
        a = first_step(s, {(s.cherry[1], s.cherry[0])})
        return a if a is not None else DiggerEnv.NOOP
    if s.bonus_mode and s.monsters:
        a = first_step(s, {(m.row, m.col) for m in s.monsters}, avoid=False)
        return a if a is not None else DiggerEnv.NOOP
    return hunt_guide(s)


def hunt_guide(s) -> int:
    d = s.digger
    if d is None or s.dying:
        return DiggerEnv.NOOP
    if not s.monsters:
        return DiggerEnv.NOOP
    if d.can_fire:
        if line_of_fire(s, d.dir)[0] == "aimed":
            return DiggerEnv.FIRE
        # turn toward a lined-up monster (turns happen at tile boundaries)
        for dr_, act in MOVES:
            kind, n = line_of_fire(s, dr_)
            if kind == "aimed" and n >= 2:
                return act
        # walk to a tile from which a monster is in a straight open line
        goals = set()
        for m in s.monsters:
            for dr_, _ in MOVES:
                dr, dc = STEP[dr_]
                r, c = m.row, m.col
                for k in range(1, 6):
                    r, c = r - dr, c - dc
                    if not (0 <= r < MHEIGHT and 0 <= c < MWIDTH) or s.dirt[r, c]:
                        break
                    if k >= 2:
                        goals.add((r, c))
        a = first_step(s, goals) if goals else None
        if a is not None:
            return a
    # unloaded or nothing to aim from: stay out of reach, drift toward
    # the nearest monster's neighbourhood at distance >= 3
    far = set()
    for r in range(MHEIGHT):
        for c in range(MWIDTH):
            k = min(abs(r - m.row) + abs(c - m.col) for m in s.monsters)
            if 3 <= k <= 4 and not s.dirt[r, c]:
                far.add((r, c))
    a = first_step(s, far) if far else None
    return a if a is not None else DiggerEnv.NOOP


GUIDES = {"cherry": cherry_guide, "hunt": hunt_guide}
