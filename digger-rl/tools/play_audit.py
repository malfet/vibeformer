"""Audit a pixel policy's deaths and shots.

Plays full games (sampled, or greedy with --argmax) and keeps an
emulator snapshot for each of the last steps. At each death it rolls
back `--back` steps and asks, with the same rollouts as
tools/policy_search.py (hold one action for 3 steps, then the greedy
policy for the rest of 30), whether

  - the greedy policy would have lived (a death from sampling), or
  - some other action would have lived (avoidable), or
  - nothing lives from there (trapped that early).

It also reports how close the nearest monster was, whether the policy
was walking toward it, and whether a loaded shot at a monster in line
was passed up in the steps before the death.

Every shot is classified from RAM when it leaves the digger: a monster
in line through dug tunnel (aimed), dirt right in front (into dirt), or
an open tunnel with nothing in line (blind); a shot kills if a monster
vanishes while the fireball is out without being under a bag. FIRE
presses while reloading do nothing and are counted apart.

    python -m tools.play_audit data/checkpoints/RUN/ppo_digger_final.pt \\
        --games 10 --out logs/audit_RUN.json
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np
import torch

from digger_env import DiggerEnv
from tools.game_state import (DIR_DOWN, DIR_LEFT, DIR_RIGHT, DIR_UP,
                              MONSTER_WORDS, N_MONSTERS, RAM_DIGGER,
                              RAM_MONSTERS)
from tools.heuristic_agent import MHEIGHT, MWIDTH
from tools.policies import make_policy
from tools.policy_search import PolicySearch
from tools.symbolic_env import SymbolicDiggerEnv, env_step_skipped

STEP = {DIR_RIGHT: (0, 1), DIR_LEFT: (0, -1), DIR_UP: (-1, 0),
        DIR_DOWN: (1, 0)}
MOVE_DIR = {DiggerEnv.LEFT: DIR_LEFT, DiggerEnv.RIGHT: DIR_RIGHT,
            DiggerEnv.UP: DIR_UP, DiggerEnv.DOWN: DIR_DOWN}


def ram(env) -> tuple[np.ndarray, np.ndarray]:
    """(digger words, monster words) straight from RAM."""
    mem = env._env._core.read_memory_region(0)
    d = np.frombuffer(mem, "<i2", 25, RAM_DIGGER).copy()
    m = np.frombuffer(mem, "<i2", N_MONSTERS * MONSTER_WORDS,
                      RAM_MONSTERS).reshape(N_MONSTERS, MONSTER_WORDS).copy()
    return d, m


def line_of_fire(state, direction: int) -> tuple[str, int]:
    """What a shot fired now along `direction` meets: ("aimed", tiles to
    the monster), ("dirt", 1) or ("blind", tiles of open tunnel)."""
    d = state.digger
    if direction not in STEP:
        return "blind", 0
    dr, dc = STEP[direction]
    mons = {(m.row, m.col) for m in state.monsters}
    bags = {(b.row, b.col) for b in state.bags if not b.broken}
    r, c, i = d.row, d.col, 0
    while True:
        r, c, i = r + dr, c + dc, i + 1
        if not (0 <= r < MHEIGHT and 0 <= c < MWIDTH) or (r, c) in bags:
            return "blind", i - 1
        if (r, c) in mons:
            return "aimed", i
        if state.dirt[r, c]:
            return ("dirt", 1) if i == 1 else ("blind", i - 1)


def nearest(state) -> int:
    d = state.digger
    if d is None or not state.monsters:
        return 99
    return min(abs(m.row - d.row) + abs(m.col - d.col)
               for m in state.monsters)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("policy", help="pixel checkpoint")
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--argmax", action="store_true")
    p.add_argument("--back", type=int, nargs="+", default=[4, 8, 12],
                   help="steps before the death onset to roll back to")
    p.add_argument("--start-states", default=None,
                   help="dir of level starts (default: game start)")
    p.add_argument("--start-prob", type=float, default=0.0)
    p.add_argument("--out", type=Path, default=None, help="per-event JSON")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    env = SymbolicDiggerEnv(max_steps=10**9, clip_reward=False,
                            episodic_life=False,
                            start_states=args.start_states,
                            start_prob=args.start_prob)
    policy = make_policy(args.policy, env, torch.device("cpu"))
    search = PolicySearch(env, policy)
    keep = max(args.back)
    deaths, shots, misses = [], [], collections.Counter()
    duds = 0
    t0 = time.monotonic()
    for game in range(args.games):
        env.reset()
        fresh = True
        hist: collections.deque = collections.deque(maxlen=keep)
        shot = None
        score = 0
        while True:
            s = env._last_state
            logits = search.observe(fresh)
            probs = torch.softmax(logits, -1).numpy()
            greedy = int(logits.argmax())
            a = (greedy if args.argmax else
                 int(torch.distributions.Categorical(logits=logits).sample()))
            d0, m0 = ram(env)
            alive = s.digger is not None and not s.dying
            if alive:
                hist.append(dict(snap=env.snapshot(),
                                 stack=list(policy.stack), a=a,
                                 greedy=greedy, p=float(probs[a]),
                                 dist=nearest(s),
                                 ready=s.digger.can_fire,
                                 los=line_of_fire(s, s.digger.dir)[0]))
                if s.digger.can_fire and hist[-1]["los"] == "aimed":
                    misses["aimed_ready_steps"] += 1
                    misses["aimed_ready_fired"] += a == DiggerEnv.FIRE
            _, _, done, info = env_step_skipped(env, a, 4)
            score = int(info.get("score", score))
            d1, m1 = ram(env)
            if a == DiggerEnv.FIRE and alive:
                if d0[9] == 0 and d1[9] > 0:
                    kind, dist = line_of_fire(s, s.digger.dir)
                    shot = dict(kind=kind, dist=dist, kill=False,
                                near=nearest(s), level=s.level)
                    shots.append(shot)
                elif d0[9] > 0 or d0[23] == 0:
                    duds += 1
            if shot is not None:
                # killmon clears flag; a bag kill sets death first
                gone = (m0[:, 15] != 0) & (m1[:, 15] == 0) & (m0[:, 10] == 0)
                shot["kill"] |= bool(gone.any())
                if d1[23] != 0:          # notfiring again: the shot is over
                    shot = None
            if info.get("death_event", False) and len(hist) >= 2:
                snap, stack = env.snapshot(), list(policy.stack)
                ev = dict(game=game, level=s.level,
                          cause=info.get("death_cause") or "monster",
                          dist=[h["dist"] for h in hist],
                          off_greedy=sum(h["a"] != h["greedy"] for h in hist),
                          min_p=min(h["p"] for h in hist),
                          passed_shot=any(
                              h["ready"] and h["los"] == "aimed"
                              and h["a"] != DiggerEnv.FIRE for h in hist))
                for k in args.back:
                    if k > len(hist):
                        continue
                    h = hist[-k]
                    res = {}
                    for act in range(DiggerEnv.NUM_ACTIONS):
                        policy.stack.clear()
                        policy.stack.extend(h["stack"])
                        res[act] = search._rollout(h["snap"], act)[0]
                    ev[f"greedy_lives_{k}"] = res[h["greedy"]]
                    ev[f"any_lives_{k}"] = any(res.values())
                env.restore_snapshot(snap)
                policy.stack.clear()
                policy.stack.extend(stack)
                deaths.append(ev)
                hist.clear()
            fresh = info.get("dying", False)
            if done:
                break
        print(f"  game {game + 1}/{args.games}: score {score}, deaths "
              f"{sum(e['game'] == game for e in deaths)}, shots "
              f"{len(shots)} ({time.monotonic() - t0:.0f}s)", flush=True)
    env.close()

    n = len(deaths)
    print(f"\n=== {args.policy} ({args.games} games, "
          f"{'argmax' if args.argmax else 'sampled'}) ===")
    print(f"deaths: {n} ({dict(collections.Counter(e['cause'] for e in deaths))})")
    for k in args.back:
        have = [e for e in deaths if f"any_lives_{k}" in e]
        if not have:
            continue
        g = sum(e[f"greedy_lives_{k}"] for e in have)
        anyl = sum(e[f"any_lives_{k}"] for e in have)
        print(f"  {k:2d} steps back: greedy lives {g}/{len(have)}, "
              f"some action lives {anyl}/{len(have)}, "
              f"trapped {len(have) - anyl}/{len(have)}")
    if n:
        d4 = [e["dist"][-min(4, len(e["dist"]))] for e in deaths]
        print(f"  nearest monster 4 steps before: "
              f"{dict(sorted(collections.Counter(d4).items()))}")
        print(f"  off-greedy actions in the last {keep} steps: "
              f"{sum(e['off_greedy'] > 0 for e in deaths)}/{n} deaths; "
              f"min action prob < 0.2 in {sum(e['min_p'] < 0.2 for e in deaths)}/{n}")
        print(f"  a loaded aimed shot was passed up: "
              f"{sum(e['passed_shot'] for e in deaths)}/{n}")
    kinds = collections.Counter(s["kind"] for s in shots)
    kills = collections.Counter(s["kind"] for s in shots if s["kill"])
    print(f"shots: {len(shots)} ({len(shots) / args.games:.1f}/game), "
          f"kills {sum(kills.values())}; FIRE while reloading {duds}")
    for k in ("aimed", "blind", "dirt"):
        print(f"  {k:6s} {kinds[k]:4d}  kills {kills[k]:3d}")
    print(f"loaded with a monster in line: fired on "
          f"{misses['aimed_ready_fired']}/{misses['aimed_ready_steps']} steps")
    if args.out:
        args.out.write_text(json.dumps(dict(deaths=deaths, shots=shots,
                                            duds=duds, misses=misses),
                                       indent=1))


if __name__ == "__main__":
    main()
