"""Save states at the start of (and inside) levels 1..N, for training and
evaluating on later levels.

Our policies rarely get past level 3, and later levels look different (a
new background colour/pattern per level plan, more and faster monsters,
shorter fire recharge), so a pixel student trained from reset never sees
them. This builds them directly:

  1. A policy plays level 1 while we keep a rolling buffer of states.
  2. When level 1 ends, take the state from `--lead` steps before the end,
     poke the level counter to N-1 (DiggerEnv.poke_level), and let the
     policy finish the level; the game then builds level N itself, the
     same as if it had been reached by play.
  3. From each level start, the policy plays on and a state is saved every
     `--every` steps (mid-level states, with tunnels already dug).

    python -m tools.level_states --levels 2-8 --per-level 4 \
        --policy data/checkpoints/ppo_ft1/ppo_sym_final.pt

Writes data/save/levels/L{level}_{k}_{step}.pkl, each a dict:
{"state": SymbolicDiggerEnv.save_state(), "level", "step" (since level
start), "lives", "score"}. Load with SymbolicDiggerEnv.load_state(d["state"]).
"""

from __future__ import annotations

import argparse
import collections
import pickle
from pathlib import Path

import numpy as np
import torch

from digger_env import SAVE_DIR
from tools.policies import env_kwargs, make_policy
from tools.symbolic_env import SymbolicDiggerEnv, env_step_skipped

OUT_DIR = SAVE_DIR / "levels"
SPAWN_XY = (152, 180)                   # digger x, y at a level start


def parse_levels(s: str) -> list[int]:
    lo, _, hi = s.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def play_until_level_change(env, policy, max_steps: int, frame_skip: int,
                            buffer=None, allow_death: bool = False):
    """Play until the level counter changes and the new level is drawn.
    Returns the step count, or None on game over, timeout, or (unless
    allow_death) a death.
    `buffer` (a deque) collects a state every 5 steps on the way."""
    start = env._env.read_level()
    obs = env.current_obs()
    for t in range(max_steps):
        if buffer is not None and t % 5 == 0:
            buffer.append(env.save_state())
        # The counter changes before the new level is drawn (old tunnels
        # and monster records linger for a few steps). Hold still from
        # then on, so the digger is still at its spawn point when the new
        # level is ready: no monsters out yet.
        changed = env._env.read_level() != start
        a = 0 if changed else policy(obs)
        obs, _, done, info = env_step_skipped(env, a, frame_skip)
        if not changed:
            policy.observe(info)
        if done or (info.get("dying", False) and not allow_death):
            return None
        s = env._last_state
        if changed and not s.monsters and (s.digger.x, s.digger.y) == SPAWN_XY:
            return t + 1
    return None


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--policy", default="teacher",
                   help="teacher | checkpoint (tools/policies.py spec)")
    p.add_argument("--levels", default="1-8", help="e.g. 2-8")
    p.add_argument("--per-level", type=int, default=4,
                   help="level starts to build per level")
    p.add_argument("--every", type=int, default=100,
                   help="also save a state every N steps into the level")
    p.add_argument("--mid-max", type=int, default=3,
                   help="mid-level states per level start")
    p.add_argument("--lead", type=int, default=40,
                   help="poke this many steps before level 1 ends")
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=OUT_DIR)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    env = SymbolicDiggerEnv(max_steps=10**9, clip_reward=False,
                            episodic_life=False, **env_kwargs(args.policy))
    policy = make_policy(args.policy, env, torch.device("cpu"),
                         seed=args.seed)

    # Level-1 endings to poke from: states `lead` steps before the end.
    endings: list[dict] = []
    while len(endings) < 3:
        env.reset()
        policy.reset()
        buf = collections.deque(maxlen=args.lead // 5 + 1)
        if play_until_level_change(env, policy, 3000, args.frame_skip,
                                   buf) is not None:
            endings.append(buf[0])
            print(f"level-1 ending {len(endings)} recorded", flush=True)

    rng = np.random.default_rng(args.seed)
    saved = 0
    for level in parse_levels(args.levels):
        k = tries = 0
        while k < args.per_level and tries < 10 * args.per_level:
            tries += 1
            env.reset()
            policy.reset()
            if level > 1:
                env.load_state(endings[rng.integers(len(endings))])
                env._env.poke_level(level - 1)
                # Deaths are fine here: the level still ends when the
                # last emerald goes, just with fewer lives.
                if play_until_level_change(env, policy, 2000,
                                           args.frame_skip,
                                           allow_death=True) is None:
                    continue
            assert env._env.read_level() == level
            # Level start, then mid-level states while the policy survives.
            obs = env.current_obs()
            for t in range(args.every * args.mid_max + 1):
                if t % args.every == 0:
                    d = dict(state=env.save_state(), level=level, step=t,
                             lives=env._last_state.lives,
                             score=env._env._read_score())
                    with open(args.out / f"L{level}_{k}_{t}.pkl", "wb") as f:
                        pickle.dump(d, f)
                    saved += 1
                a = policy(obs)
                obs, _, done, info = env_step_skipped(env, a,
                                                      args.frame_skip)
                policy.observe(info)
                if done or info.get("dying", False) \
                        or env._env.read_level() != level:
                    break
            print(f"level {level}: start {k} saved "
                  f"(try {tries}, survived {t} steps)", flush=True)
            k += 1
    env.close()
    print(f"{saved} states in {args.out}")


if __name__ == "__main__":
    main()
