"""Bag-kill start states: moments shortly before a monster is crushed.

A policy plays (from tools/level_states.py starts, or level 1) while we
keep a save state for each of the last steps. When a falling bag
squashes a monster (monster.c squashmonster sets its `death` to 1;
nothing else does), each state `--back` steps earlier is probed: hold
NOOP from there for as many steps. If the monster is crushed anyway,
the kill did not depend on the digger and the state is skipped
("passive"); otherwise it is saved as a start state, where the right
play earns the kill and the wrong one does not. Kills within
max(--back) steps of the previous one (often the same bag) are counted
but not saved.

The probes restore save states mid-game; the played game is then
restored to where it was, which replays exactly with the lockstep core
(README Lesson 21), so the policy's frame stack stays valid.

    python -m tools.bag_kills data/checkpoints/px128_ppo3/ppo_digger_final.pt \\
        --out-dir data/save/bag_kills --kills 150 --seed 0

Use the directory as a train_pixel.py --start-states pool.
"""

from __future__ import annotations

import argparse
import collections
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from digger_env import DiggerEnv
from tools.game_state import N_MONSTERS, MONSTER_WORDS, RAM_MONSTERS
from tools.policies import env_kwargs, make_policy
from tools.symbolic_env import SymbolicDiggerEnv, env_step_skipped


def squashed(env: SymbolicDiggerEnv) -> int:
    """Monsters on screen currently crushed by a bag (death 1, or 4
    while their death animation ends)."""
    mem = env._env._core.read_memory_region(0)
    m = np.frombuffer(mem, dtype="<i2", count=N_MONSTERS * MONSTER_WORDS,
                      offset=RAM_MONSTERS).reshape(N_MONSTERS, MONSTER_WORDS)
    return int(((m[:, 15] != 0) & np.isin(m[:, 10], (1, 4))).sum())


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("policy", help="teacher | checkpoint")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--kills", type=int, default=150,
                   help="stop after this many bag kills")
    p.add_argument("--back", type=int, nargs="+", default=[6, 12, 20],
                   help="steps before the kill to save a state from")
    p.add_argument("--max-steps", type=int, default=200_000,
                   help="stop after this many policy steps regardless")
    p.add_argument("--start-states", default="data/save/levels")
    p.add_argument("--start-prob", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    env = SymbolicDiggerEnv(max_steps=10**9, clip_reward=False,
                            episodic_life=False,
                            start_states=args.start_states,
                            start_prob=args.start_prob,
                            **env_kwargs(args.policy))
    policy = make_policy(args.policy, env, torch.device("cpu"),
                         seed=args.seed)
    keep = max(args.back)

    def noop_kills(state, steps: int) -> bool:
        env.load_state(state)
        n0 = squashed(env)
        for _ in range(steps):
            _, _, done, info = env_step_skipped(env, DiggerEnv.NOOP, 4)
            if squashed(env) > n0:
                return True
            if done or info.get("dying", False):
                break
        return False

    stats = collections.Counter()
    t0 = time.monotonic()
    env.reset()
    policy.reset()
    obs = env.current_obs()
    history = collections.deque(maxlen=keep)
    prev = squashed(env)
    steps = 0
    last_kill = -keep
    while stats["kills"] < args.kills and steps < args.max_steps:
        history.append((env.save_state(), env._last_state.lives))
        a = policy(obs)
        obs, _, done, info = env_step_skipped(env, a, 4)
        steps += 1
        policy.observe(info)
        if done:
            env.reset()
            policy.reset()
            obs = env.current_obs()
            history.clear()
            prev = squashed(env)
            continue
        cur = squashed(env)
        if cur > prev and steps - last_kill < keep:
            stats["kills_skipped"] += 1     # same bag / near the last one
            last_kill = steps
        elif cur > prev:
            last_kill = steps
            stats["kills"] += 1
            level = env._env.read_level()
            stats[f"kills_L{level}"] += 1
            now = env.save_state()
            for back in args.back:
                if back > len(history):
                    continue
                state, lives = history[len(history) - back]
                if noop_kills(state, back):
                    stats["passive"] += 1
                    continue
                name = f"L{level}_bk{args.seed}_{stats['kills']}_b{back}.pkl"
                with open(args.out_dir / name, "wb") as fh:
                    pickle.dump(dict(state=state, level=level, step=-back,
                                     lives=lives), fh)
                stats["saved"] += 1
            obs = env.load_state(now)
            print(f"kill {stats['kills']} at step {steps} L{level}: "
                  f"saved {stats['saved']}, passive {stats['passive']} "
                  f"({time.monotonic() - t0:.0f}s)", flush=True)
        prev = cur
    env.close()
    print(f"\n{steps} steps, {dict(stats)}")


if __name__ == "__main__":
    main()
