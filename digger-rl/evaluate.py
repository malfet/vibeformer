"""Evaluate any policy over N full games.

    python evaluate.py teacher --episodes 20          # same-session baseline
    python evaluate.py data/checkpoints/RUN/ppo_sym_final.pt --episodes 20
    python evaluate.py data/checkpoints/RUN/ppo_digger_final.pt   # pixel ckpt
    python evaluate.py random | search
    python evaluate.py teacher --levels data/save/levels   # per level

The policy is a spec from tools/policies.py: teacher, search, random, or a
checkpoint from train_symbolic.py or train_pixel.py. Reports:

  - **game score**: total score across a full 3-life game.
  - **life length**: agent steps between deaths, segmented at death onset
    (RAM death stage), excluding the ~100 input-ignored dying steps.
  - **level 1**: games that cleared level 1, and that did so without
    losing a life (the project goal).

With --levels DIR (states from tools/level_states.py) it instead plays one
life from every saved level start and reports, per level: how often the
level is cleared before the first death, the share of its emeralds eaten,
and the score gained.

Cross-session numbers drift; compare against a same-session teacher run.
"""

from __future__ import annotations

import argparse
import collections
import pickle
import re
import statistics
from pathlib import Path

import torch

from tools.policies import env_kwargs, make_policy
from tools.symbolic_env import SymbolicDiggerEnv, env_step_skipped
from train_pixel import select_device


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("policy", help="teacher | search | random | checkpoint path")
    p.add_argument("--episodes", type=int, default=20,
                   help="full 3-life games")
    p.add_argument("--argmax", action="store_true",
                   help="argmax instead of Categorical sampling")
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--label", type=str, default=None)
    p.add_argument("--search-horizon", type=int, default=24,
                   help="search: rollout length in agent steps")
    p.add_argument("--search-commit", type=int, default=3,
                   help="search: steps the candidate is held")
    p.add_argument("--search-rollouts", type=int, default=2,
                   help="search: rollouts per candidate")
    p.add_argument("--levels", type=Path, default=None,
                   help="evaluate per level from the level starts in DIR")
    p.add_argument("--max-steps", type=int, default=3000,
                   help="--levels: steps allowed per level attempt")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = select_device(args.force_cpu)
    # episodic_life=False: one env episode == one full game, so per-life
    # segmentation comes from death events, not from done.
    env = SymbolicDiggerEnv(max_steps=10**9, clip_reward=False,
                            episodic_life=False, **env_kwargs(args.policy))
    search_kwargs = (dict(horizon=args.search_horizon,
                          commit=args.search_commit,
                          rollouts=args.search_rollouts)
                     if args.policy == "search" else {})
    policy = make_policy(args.policy, env, device, argmax=args.argmax,
                         seed=args.seed, **search_kwargs)

    label = args.label or args.policy
    if args.levels is not None:
        evaluate_levels(env, policy, args, label)
        env.close()
        return
    game_scores: list[int] = []
    life_lengths: list[int] = []
    clean_l1 = 0
    any_l1 = 0

    for ep in range(args.episodes):
        obs = env.reset()
        policy.reset()
        steps_this_life = 0
        score = 0
        died_yet = False
        levelled = False
        clean_game = False
        prev_em = None
        while True:
            a = policy(obs)
            obs, _, done, info = env_step_skipped(env, a, args.frame_skip)
            policy.observe(info)
            steps_this_life += 1
            score = int(info.get("score", score))
            # A level-up shows as the emerald grid refilling.
            n_em = int(env._last_state.emeralds.sum())
            if prev_em is not None and n_em - prev_em >= 4 and not levelled:
                levelled = True
                any_l1 += 1
                clean_game = not died_yet
                clean_l1 += clean_game
            prev_em = n_em
            if info.get("death_event", False):
                died_yet = True
                life_lengths.append(steps_this_life)
                steps_this_life = 0
            elif steps_this_life and info.get("dying", False):
                steps_this_life -= 1   # don't count dead time
            if done:
                break
        game_scores.append(score)
        print(f"  [{label}] game {ep + 1}/{args.episodes} "
              f"score={score} lives_recorded={len(life_lengths)} "
              f"L1={'clean' if clean_game else ('yes' if levelled else 'no')}",
              flush=True)
        if args.policy == "search":
            print(f"  [{label}]   {policy.teacher.summary()}", flush=True)

    env.close()

    def sem(xs):
        return statistics.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else 0.0

    print(f"\n=== {label} ({args.episodes} games"
          f"{', argmax' if args.argmax else ', stochastic'}) ===")
    print(f"game score : mean {statistics.mean(game_scores):7.1f} "
          f"+-{sem(game_scores):5.1f}  median {statistics.median(game_scores):6.0f}  "
          f"max {max(game_scores)}")
    if life_lengths:
        print(f"life length: mean {statistics.mean(life_lengths):7.1f} "
              f"+-{sem(life_lengths):5.1f}  median "
              f"{statistics.median(life_lengths):6.0f}  "
              f"max {max(life_lengths)}  n={len(life_lengths)} agent steps")
    print(f"level 1    : cleared without dying {clean_l1}/{args.episodes}  "
          f"cleared at all {any_l1}/{args.episodes}")


def evaluate_levels(env, policy, args, label: str) -> None:
    starts = sorted(args.levels.glob("L*_*_0.pkl"),
                    key=lambda f: int(re.match(r"L(\d+)_", f.name)[1]))
    if not starts:
        raise SystemExit(f"no level starts (L*_*_0.pkl) in {args.levels}")
    env.reset()                               # boots the core to load into
    res = collections.defaultdict(list)       # level -> [(clear, eaten, pts)]
    for f in starts:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        level = d["level"]
        for _ in range(args.episodes):
            obs = env.load_state(d["state"])
            policy.reset()
            n0 = int(env._last_state.emeralds.sum())
            score0 = score = d["score"]
            n_left, cleared = n0, False
            for _ in range(args.max_steps):
                a = policy(obs)
                obs, _, done, info = env_step_skipped(env, a, args.frame_skip)
                policy.observe(info)
                score = int(info.get("score", score))
                if env._env.read_level() != level:
                    cleared = True
                    break
                n_left = int(env._last_state.emeralds.sum())
                if done or info.get("death_event", False) \
                        or info.get("dying", False):
                    break
            eaten = 1.0 if cleared else 1 - n_left / max(n0, 1)
            res[level].append((cleared, eaten, score - score0))
            print(f"  [{label}] L{level} {f.stem}: "
                  f"{'CLEAR' if cleared else 'died '} "
                  f"eaten {eaten:4.0%} +{score - score0}", flush=True)
    print(f"\n=== {label} per level ({args.episodes} x "
          f"{len(starts)} starts, one life each"
          f"{', argmax' if args.argmax else ', stochastic'}) ===")
    print("level  n  cleared  emeralds eaten  score gained")
    for level in sorted(res):
        r = res[level]
        print(f"{level:5d} {len(r):2d}  {sum(c for c, _, _ in r):3d}/{len(r):<3d}"
              f"  {statistics.mean(e for _, e, _ in r):13.0%}"
              f"  {statistics.mean(p for _, _, p in r):12.0f}")


if __name__ == "__main__":
    main()
