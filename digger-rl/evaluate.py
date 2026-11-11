"""Evaluate any policy over N full games.

    python evaluate.py teacher --episodes 20          # same-session baseline
    python evaluate.py data/checkpoints/RUN/ppo_sym_final.pt --episodes 20
    python evaluate.py data/checkpoints/RUN/ppo_digger_final.pt   # pixel ckpt
    python evaluate.py random | search

The policy is a spec from tools/policies.py: teacher, search, random, or a
checkpoint from train_symbolic.py or train_pixel.py. Reports:

  - **game score**: total score across a full 3-life game.
  - **life length**: agent steps between deaths, segmented at death onset
    (RAM death stage), excluding the ~100 input-ignored dying steps.
  - **level 1**: games that cleared level 1, and that did so without
    losing a life (the project goal).

Cross-session numbers drift; compare against a same-session teacher run.
"""

from __future__ import annotations

import argparse
import statistics

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


if __name__ == "__main__":
    main()
