"""Hesitation start states: where a policy is unsure, or dithers.

A pixel policy plays (from tools/level_states.py starts, or level 1)
and two kinds of moments are saved as start states:

  - unsure: the policy's entropy is at least `--entropy` (ln 6 = 1.79 is
    uniform; the trained policies sit near 0 most of the time). The
    state `--back` steps earlier is saved, so the start replays the
    approach to the decision too;
  - dithering: over the last `--window` steps the digger stayed on at
    most 2 tiles and scored nothing (moving back and forth, or stuck).
    The state at the start of that window is saved.

Saves are at least `--gap` steps apart, so one long hesitation gives
one state. Use the directory as a train_pixel.py --start-states pool;
PPO from there pays for whichever way out scores sooner.

    python -m tools.hesitation checkpoints/digger_px128_exit1.pt \\
        --out-dir data/save/hesitation --states 200 --seed 0
"""

from __future__ import annotations

import argparse
import collections
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tools.policies import env_kwargs, make_policy
from tools.symbolic_env import SymbolicDiggerEnv, env_step_skipped


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("policy", help="pixel checkpoint")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--states", type=int, default=200,
                   help="stop after saving this many states")
    p.add_argument("--entropy", type=float, default=1.2,
                   help="save where the policy's entropy is at least this")
    p.add_argument("--back", type=int, default=4,
                   help="save the state this many steps before an unsure one")
    p.add_argument("--window", type=int, default=20,
                   help="steps on <= 2 tiles without scoring that count as "
                        "dithering")
    p.add_argument("--gap", type=int, default=50,
                   help="minimum steps between saved states")
    p.add_argument("--max-steps", type=int, default=200_000)
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
    if not hasattr(policy, "_act"):
        raise SystemExit("needs a pixel or symbolic checkpoint")
    keep = max(args.back, args.window) + 1

    stats = collections.Counter()
    ents: list[float] = []
    t0 = time.monotonic()

    def fresh():
        env.reset()
        policy.reset()
        return env.current_obs(), collections.deque(maxlen=keep)

    obs, hist = fresh()
    steps, last_save = 0, -args.gap

    def save(kind: str, back: int) -> None:
        nonlocal last_save
        state, lives, _, _ = hist[len(hist) - 1 - back]
        level = env._env.read_level()
        name = f"L{level}_{kind}{args.seed}_{stats['saved']}.pkl"
        with open(args.out_dir / name, "wb") as fh:
            pickle.dump(dict(state=state, level=level, step=-back,
                             lives=lives, kind=kind), fh)
        stats["saved"] += 1
        stats[kind] += 1
        stats[f"{kind}_L{level}"] += 1
        last_save = steps
        print(f"{kind} at step {steps} L{level}: saved {stats['saved']} "
              f"({time.monotonic() - t0:.0f}s)", flush=True)

    while stats["saved"] < args.states and steps < args.max_steps:
        s = env._last_state
        tile = (s.digger.col, s.digger.row) if s.digger is not None else None
        hist.append((env.save_state(), s.lives, tile, env._env._last_score))
        a = policy(obs)
        ent = float(-(F.softmax(policy.last_logits.float(), -1)
                      * F.log_softmax(policy.last_logits.float(), -1)).sum())
        ents.append(ent)
        obs, _, done, info = env_step_skipped(env, a, 4)
        steps += 1
        policy.observe(info)
        if done or info.get("dying", False):
            if done:
                obs, hist = fresh()
            else:
                hist.clear()            # no starts from inside a death
            continue
        if steps - last_save < args.gap:
            continue
        if ent >= args.entropy and len(hist) > args.back:
            save("unsure", args.back)
        elif len(hist) > args.window:
            recent = list(hist)[-args.window:]
            tiles = {h[2] for h in recent}
            if (None not in tiles and len(tiles) <= 2
                    and recent[-1][3] == recent[0][3]):
                save("dither", args.window)
    env.close()
    q = np.quantile(ents, [0.5, 0.9, 0.99]) if ents else [0, 0, 0]
    print(f"\n{steps} steps, entropy median {q[0]:.3f} p90 {q[1]:.3f} "
          f"p99 {q[2]:.3f}, {dict(stats)}")


if __name__ == "__main__":
    main()
