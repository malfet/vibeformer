"""Hand-coded baseline policy: greedy walk toward nearest emerald.

The user observed that Digger can dig through everything except bags, so
no A* path search is needed -- the agent just picks the action that takes
it one Manhattan step closer to the closest emerald, and the env's dirt
takes care of itself.

Compared to PPO/Dreamer baselines, this gives us:
  - a sanity check that the symbolic env is functional end-to-end
  - a yardstick: any learned agent should at least match the greedy
    policy on level 1, otherwise the learning isn't doing anything
  - a candidate teacher policy for imitation-learning bootstraps

    python heuristic_agent.py --episodes 5 --max-steps 4000
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from digger_env import DiggerEnv
from game_state import MWIDTH, MHEIGHT
from symbolic_env import SymbolicDiggerEnv


def greedy_emerald(state) -> int:
    """One step of greedy Manhattan navigation toward the nearest emerald."""
    if state.digger is None or not state.digger.present:
        return DiggerEnv.NOOP
    em_rows, em_cols = np.where(state.emeralds)
    if em_rows.size == 0:
        return DiggerEnv.NOOP
    dr, dc = state.digger.row, state.digger.col
    distances = np.abs(em_rows - dr) + np.abs(em_cols - dc)
    i = int(np.argmin(distances))
    er, ec = int(em_rows[i]), int(em_cols[i])
    # Resolve the larger gap first so the path is mostly L-shaped, not
    # zigzag (slight optimisation against the digger spending extra
    # frames at tile boundaries when alternating).
    if abs(ec - dc) >= abs(er - dr):
        if ec < dc: return DiggerEnv.LEFT
        if ec > dc: return DiggerEnv.RIGHT
        if er < dr: return DiggerEnv.UP
        if er > dr: return DiggerEnv.DOWN
    else:
        if er < dr: return DiggerEnv.UP
        if er > dr: return DiggerEnv.DOWN
        if ec < dc: return DiggerEnv.LEFT
        if ec > dc: return DiggerEnv.RIGHT
    return DiggerEnv.NOOP


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=4000,
                   help="env.step calls per episode before timing out")
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--no-episodic-life", action="store_true",
                   help="play to actual game-over (default); without this, "
                        "each life-loss counts as an episode")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env = SymbolicDiggerEnv(max_steps=10**9, episodic_life=not args.no_episodic_life)

    scores: list[int] = []
    lengths: list[int] = []
    t0 = time.monotonic()

    for ep in range(args.episodes):
        env.reset()
        # symbolic_env exposes the last extracted GameState as _last_state
        ep_score = 0
        ep_len = 0
        for step in range(args.max_steps):
            action = greedy_emerald(env._last_state)
            # frame_skip aggregation
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
    print(f"=== heuristic baseline: greedy nearest-emerald ===")
    print(f"  episodes: {args.episodes}")
    print(f"  mean score:  {arr.mean():.1f}")
    print(f"  median:      {np.median(arr):.0f}")
    print(f"  min/max:     {arr.min()} / {arr.max()}")
    print(f"  mean length: {np.mean(lengths):.0f} steps")
    print(f"  wall time:   {elapsed:.1f}s")
    env.close()


if __name__ == "__main__":
    main()
