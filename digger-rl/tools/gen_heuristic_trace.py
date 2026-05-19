"""Generate a BC trace by playing DIGGER with the greedy heuristic.

Output format matches `run_digger.py --record-playtrace` exactly so the
resulting .npz can be fed directly to `train_ppo.py --bc-traces`.

Why: the human play traces we recorded are noisy (mixed strategies,
inconsistent reactions). The greedy heuristic gives noiseless labels
-- the same symbolic state always yields the same action -- which is a
much cleaner BC training signal. Lets us cleanly test whether pixel
PPO can learn the representation from teacher demonstrations at all.

    python -m tools.gen_heuristic_trace --out data/heuristic.npz --steps 10000

The script captures per policy-step:
  preprocessed frame (84x84 RGB by default) at decision time,
  action chosen by the heuristic,
  summed raw reward across the frame-skip window,
  end-of-window score / lives,
  reset flag (True for the first sample after a heuristic-induced reset
    on game-over).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from digger_env import DiggerEnv, preprocess_uint8
from tools.game_state import extract_state_fast
from tools.heuristic_agent import GreedyEmerald


def collect(num_steps: int, frame_skip: int, color: bool, obs_size: int):
    env = DiggerEnv(max_steps=10**9)
    chaser = GreedyEmerald()

    raw = env.reset()
    state = extract_state_fast(raw)

    frames: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    scores: list[int] = []
    lives: list[int] = []
    resets: list[bool] = []

    pending_reset = False    # mark next snapshot as the start of a new episode
    seen_alive = False
    last_score = 0; last_lives = 0
    t0 = time.monotonic()

    while len(frames) < num_steps:
        action = chaser(state)

        total_r = 0.0
        last_info: dict = {}
        last_raw = None
        ended = False
        for _ in range(frame_skip):
            s = env.step(action)
            total_r += float(s.reward)
            last_info = s.info
            last_raw = s.obs
            if s.done:
                ended = True
                break

        # The frame at the END of the skip window is what the next
        # decision will see; it matches what run_digger.py records.
        proc = preprocess_uint8(last_raw, obs_size, color)
        frames.append(proc)
        actions.append(action)
        rewards.append(total_r)
        scores.append(int(last_info.get("score", last_score)))
        lives.append(int(last_info.get("lives", last_lives)))
        resets.append(pending_reset)
        pending_reset = False
        last_score = scores[-1]; last_lives = lives[-1]

        # Track alive state for game-over detection.
        if last_lives > 0:
            seen_alive = True

        state = extract_state_fast(last_raw)

        if ended or (seen_alive and last_lives == 0):
            # Game over -> reset env + chaser, mark next sample as a new
            # episode boundary.
            raw = env.reset()
            state = extract_state_fast(raw)
            chaser.reset()
            pending_reset = True
            seen_alive = False

        if len(frames) % 500 == 0:
            elapsed = time.monotonic() - t0
            sps = len(frames) / max(elapsed, 1e-6)
            print(f"  collected {len(frames):>6d}/{num_steps}  "
                  f"score so far {last_score}  "
                  f"({sps:.0f} steps/s)", flush=True)

    env.close()
    return frames, actions, rewards, scores, lives, resets


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True,
                   help="output .npz path (e.g. data/heuristic_trace.npz)")
    p.add_argument("--steps", type=int, default=10_000,
                   help="number of policy steps to record")
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--obs-size", type=int, default=84)
    p.add_argument("--color", default=True,
                   action=argparse.BooleanOptionalAction,
                   help="record RGB (default) or --no-color for grayscale")
    args = p.parse_args()

    print(f"recording {args.steps} heuristic transitions to {args.out}")
    print(f"  frame_skip={args.frame_skip}  obs_size={args.obs_size}  "
          f"color={args.color}", flush=True)

    frames, actions, rewards, scores, lives, resets = collect(
        args.steps, args.frame_skip, args.color, args.obs_size)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        frames=np.stack(frames, axis=0).astype(np.uint8),
        actions=np.array(actions, dtype=np.uint8),
        raw_rewards=np.array(rewards, dtype=np.float32),
        scores=np.array(scores, dtype=np.int32),
        lives=np.array(lives, dtype=np.uint8),
        resets=np.array(resets, dtype=bool),
        frame_skip=np.int32(args.frame_skip),
        obs_size=np.int32(args.obs_size),
        color=np.bool_(args.color),
    )
    n_resets = sum(resets)
    final_score = scores[-1] if scores else 0
    counts = np.bincount(actions, minlength=6)
    print()
    print(f"Wrote {len(frames)} samples to {args.out}")
    print(f"  final score {final_score},  episode boundaries: {n_resets}")
    print(f"  action distribution (NOOP/L/R/U/D/FIRE): {counts.tolist()}")


if __name__ == "__main__":
    main()
