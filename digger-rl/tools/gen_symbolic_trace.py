"""Generate a BC trace by playing DIGGER with a heuristic policy.

Records the *symbolic* (C*frame_stack, 10, 15) observation at each policy
step alongside the action the heuristic chose. The resulting .npz is
consumed by `train_ppo_symbolic.py --bc-traces PATH` as supervised data
for the actor (and optionally the critic via MC returns).

Why symbolic + SmartHeuristic: prior DAGGER pixel runs used GreedyEmerald
as teacher and topped out at 783; SmartHeuristic v5 scores ~1475 vs
GreedyEmerald's ~1342 baseline and includes monster avoidance + LoS fire,
so the BC labels carry strictly more useful signal. Recording the
*symbolic* obs avoids the pixel-to-tile perception problem that bottle-
necked the pixel runs.

    python -m tools.gen_symbolic_trace --out data/sym_smart.npz \\
        --steps 50000 --frame-stack 4 --teacher smart
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from tools.heuristic_agent import DodgeMonsters, GreedyEmerald, SmartHeuristic
from tools.symbolic_env import BASE_OBS_CHANNELS, SymbolicDiggerEnv
from train_dagger import env_step_skipped


def collect(num_steps: int, frame_skip: int, frame_stack: int,
             teacher_name: str) -> dict:
    env = SymbolicDiggerEnv(max_steps=10**9, episodic_life=True,
                             frame_stack=frame_stack)
    teachers = {
        "smart":  SmartHeuristic,
        "greedy": GreedyEmerald,
        "dodge":  DodgeMonsters,
    }
    if teacher_name not in teachers:
        raise SystemExit(f"unknown teacher {teacher_name!r}; "
                         f"choose from {sorted(teachers)}")
    teacher = teachers[teacher_name]()
    in_ch = BASE_OBS_CHANNELS * frame_stack

    obs_buf = np.zeros((num_steps, in_ch, 10, 15), dtype=np.float32)
    act_buf = np.zeros((num_steps,), dtype=np.int64)
    rew_buf = np.zeros((num_steps,), dtype=np.float32)
    done_buf = np.zeros((num_steps,), dtype=bool)
    score_buf = np.zeros((num_steps,), dtype=np.int32)

    cur_obs = env.reset()
    teacher.reset()
    t0 = time.monotonic()
    ep_score_at_death: list[int] = []
    last_print = t0

    for t in range(num_steps):
        obs_buf[t] = cur_obs
        action = teacher(env._last_state)
        act_buf[t] = action
        next_obs, total_r, done, info = env_step_skipped(
            env, int(action), frame_skip)
        rew_buf[t] = float(total_r)
        done_buf[t] = bool(done)
        score_buf[t] = int(info.get("score", 0))
        if done:
            ep_score_at_death.append(int(info.get("score", 0)))
            cur_obs = env.reset()
            teacher.reset()
        else:
            cur_obs = next_obs

        now = time.monotonic()
        if now - last_print > 5.0:
            sps = (t + 1) / (now - t0)
            avg = (sum(ep_score_at_death) / max(1, len(ep_score_at_death)))
            print(f"  {t+1:>7d}/{num_steps}  sps {sps:.0f}  "
                  f"episodes {len(ep_score_at_death)}  "
                  f"mean death-score {avg:.0f}", flush=True)
            last_print = now

    env.close()
    elapsed = time.monotonic() - t0
    final_avg = (sum(ep_score_at_death) / max(1, len(ep_score_at_death)))
    print(f"\nCollected {num_steps} steps in {elapsed:.0f}s "
          f"({len(ep_score_at_death)} episodes, "
          f"mean death-score {final_avg:.0f})", flush=True)
    return dict(obs=obs_buf, actions=act_buf, rewards=rew_buf,
                 dones=done_buf, scores=score_buf,
                 frame_skip=np.int32(frame_skip),
                 frame_stack=np.int32(frame_stack),
                 obs_channels=np.int32(in_ch),
                 teacher=np.array(teacher_name, dtype=object))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True,
                   help="output .npz path")
    p.add_argument("--steps", type=int, default=50_000,
                   help="number of policy-steps to record")
    p.add_argument("--frame-skip", type=int, default=4,
                   help="must match the trainer's --frame-skip")
    p.add_argument("--frame-stack", type=int, default=4,
                   help="must match the trainer's --frame-stack")
    p.add_argument("--teacher", type=str, default="smart",
                   choices=["smart", "greedy", "dodge"],
                   help="which heuristic to record (smart = SmartHeuristic v5)")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data = collect(args.steps, args.frame_skip, args.frame_stack,
                   args.teacher)
    np.savez_compressed(args.out, **data)
    size_mb = args.out.stat().st_size / 1e6
    print(f"Wrote {size_mb:.1f} MB to {args.out}")


if __name__ == "__main__":
    main()
