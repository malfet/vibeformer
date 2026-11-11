"""Record teacher-labelled traces for behavioural cloning / DAgger.

Each policy step stores the symbolic (13*frame_stack, 10, 15) uint8
observation and the teacher's (SmartHeuristic's) action. Who *plays* is
separate from who labels:

  - default: the teacher plays (plain BC data);
  - --student CKPT: a train_symbolic.py checkpoint plays (DAgger), so the
    labels cover the states the student actually reaches;
  - --explore P: with probability P a random action is played instead.

The label is always the teacher's. Parallel collectors need different
--seed values (the emulator itself is deterministic).

    for i in $(seq 0 7); do
      python collect.py --out data/traces/bc_$i.npz --steps 25000 \
          --explore 0.05 --seed $i &
    done; wait
    python train_symbolic.py --total-timesteps 0 \
        --bc-traces data/traces/bc_{0..6}.npz --bc-val data/traces/bc_7.npz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from digger_env import DiggerEnv
from tools.heuristic_agent import DodgeMonsters, GreedyEmerald, SmartHeuristic
from tools.symbolic_env import BASE_OBS_CHANNELS, SymbolicDiggerEnv
from tools.symbolic_env import env_step_skipped


def collect(num_steps: int, frame_skip: int, frame_stack: int,
             teacher_name: str, egocentric: bool = False,
             explore: float = 0.0, seed: int = 0,
             student: Path | None = None) -> dict:
    env = SymbolicDiggerEnv(max_steps=10**9, episodic_life=True,
                             frame_stack=frame_stack,
                             egocentric=egocentric)
    teachers = {
        "smart":  SmartHeuristic,
        "greedy": GreedyEmerald,
        "dodge":  DodgeMonsters,
    }
    if teacher_name not in teachers:
        raise SystemExit(f"unknown teacher {teacher_name!r}; "
                         f"choose from {sorted(teachers)}")
    teacher = teachers[teacher_name]()
    in_ch, obs_h, obs_w = env.obs_shape

    # Every plane is a 0/1 mask, so uint8 loses nothing at 1/4 the size.
    obs_buf = np.zeros((num_steps, in_ch, obs_h, obs_w), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    if student is not None:
        # DAgger: the student drives, the teacher only labels. Imported
        # here so plain collection doesn't need torch.
        import torch
        from tools.policies import load_symbolic_agent
        torch.manual_seed(seed)
        agent, _ = load_symbolic_agent(student, torch.device("cpu"))
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
        if student is not None:
            with torch.no_grad():
                x = torch.from_numpy(np.asarray(cur_obs, np.float32))[None]
                logits = agent.actor(agent.encode(x))
                action = int(torch.distributions.Categorical(
                    logits=logits).sample().item())
        # Noise injection: sometimes play a random action, but always
        # record the teacher's. The student then sees states the teacher
        # would not reach, with the teacher's recovery as the label, and
        # parallel collectors (the emulator is deterministic) diverge.
        if explore > 0 and rng.random() < explore:
            action = int(rng.integers(DiggerEnv.NUM_ACTIONS))
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
                 egocentric=np.bool_(egocentric),
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
                   help="which heuristic to record (smart = SmartHeuristic)")
    p.add_argument("--explore", type=float, default=0.0,
                   help="probability of playing a random action instead of "
                        "the teacher's (the label stays the teacher's)")
    p.add_argument("--student", type=Path, default=None,
                   help="DAgger: a train_symbolic checkpoint that plays "
                        "(sampled); the teacher's action is still the label")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for --explore; give parallel collectors "
                        "different seeds")
    p.add_argument("--egocentric", action="store_true",
                   help="record the digger-centred obs; must match the "
                        "trainer's --egocentric")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data = collect(args.steps, args.frame_skip, args.frame_stack,
                   args.teacher, args.egocentric, args.explore, args.seed,
                   args.student)
    np.savez_compressed(args.out, **data)
    size_mb = args.out.stat().st_size / 1e6
    print(f"Wrote {size_mb:.1f} MB to {args.out}")


if __name__ == "__main__":
    main()
