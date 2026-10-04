"""Record pixel BC traces labelled by the search teacher.

The search teacher costs ~0.3 s per step, so collection runs as many
independent processes, each writing its own .npz; `train_ppo.py
--bc-traces` takes them all. Format matches tools/gen_heuristic_trace.py
except that frames[t] is the frame the policy saw *when it chose*
actions[t] (the trainer pairs frames[t] with actions[t]).

Episodes are single lives (episodic_life=True): the env ends the
episode at death onset and reset() skips the death sequence, which is
what DiggerVecEnv shows a deployed pixel agent. The first sample after
each reset is flagged so the trainer never stacks across a boundary.

    # teacher drives (plain BC data)
    python -m tools.gen_search_trace --out data/traces/search_px_0.npz --steps 7000

    # DAgger: a pixel checkpoint drives, the search teacher labels
    python -m tools.gen_search_trace --out data/traces/search_px_d1_0.npz \\
        --steps 3000 --student data/checkpoints/<run>/ppo_digger_bc_only.pt
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from digger_env import DiggerEnv, preprocess_uint8
from tools.search_teacher import SearchTeacher
from tools.symbolic_env import SymbolicDiggerEnv, state_to_tensor
from train_dagger import env_step_skipped


def _resize_rgb(rgba: np.ndarray, h: int, w: int) -> np.ndarray:
    """Area-resize keeping the frame's 1.6 aspect ratio (preprocess_uint8
    squashes to a square). (h, w, 3) uint8."""
    import torch.nn.functional as F
    t = torch.from_numpy(rgba[..., :3]).permute(2, 0, 1)[None].float()
    t = F.interpolate(t, size=(h, w), mode="area")
    return t[0].permute(1, 2, 0).round().clamp(0, 255).byte().numpy()


def _load_student(path: Path):
    from train_ppo import Agent
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    cfg = ckpt["config"]
    in_ch = cfg["frame_stack"] * (3 if cfg["color"] else 1)
    agent = Agent(num_actions=DiggerEnv.NUM_ACTIONS, in_channels=in_ch,
                  width=cfg["encoder_width"],
                  separate_critic=cfg.get("separate_critic", False),
                  in_hw=tuple(cfg.get("obs_hw", (cfg["obs_size"],) * 2)))
    agent.load_state_dict(ckpt["agent"])
    agent.eval()
    return agent, cfg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--obs-size", type=int, default=84)
    p.add_argument("--student", type=Path, default=None,
                   help="pixel checkpoint that drives (DAgger); the search "
                        "teacher still provides every label")
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--rollouts", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hires", type=str, default="100x160",
                   help="also store native-aspect HxW frames ('' to skip)")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    torch.set_num_threads(1)   # many collectors share the machine
    env = SymbolicDiggerEnv(max_steps=10**9, clip_reward=False,
                            episodic_life=True, frame_stack=1)
    teacher = SearchTeacher(env, horizon=a.horizon, rollouts=a.rollouts,
                            frame_skip=a.frame_skip, verbose=False)
    student = stack = None
    if a.student is not None:
        student, scfg = _load_student(a.student)
        if scfg["obs_size"] != a.obs_size or scfg["frame_skip"] != a.frame_skip:
            raise SystemExit("student obs_size/frame_skip differ from --obs-size/--frame-skip")
        stack = collections.deque(maxlen=scfg["frame_stack"])

    frames, labels, executed, rewards, scores, lives, resets = ([] for _ in range(7))
    syms, hires = [], []
    hr_hw = tuple(int(v) for v in a.hires.split("x")) if a.hires else None
    env.reset()
    teacher.reset()
    pending_reset = True
    agree = 0
    t0 = time.monotonic()
    while len(frames) < a.steps:
        raw_frame = env._env._core.get_frame()
        proc = preprocess_uint8(raw_frame, a.obs_size, True)
        # Symbolic obs of the same decision-time state: lets the same labels
        # train a symbolic student, separating perception from imitation.
        syms.append(state_to_tensor(env._last_state).astype(np.uint8))
        if hr_hw:
            hires.append(_resize_rgb(raw_frame, *hr_hw))
        if student is not None and pending_reset:
            stack.clear()
            stack.extend([proc] * stack.maxlen)
        elif student is not None:
            stack.append(proc)
        label = int(teacher(env._last_state))
        act = label
        if student is not None:
            obs = np.concatenate([f.transpose(2, 0, 1) for f in stack], 0)
            with torch.no_grad():
                x = torch.from_numpy(obs).float().div_(255.0).unsqueeze(0)
                act, _, _, _ = student.act(x)
            act = int(act.item())
        agree += act == label
        _, r, done, info = env_step_skipped(env, act, a.frame_skip)
        frames.append(proc)
        labels.append(label)
        executed.append(act)
        rewards.append(float(info.get("score_reward", r)))
        scores.append(int(info.get("score", 0)))
        lives.append(int(info.get("lives", 0)))
        resets.append(pending_reset)
        pending_reset = False
        if done:
            env.reset()
            teacher.reset()
            pending_reset = True
        if len(frames) % 500 == 0:
            el = time.monotonic() - t0
            print(f"  {len(frames)}/{a.steps}  {len(frames) / el:.1f} steps/s  "
                  f"agree {agree / len(frames):.1%}  {teacher.summary()}",
                  flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out,
        frames=np.stack(frames).astype(np.uint8),
        sym=np.stack(syms),
        **({"frames_hires": np.stack(hires)} if hr_hw else {}),
        actions=np.array(labels, dtype=np.uint8),
        executed=np.array(executed, dtype=np.uint8),
        raw_rewards=np.array(rewards, dtype=np.float32),
        scores=np.array(scores, dtype=np.int32),
        lives=np.array(lives, dtype=np.uint8),
        resets=np.array(resets, dtype=bool),
        frame_skip=np.int32(a.frame_skip),
        obs_size=np.int32(a.obs_size),
        color=np.bool_(True),
    )
    print(f"Wrote {len(frames)} samples to {a.out}; episodes {sum(resets)}; "
          f"label histogram {np.bincount(labels, minlength=6).tolist()}; "
          f"student/teacher agree {agree / len(frames):.1%}")


if __name__ == "__main__":
    main()
