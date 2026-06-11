"""Load a saved pixel PPO / BC checkpoint and evaluate over N episodes.

For tightening estimates from the 5/10-episode evals baked into the
trainer. Variance across single Digger episodes is huge (score range
~150 to ~1500 for the same policy), so a 50+ episode pass is needed
to compare two policies without seed-luck noise.

Uses the same stochastic Categorical sampling the trainer uses --
matches the in-loop ep_end / mid-eval semantics so numbers are
directly comparable to what the trainer printed.

Usage:
    python eval_checkpoint.py CHECKPOINT [--episodes 50]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from digger_env import DiggerEnv, DiggerVecEnv
from train_ppo import Agent, select_device


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    cfg = ckpt["config"]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = select_device(args.force_cpu)

    in_ch = cfg["frame_stack"] * (3 if cfg["color"] else 1)
    agent = Agent(num_actions=DiggerEnv.NUM_ACTIONS,
                  in_channels=in_ch,
                  width=cfg["encoder_width"]).to(device)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()

    env_kwargs = dict(max_steps=10**9,
                      clip_reward=cfg["clip_reward"],
                      episodic_life=cfg["episodic_life"],
                      death_penalty=0.0)  # eval-time: no reward shaping
    vec = DiggerVecEnv(num_envs=1, frame_skip=cfg["frame_skip"],
                       frame_stack=cfg["frame_stack"],
                       obs_size=cfg["obs_size"],
                       color=cfg["color"], env_kwargs=env_kwargs)

    print(f"checkpoint: {args.checkpoint}")
    print(f"  trained on: warmup_steps={cfg.get('warmup_steps')}, "
          f"warmup_epochs={cfg.get('warmup_epochs')}, "
          f"dagger_iters={cfg.get('dagger_iters', 0)}")
    print(f"  agent: width={cfg['encoder_width']}, in_ch={in_ch} "
          f"({'color' if cfg['color'] else 'gray'}), "
          f"episodic_life={cfg['episodic_life']}")
    print(f"  device: {device}")
    print(f"  evaluating {args.episodes} episodes...")
    print()

    scores: list[int] = []
    obs_np = vec.reset()
    obs_t = torch.from_numpy(obs_np).to(device).float().mul_(1.0 / 255.0)
    while len(scores) < args.episodes:
        with torch.no_grad():
            action, _, _, _ = agent.act(obs_t)
        obs_np, _, dones, infos = vec.step(action.cpu().numpy())
        obs_t = torch.from_numpy(obs_np).to(device).float().mul_(1.0 / 255.0)
        if dones[0]:
            score = int(infos[0].get("score", 0))
            scores.append(score)
            print(f"  ep {len(scores):>3d}/{args.episodes}: score {score}",
                  flush=True)

    arr = np.array(scores, dtype=np.int32)
    n = len(arr)
    sem = float(arr.std() / np.sqrt(n))
    print()
    print(f"== {args.checkpoint.parent.name} ==")
    print(f"  episodes : {n}")
    print(f"  mean     : {arr.mean():.1f}  (sem +/- {sem:.1f})")
    print(f"  median   : {int(np.median(arr))}")
    print(f"  std      : {arr.std():.1f}")
    print(f"  min/max  : {arr.min()} / {arr.max()}")
    print(f"  quartiles: 25%={int(np.percentile(arr, 25))}, "
          f"75%={int(np.percentile(arr, 75))}")
    vec.close()


if __name__ == "__main__":
    main()
