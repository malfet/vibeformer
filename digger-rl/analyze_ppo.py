"""Headless rollout of a PPO checkpoint with per-step action analysis.

Runs the agent through `--episodes` full games (no episodic-life soft reset),
prints a per-event log (reward / life change), and an end-of-episode summary
with action distribution, average entropy, value-estimate stats.

    python analyze_ppo.py data/checkpoints/exp4_both_traces/ppo_digger_final.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from digger_env import (
    DiggerEnv, FrameStack, _env_step_skipped, preprocess_uint8,
)
from train_ppo import Agent, select_device

env_step_skipped = _env_step_skipped

def preprocess(rgba, size=84, color=False):
    """Float32 84x84 in [0,1]; grayscale by default."""
    return preprocess_uint8(rgba, size, color).astype(np.float32) * (1.0 / 255.0)

ACTION_NAMES = ["NOOP", "LEFT", "RIGHT", "UP", "DOWN", "FIRE"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--stochastic", action="store_true",
                   help="sample from the policy instead of argmax")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--obs-size", type=int, default=84)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.force_cpu)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_cfg = ckpt.get("config", {})
    width = int(ckpt_cfg.get("encoder_width", 1))
    color = bool(ckpt_cfg.get("color", False))
    in_ch = args.frame_stack * (3 if color else 1)
    agent = Agent(num_actions=DiggerEnv.NUM_ACTIONS,
                  in_channels=in_ch, width=width).to(device)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()
    step_trained = ckpt.get("step", "?")
    print(f"checkpoint: {args.checkpoint}")
    print(f"  trained for {step_trained} emulator frames; "
          f"policy mode = {'stochastic' if args.stochastic else 'argmax'}")
    print()

    # episodic_life=False so an episode runs to a real game-over.
    env = DiggerEnv(max_steps=10**6)

    for ep in range(args.episodes):
        stack = FrameStack(k=args.frame_stack, size=args.obs_size, color=color)
        raw = env.reset()
        obs = stack.reset(preprocess(raw, args.obs_size, color))

        action_counts = [0] * DiggerEnv.NUM_ACTIONS
        ent_sum = 0.0
        v_sum = 0.0
        n_decisions = 0
        events: list[str] = []
        last_score = 0
        last_lives = -1
        seen_alive = False
        step_no = 0

        while True:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                z = agent.encode(obs_t)
                logits = agent.actor(z)
                v = agent.critic(z).squeeze(-1)
                dist = Categorical(logits=logits)
                if args.stochastic:
                    a = dist.sample()
                else:
                    a = logits.argmax(-1)
                action = int(a[0].item())
                ent = float(dist.entropy()[0].item())
                probs = dist.probs[0].cpu().numpy()
                v_val = float(v[0].item())

            raw, reward, done, info = env_step_skipped(
                env, action, args.frame_skip)
            step_no += args.frame_skip
            action_counts[action] += 1
            ent_sum += ent
            v_sum += v_val
            n_decisions += 1

            score = info["score"]
            lives = info["lives"]
            if lives > 0:
                seen_alive = True
            if last_lives < 0:
                last_lives = lives

            # Log when score or lives change. Include the policy's top-3
            # action probs so we can see whether the policy was decisive
            # or hedging at the decision moment.
            score_changed = score != last_score
            life_lost = lives < last_lives
            if score_changed or life_lost:
                top3 = sorted(enumerate(probs), key=lambda x: -x[1])[:3]
                top_str = " ".join(
                    f"{ACTION_NAMES[i]}={p:.2f}" for i, p in top3)
                tag = []
                if score_changed:
                    tag.append(f"+{score - last_score}=>{score}")
                if life_lost:
                    tag.append(f"DIED {last_lives}->{lives}")
                events.append(
                    f"  step {step_no:>5d}  a={ACTION_NAMES[action]:<5s}  "
                    f"V={v_val:+7.1f}  H={ent:.2f}  "
                    f"{'  '.join(tag):<25s}  top: {top_str}")
                last_score = score
                last_lives = lives

            game_over = seen_alive and lives == 0
            if game_over or done:
                break
            obs = stack.push(preprocess(raw, args.obs_size, color))

        print(f"=== Episode {ep + 1}: ended at step {step_no}, "
              f"final score {info['score']}, lives {info['lives']} ===")
        total = max(1, sum(action_counts))
        dist_str = "  ".join(
            f"{n}={c}({100 * c / total:.0f}%)"
            for n, c in zip(ACTION_NAMES, action_counts))
        print(f"  actions ({total} decisions): {dist_str}")
        print(f"  avg entropy: {ent_sum / max(1, n_decisions):.2f}  "
              f"avg V: {v_sum / max(1, n_decisions):+.1f}")
        print(f"  reward / life events: {len(events)}")
        for e in events:
            print(e)
        print()

    env.close()


if __name__ == "__main__":
    main()
