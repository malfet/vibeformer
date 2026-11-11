"""Watch a trained PPO checkpoint play Digger (tools/display.py viewer).

    python run_ppo.py data/checkpoints/ppo_digger_final.pt

Default policy mode is argmax (deterministic). Pass --stochastic to sample
from the policy's Categorical -- often more interesting to watch on an
under-trained checkpoint where argmax collapses to one action.
"""

from __future__ import annotations

import argparse
import time
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
    """Float32 84x84 in [0,1]; grayscale by default, RGB if color=True."""
    return preprocess_uint8(rgba, size, color).astype(np.float32) * (1.0 / 255.0)

_ACTION_NAMES = {
    DiggerEnv.NOOP:  "noop ",
    DiggerEnv.LEFT:  "left ",
    DiggerEnv.RIGHT: "right",
    DiggerEnv.UP:    "up   ",
    DiggerEnv.DOWN:  "down ",
    DiggerEnv.FIRE:  "fire ",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path,
                   help="path to a .pt file produced by train_ppo.py")
    p.add_argument("--force-cpu", action="store_true",
                   help="force CPU even if a CUDA/MPS accelerator is available")
    p.add_argument("--stochastic", action="store_true",
                   help="sample from the policy instead of taking argmax")
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--obs-size", type=int, default=84)
    p.add_argument("--episodes", type=int, default=1,
                   help="number of game-over events before exiting")
    p.add_argument("--show-stack", action="store_true",
                   help="show the policy's frame stack next to the game "
                        "(past 3 frames merged into one RGB image: R=t-3, G=t-2, B=t-1)")
    p.add_argument("--display", choices=("auto", "kitty", "mpl"),
                   default="auto",
                   help="viewer: Kitty graphics in the terminal, a matplotlib "
                        "window, or auto-detect (default)")
    return p.parse_args()


def main():
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
    print(f"loaded checkpoint trained for {step_trained} emulator frames "
          f"({'stochastic' if args.stochastic else 'argmax'} policy)",
          flush=True)

    from tools.display import open_display

    env = DiggerEnv()
    stack = FrameStack(k=args.frame_stack, size=args.obs_size, color=color)

    raw = env.reset()
    obs = stack.reset(preprocess(raw, args.obs_size, color))

    target_fps = 70.087
    target_dt = (1.0 / target_fps) * args.frame_skip  # real-time pacing

    # Optional: the policy's frame stack collapsed into a single RGB image
    # (R=t-3, G=t-2, B=t-1) shown to the right of the gameplay. This makes
    # motion legible at a glance -- stationary pixels appear gray, moving
    # sprites leave colored trails as they shift between channels.
    def frame(raw):
        if not args.show_stack:
            return raw
        st = np.stack([stack.frames[0], stack.frames[1], stack.frames[2]],
                      axis=-1)
        st = (st * 255 if st.dtype != np.uint8 else st).astype(np.uint8)
        k = raw.shape[0] // st.shape[0]
        st = st.repeat(k, 0).repeat(k, 1)
        pad = np.zeros((raw.shape[0], st.shape[1], 3), np.uint8)
        pad[:st.shape[0]] = st[..., :3]
        return np.concatenate([raw[..., :3], pad], axis=1)

    disp = open_display(args.display, "DIGGER PPO")
    disp.show(frame(raw))

    seen_alive = False
    episodes_done = 0
    last_score = 0
    fps_ema = 1.0 / target_dt
    step_no = 0
    last_wall = time.monotonic()

    while disp.is_open and episodes_done < args.episodes:
        disp.poll()                     # drain keys; Ctrl+C closes Kitty
        now = time.monotonic()
        elapsed = now - last_wall
        last_wall = now

        # Policy forward (batch size 1, MPS is overkill but fine)
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits = agent.actor(agent.encode(obs_t))
            if args.stochastic:
                a = Categorical(logits=logits).sample()
            else:
                a = logits.argmax(-1)
        action = int(a[0].item())

        raw, reward, done, info = env_step_skipped(env, action, args.frame_skip)
        step_no += args.frame_skip

        disp.show(frame(raw))

        lives = info["lives"]
        if lives > 0:
            seen_alive = True
        game_over = seen_alive and lives == 0

        if elapsed > 0:
            fps_ema = 0.9 * fps_ema + 0.1 * (args.frame_skip / elapsed)
        disp.set_title(
            f"score {info['score']:>6d} -- lives {lives} -- "
            f"a={_ACTION_NAMES[action]} -- {fps_ema:5.1f} fps -- step {step_no}")
        last_score = info["score"]

        if game_over or done:
            print(f"episode {episodes_done + 1} ended: score {last_score}, "
                  f"steps {step_no}", flush=True)
            episodes_done += 1
            if episodes_done >= args.episodes:
                break
            raw = env.reset()
            obs = stack.reset(preprocess(raw, args.obs_size, color))
            seen_alive = False
            step_no = 0
        else:
            obs = stack.push(preprocess(raw, args.obs_size, color))

        # Pace to real-time so it's watchable
        slack = target_dt - (time.monotonic() - now)
        if slack > 0:
            time.sleep(slack)

    disp.close()
    env.close()


if __name__ == "__main__":
    main()
