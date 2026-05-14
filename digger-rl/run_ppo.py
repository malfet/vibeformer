"""Watch a trained PPO checkpoint play Digger in a matplotlib window.

    python run_ppo.py data/checkpoints/ppo_digger_final.pt

Default policy mode is argmax (deterministic). Pass --stochastic to sample
from the policy's Categorical -- often more interesting to watch on an
under-trained checkpoint where argmax collapses to one action.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.distributions import Categorical

from digger_env import DiggerEnv
from train_ppo import Agent, FrameStack, env_step_skipped, preprocess, select_device

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
    return p.parse_args()


def main():
    args = parse_args()
    device = select_device(args.force_cpu)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    agent = Agent(num_actions=DiggerEnv.NUM_ACTIONS,
                  in_channels=args.frame_stack).to(device)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()
    step_trained = ckpt.get("step", "?")
    print(f"loaded checkpoint trained for {step_trained} emulator frames "
          f"({'stochastic' if args.stochastic else 'argmax'} policy)",
          flush=True)

    import matplotlib.pyplot as plt

    env = DiggerEnv()
    stack = FrameStack(k=args.frame_stack, size=args.obs_size)

    raw = env.reset()
    obs = stack.reset(preprocess(raw, args.obs_size))

    target_fps = 70.087
    target_dt = (1.0 / target_fps) * args.frame_skip  # real-time pacing

    fig, ax = plt.subplots()
    ax.set_axis_off()
    img = ax.imshow(raw)
    fig.tight_layout(pad=0)
    fig.canvas.manager.set_window_title("DIGGER PPO -- gameplay")

    # Second window: what the policy network actually sees. Four 84x84
    # grayscale frames stacked along the channel axis, oldest on the left.
    fig2, axes2 = plt.subplots(1, args.frame_stack, figsize=(2 * args.frame_stack, 2.3))
    if args.frame_stack == 1:
        axes2 = [axes2]
    imgs2 = []
    for i, ax_i in enumerate(axes2):
        ax_i.set_axis_off()
        ax_i.set_title(f"t-{args.frame_stack - 1 - i}", fontsize=9)
        imgs2.append(ax_i.imshow(obs[i], cmap="gray", vmin=0.0, vmax=1.0))
    fig2.tight_layout(pad=0.3)
    fig2.canvas.manager.set_window_title("PPO input -- 4x grayscale frame stack")

    plt.ion()
    plt.show()

    seen_alive = False
    episodes_done = 0
    last_score = 0
    fps_ema = 1.0 / target_dt
    step_no = 0
    last_wall = time.monotonic()

    while (plt.fignum_exists(fig.number)
           and plt.fignum_exists(fig2.number)
           and episodes_done < args.episodes):
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

        img.set_data(raw)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        # Mirror the policy's input stack into the secondary window.
        for i, frame in enumerate(stack.frames):
            imgs2[i].set_data(frame)
        fig2.canvas.draw_idle()
        fig2.canvas.flush_events()

        lives = info["lives"]
        if lives > 0:
            seen_alive = True
        game_over = seen_alive and lives == 0

        if elapsed > 0:
            fps_ema = 0.9 * fps_ema + 0.1 * (args.frame_skip / elapsed)
        fig.canvas.manager.set_window_title(
            f"DIGGER PPO -- score {info['score']:>6d} -- lives {lives} -- "
            f"a={_ACTION_NAMES[action]} -- {fps_ema:5.1f} fps -- step {step_no}"
        )
        last_score = info["score"]

        if game_over or done:
            print(f"episode {episodes_done + 1} ended: score {last_score}, "
                  f"steps {step_no}", flush=True)
            episodes_done += 1
            if episodes_done >= args.episodes:
                break
            raw = env.reset()
            obs = stack.reset(preprocess(raw, args.obs_size))
            seen_alive = False
            step_no = 0
        else:
            obs = stack.push(preprocess(raw, args.obs_size))

        # Pace to real-time so it's watchable
        slack = target_dt - (time.monotonic() - now)
        if slack > 0:
            time.sleep(slack)

    plt.close("all")
    env.close()


if __name__ == "__main__":
    main()
