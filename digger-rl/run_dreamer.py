"""Watch a Dreamer agent play Digger in real time.

Loads a checkpoint produced by train_dreamer.py (world_model + actor +
critic), runs a real DiggerEnv, threads each real observation through the
RSSM via observe_step (so the recurrent state stays grounded in reality
rather than imagined), and renders gameplay in a matplotlib window.

    python run_dreamer.py data/checkpoints/dreamer_wm.pt

With --show-recon, a second window shows the world model's 64x64 input
alongside the decoder's reconstruction -- useful to see whether the model
is "looking at" the same thing the human can see.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from digger_env import DiggerEnv, preprocess_uint8
from dreamer import Actor, Critic, DreamerConfig, WorldModel
from train_ppo import select_device

ACTION_NAMES = ["NOOP", "LEFT", "RIGHT", "UP", "DOWN", "FIRE"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--stochastic", action="store_true",
                   help="sample from the actor instead of argmax")
    p.add_argument("--frame-skip", type=int, default=4,
                   help="emulator frames per policy step (matches training)")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--show-recon", action="store_true",
                   help="open a second window: WM input vs decoded reconstruction")
    return p.parse_args()


def step_env_with_skip(env, action, skip):
    """Apply `action` for `skip` emulator frames; return last raw obs + sum reward."""
    total_r = 0.0
    info: dict = {}
    raw = None
    done = False
    for _ in range(skip):
        s = env.step(action)
        total_r += s.reward
        raw = s.obs
        info = s.info
        if s.done:
            done = True
            break
    return raw, total_r, done, info


def main() -> None:
    args = parse_args()
    device = select_device(args.force_cpu)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = DreamerConfig(**ckpt["config"])
    wm = WorldModel(cfg).to(device).eval()
    actor = Actor(cfg).to(device).eval()
    critic = Critic(cfg).to(device).eval()
    wm.load_state_dict(ckpt["world_model"])
    actor.load_state_dict(ckpt["actor"])
    critic.load_state_dict(ckpt["critic"])
    print(f"loaded {args.checkpoint}")
    print(f"  mode={'stochastic' if args.stochastic else 'argmax'}  "
          f"obs_size={cfg.obs_size}  frame_skip={args.frame_skip}")

    import matplotlib.pyplot as plt

    env = DiggerEnv()
    raw = env.reset()

    target_fps = 70.087
    target_dt = (1.0 / target_fps) * args.frame_skip

    fig, ax = plt.subplots()
    ax.set_axis_off()
    img = ax.imshow(raw)
    fig.tight_layout(pad=0)
    fig.canvas.manager.set_window_title("DIGGER Dreamer -- gameplay")

    fig2 = ax2_in = ax2_rec = None
    if args.show_recon:
        fig2, (ax2_in, ax2_rec) = plt.subplots(1, 2, figsize=(8, 4))
        ax2_in.set_axis_off(); ax2_in.set_title("WM input")
        ax2_rec.set_axis_off(); ax2_rec.set_title("decoded")
        fig2.tight_layout(pad=0.5)
        fig2.canvas.manager.set_window_title("Dreamer world-model view")

    plt.ion()
    plt.show()

    # RSSM state lives across env steps; reset on episode boundary.
    h, z = wm.rssm.initial(batch_size=1, device=device)
    prev_action = torch.zeros(1, dtype=torch.long, device=device)

    seen_alive = False
    episodes_done = 0
    fps_ema = 1.0 / target_dt
    step_no = 0
    last_wall = time.monotonic()

    while plt.fignum_exists(fig.number) and episodes_done < args.episodes:
        now = time.monotonic()
        elapsed = now - last_wall
        last_wall = now

        # Preprocess the current real frame for the world model.
        obs_u8 = preprocess_uint8(raw, cfg.obs_size, color=True)  # (64, 64, 3)
        obs_t = (torch.from_numpy(obs_u8).permute(2, 0, 1).unsqueeze(0)
                 .to(device).float().mul_(1.0 / 255.0))             # (1, 3, 64, 64)

        with torch.no_grad():
            embed = wm.encoder(obs_t)                                # (1, embed_dim)
            prev_a_oh = F.one_hot(prev_action,
                                  num_classes=cfg.num_actions).float()
            h, z, _, _ = wm.rssm.observe_step(h, z, prev_a_oh, embed)
            feat = wm.rssm.feature(h, z)
            dist = actor(feat)
            a = dist.sample() if args.stochastic else dist.probs.argmax(-1)
            action = int(a[0].item())
            entropy = float(dist.entropy()[0].item())
            value = float(critic(feat)[0].item())

            if args.show_recon and fig2 is not None and plt.fignum_exists(fig2.number):
                recon = wm.decoder(feat).clamp(0, 1)                 # (1, 3, 64, 64)
                recon_np = recon[0].permute(1, 2, 0).cpu().numpy()

        raw, reward, done, info = step_env_with_skip(env, action, args.frame_skip)
        step_no += args.frame_skip
        prev_action = a

        img.set_data(raw)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        if args.show_recon and fig2 is not None and plt.fignum_exists(fig2.number):
            ax2_in.images.clear() if ax2_in.images else None
            ax2_rec.images.clear() if ax2_rec.images else None
            ax2_in.imshow(obs_u8, vmin=0, vmax=255)
            ax2_rec.imshow(recon_np, vmin=0.0, vmax=1.0)
            fig2.canvas.draw_idle()
            fig2.canvas.flush_events()

        lives = info.get("lives", 0)
        score = info.get("score", 0)
        if lives > 0:
            seen_alive = True
        game_over = seen_alive and lives == 0

        if elapsed > 0:
            fps_ema = 0.9 * fps_ema + 0.1 * (args.frame_skip / elapsed)
        fig.canvas.manager.set_window_title(
            f"DIGGER Dreamer -- score {score:>6d} -- lives {lives} -- "
            f"a={ACTION_NAMES[action]:<5s} H={entropy:.2f} V={value:+6.1f} -- "
            f"{fps_ema:5.1f} fps -- step {step_no}"
        )

        if game_over or done:
            print(f"episode {episodes_done + 1} ended: score {score}, "
                  f"steps {step_no}")
            episodes_done += 1
            if episodes_done >= args.episodes:
                break
            raw = env.reset()
            seen_alive = False
            step_no = 0
            h, z = wm.rssm.initial(batch_size=1, device=device)
            prev_action = torch.zeros(1, dtype=torch.long, device=device)

        slack = target_dt - (time.monotonic() - now)
        if slack > 0:
            time.sleep(slack)

    plt.close("all")
    env.close()


if __name__ == "__main__":
    main()
