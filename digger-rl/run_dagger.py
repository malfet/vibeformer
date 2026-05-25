"""Watch a trained DAGGER policy play Digger in a matplotlib window.

    python run_dagger.py data/checkpoints/dagger_v1/dagger_final.pt
    python run_dagger.py data/checkpoints/dagger_v1/dagger_iter05.pt --stochastic

The checkpoint format is what `train_dagger.py` writes:
    {"policy": <state_dict>, "iter": int, "config": dict}

The policy is the symbolic-state PolicyNet -- it consumes the 6-channel
(dirt / emerald / digger / monster / bag / cherry) mask tensor produced
by tools.symbolic_env.state_to_tensor, not raw pixels.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from digger_env import DiggerEnv, _env_step_skipped
from tools.game_state import extract_state_fast, render_overlay
from tools.symbolic_env import state_to_tensor
from train_dagger import PolicyNet
from train_ppo import select_device

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
                   help="path to a .pt file produced by train_dagger.py")
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--stochastic", action="store_true",
                   help="sample from the policy instead of taking argmax")
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--overlay", action="store_true",
                   help="draw markers on detected monsters / bags / emeralds")
    p.add_argument("--show-digger", action="store_true",
                   help="(with --overlay) mark the inferred digger tile too")
    p.add_argument("--show-emeralds", action="store_true",
                   help="(with --overlay) mark every detected emerald")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.force_cpu)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    policy = PolicyNet().to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    iter_no = ckpt.get("iter", "?")
    print(f"loaded {args.checkpoint} (DAGGER iter {iter_no}, "
          f"{'stochastic' if args.stochastic else 'argmax'} policy)",
          flush=True)

    import matplotlib.pyplot as plt

    env = DiggerEnv()
    raw = env.reset()
    state = extract_state_fast(raw)

    target_fps = 70.087
    target_dt = (1.0 / target_fps) * args.frame_skip

    fig, ax = plt.subplots()
    ax.set_axis_off()
    overlay = render_overlay(raw, state,
                             show_digger=args.show_digger,
                             show_emeralds=args.show_emeralds) \
        if args.overlay else raw
    img = ax.imshow(overlay)
    fig.tight_layout(pad=0)
    fig.canvas.manager.set_window_title("DIGGER DAGGER -- gameplay")

    plt.ion()
    plt.show()

    seen_alive = False
    episodes_done = 0
    last_score = 0
    fps_ema = 1.0 / target_dt
    step_no = 0
    last_wall = time.monotonic()

    while plt.fignum_exists(fig.number) and episodes_done < args.episodes:
        now = time.monotonic()
        elapsed = now - last_wall
        last_wall = now

        obs = state_to_tensor(state)
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = policy(obs_t)
            if args.stochastic:
                a = Categorical(logits=logits).sample()
            else:
                a = logits.argmax(-1)
        action = int(a[0].item())

        raw, reward, done, info = _env_step_skipped(env, action, args.frame_skip)
        step_no += args.frame_skip
        state = extract_state_fast(raw)

        if args.overlay:
            img.set_data(render_overlay(raw, state,
                                         show_digger=args.show_digger,
                                         show_emeralds=args.show_emeralds))
        else:
            img.set_data(raw)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        lives = info["lives"]
        if lives > 0:
            seen_alive = True
        game_over = seen_alive and lives == 0

        if elapsed > 0:
            fps_ema = 0.9 * fps_ema + 0.1 * (args.frame_skip / elapsed)
        fig.canvas.manager.set_window_title(
            f"DIGGER DAGGER -- score {info['score']:>6d} -- lives {lives} -- "
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
            state = extract_state_fast(raw)
            seen_alive = False
            step_no = 0

        slack = target_dt - (time.monotonic() - now)
        if slack > 0:
            time.sleep(slack)

    plt.close("all")
    env.close()


if __name__ == "__main__":
    main()
