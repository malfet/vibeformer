"""Watch any policy play Digger in real time.

    python watch.py teacher
    python watch.py data/checkpoints/dagger3/ppo_sym_final.pt
    python watch.py data/checkpoints/RUN/ppo_digger_final.pt --argmax
    python watch.py search --overlay
    python watch.py teacher --dos-ckpt L5      # from level 5's start

The policy is a spec from tools/policies.py (teacher, search, random, or a
checkpoint from either trainer). Frames show inline in Kitty / Ghostty /
WezTerm, else in a matplotlib window (--display). Q quits, P pauses,
+/- change the speed. --sound plays the game's audio (needs
sounddevice).
"""

from __future__ import annotations

import argparse
import time

import torch

from digger_env import load_dos_ckpt
from tools.display import open_display
from tools.game_state import render_overlay
from tools.heuristic_agent import ACTION_NAMES
from tools.sound import open_sound
from tools.policies import env_kwargs, make_policy
from tools.symbolic_env import SymbolicDiggerEnv, env_step_skipped
from train_pixel import select_device

FPS = 70.087                                # DIGGER.EXE frame rate


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("policy", help="teacher | search | random | checkpoint path")
    p.add_argument("--games", type=int, default=1,
                   help="full 3-life games to play before exiting")
    p.add_argument("--argmax", action="store_true",
                   help="argmax instead of sampling (checkpoints only)")
    p.add_argument("--speed", type=float, default=1.0,
                   help="playback speed; 0 = as fast as possible")
    p.add_argument("--overlay", action="store_true",
                   help="mark the monsters and bags the policy's state has")
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--display", choices=("auto", "kitty", "mpl"),
                   default="auto")
    p.add_argument("--sound", action="store_true",
                   help="play the game's audio (needs sounddevice)")
    p.add_argument("--dos-ckpt", default=None, metavar="STATE",
                   help="start every game from this emulator state: a "
                        "state pickle, or L<n>[_<k>] for level n's start "
                        "in data/save/levels (tools/level_states.py)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--force-cpu", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    env = SymbolicDiggerEnv(max_steps=10**9, clip_reward=False,
                            episodic_life=False, **env_kwargs(args.policy))
    policy = make_policy(args.policy, env, select_device(args.force_cpu),
                         argmax=args.argmax, seed=args.seed)

    start = None
    if args.dos_ckpt is not None:
        start, desc = load_dos_ckpt(args.dos_ckpt)
        print(f"starting from {desc}", flush=True)

    def frame():
        raw = env._env._core.get_frame()
        return render_overlay(raw, env._last_state) if args.overlay else raw

    snd = None
    speed, paused, quit_ = args.speed, False, False
    results = []
    with open_display(args.display, f"DIGGER {args.policy}") as disp:
        for game in range(args.games):
            # reset() boots a fresh core; the old one must be released
            # first (one core per process), so drop the player's handle.
            if snd is not None:
                snd.close()
                snd = None
            obs = env.reset()
            if args.sound:
                snd = open_sound(env._env._core)
            if start is not None:
                obs = env.load_state({"env": start})
            policy.reset()
            step, score, a = 0, 0, 0
            while disp.is_open and not quit_:
                for key, pressed in disp.poll():
                    if not pressed:
                        continue
                    if key == "q":
                        quit_ = True
                    elif key == "p":
                        paused = not paused
                    elif key in ("+", "="):
                        speed = speed * 2 if speed else 0
                    elif key in ("-", "_"):
                        speed = speed / 2 if speed else 4.0
                if quit_:
                    break
                t0 = time.monotonic()
                if paused:
                    time.sleep(0.05)
                    if snd is not None:
                        snd.flush()
                else:
                    a = policy(obs)
                    obs, _, done, info = env_step_skipped(
                        env, a, args.frame_skip)
                    policy.observe(info)
                    if snd is not None:
                        snd.feed()
                    step += 1
                    score = int(info.get("score", score))
                    if done:
                        break
                disp.show(frame())
                disp.set_title(
                    f"game {game + 1}/{args.games} -- score {score:>6d} -- "
                    f"lives {env._last_state.lives} -- "
                    f"a={ACTION_NAMES[a]:<5s} -- step {step} -- "
                    f"x{speed:g}{' -- PAUSED' if paused else ''}")
                if speed:
                    slack = args.frame_skip / FPS / speed \
                        - (time.monotonic() - t0)
                    if slack > 0:
                        time.sleep(slack)
            if quit_ or not disp.is_open:
                break
            results.append(score)
    if snd is not None:
        snd.close()
        snd = None
    env.close()
    for i, s in enumerate(results):
        print(f"game {i + 1}: score {s}")


if __name__ == "__main__":
    main()
