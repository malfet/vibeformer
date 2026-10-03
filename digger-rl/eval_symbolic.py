"""Evaluate a symbolic-obs policy over N full games.

`eval_checkpoint.py` only knows how to rebuild the *pixel* NatureCNN
agent. This is its symbolic counterpart, and it reports the two metrics
the survival experiments care about, which are not the same metric:

  - **game score**: total score across a full 3-life game. The number
    every earlier scoreboard row is quoting.
  - **life length**: agent steps between deaths. The thing a
    survival-reward policy is actually being trained to maximise, and
    the number to compare against the v6 teacher's 280.

Runs with episodic_life=False so one episode is one whole game; lives
are segmented at each death onset (info["death_event"]).

    python eval_symbolic.py data/checkpoints/RUN/ppo_sym_best.pt --episodes 20
    python eval_symbolic.py --policy random  --episodes 20
    python eval_symbolic.py --policy teacher --episodes 20
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import numpy as np
import torch

from tools.heuristic_agent import SmartHeuristic
from tools.symbolic_env import SymbolicDiggerEnv
from train_dagger import env_step_skipped
from train_ppo import select_device
from train_ppo_symbolic import IterAgent, SymbolicAgent


def build_agent(ckpt_path: Path, device) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    cfg = ckpt["config"]
    env_kwargs = dict(frame_stack=cfg["frame_stack"],
                      egocentric=cfg.get("egocentric", False))
    probe = SymbolicDiggerEnv(**env_kwargs)
    in_ch, h, w = probe.obs_shape
    num_actions = probe.NUM_ACTIONS
    if cfg.get("arch", "cnn") == "iter":
        agent = IterAgent(in_channels=in_ch, num_actions=num_actions,
                          h=h, w=w, ch=cfg["iter_channels"],
                          iters=cfg["iter_steps"], readout=cfg["readout"])
    else:
        agent = SymbolicAgent(in_channels=in_ch, num_actions=num_actions,
                              h=h, w=w)
    agent.load_state_dict(ckpt["agent"])
    return agent.to(device).eval(), cfg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path, nargs="?")
    p.add_argument("--policy", choices=("ckpt", "random", "teacher"),
                   default="ckpt",
                   help="random / teacher give same-session baselines; "
                        "cross-session numbers are not comparable (see "
                        "the drift caveat in README).")
    p.add_argument("--episodes", type=int, default=20,
                   help="full 3-life games")
    p.add_argument("--argmax", action="store_true",
                   help="argmax instead of Categorical sampling")
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--frame-stack", type=int, default=4,
                   help="only used by --policy random/teacher")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--label", type=str, default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = select_device(args.force_cpu)

    agent = None
    egocentric = False
    frame_stack = args.frame_stack
    if args.policy == "ckpt":
        if args.checkpoint is None:
            raise SystemExit("--policy ckpt needs a checkpoint path")
        agent, cfg = build_agent(args.checkpoint, device)
        egocentric = cfg.get("egocentric", False)
        frame_stack = cfg["frame_stack"]

    # episodic_life=False: one env episode == one full game, so per-life
    # segmentation comes from watching lives, not from done.
    env = SymbolicDiggerEnv(max_steps=10**9, clip_reward=False,
                            episodic_life=False, frame_stack=frame_stack,
                            egocentric=egocentric)
    teacher = SmartHeuristic() if args.policy == "teacher" else None

    label = args.label or (args.policy if args.policy != "ckpt"
                           else str(args.checkpoint))
    game_scores: list[int] = []
    life_lengths: list[int] = []

    for ep in range(args.episodes):
        obs = env.reset()
        if teacher is not None:
            teacher.reset()
        steps_this_life = 0
        score = 0
        while True:
            if args.policy == "random":
                a = int(rng.integers(env.NUM_ACTIONS))
            elif teacher is not None:
                a = int(teacher(env._last_state))
            else:
                with torch.no_grad():
                    x = torch.from_numpy(obs).to(device).unsqueeze(0)
                    logits = agent.actor(agent.encode(x))
                    a = int(logits.argmax(-1).item() if args.argmax else
                            torch.distributions.Categorical(
                                logits=logits).sample().item())
            obs, _, done, info = env_step_skipped(env, a, args.frame_skip)
            steps_this_life += 1
            score = int(info.get("score", score))
            # Lives are segmented at death onset (RAM death stage), so a
            # life's length excludes the ~100 input-ignored steps of the
            # death sequence. Lengths measured before 2026-10 counted
            # lives-drop to lives-drop and included them.
            if info.get("death_event", False):
                life_lengths.append(steps_this_life)
                steps_this_life = 0
            elif steps_this_life and info.get("dying", False):
                steps_this_life -= 1   # don't count dead time
            if done:
                break
        game_scores.append(score)
        print(f"  [{label}] game {ep + 1}/{args.episodes} "
              f"score={score} lives_recorded={len(life_lengths)}", flush=True)

    env.close()

    def sem(xs):
        return statistics.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else 0.0

    print(f"\n=== {label} ({args.episodes} games"
          f"{', argmax' if args.argmax else ', stochastic'}) ===")
    print(f"game score : mean {statistics.mean(game_scores):7.1f} "
          f"+-{sem(game_scores):5.1f}  median {statistics.median(game_scores):6.0f}  "
          f"max {max(game_scores)}")
    if life_lengths:
        print(f"life length: mean {statistics.mean(life_lengths):7.1f} "
              f"+-{sem(life_lengths):5.1f}  median "
              f"{statistics.median(life_lengths):6.0f}  "
              f"max {max(life_lengths)}  n={len(life_lengths)} agent steps")


if __name__ == "__main__":
    main()
