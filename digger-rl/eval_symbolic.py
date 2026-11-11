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
from tools.symbolic_env import env_step_skipped
from train_ppo import select_device
from train_ppo_symbolic import IterAgent, SymbolicAgent


def add_prev(x: torch.Tensor, cfg: dict, prev: int) -> torch.Tensor:
    """Append the previous action as 6 one-hot planes for checkpoints
    trained with tools/train_bc_traces.py --prev-action."""
    if not cfg.get("prev_action"):
        return x
    oh = torch.zeros(1, 6, *x.shape[2:], dtype=x.dtype, device=x.device)
    oh[:, prev] = 1.0
    return torch.cat([x, oh], dim=1)


def build_agent(ckpt_path: Path, device) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    cfg = ckpt["config"]
    env_kwargs = dict(frame_stack=cfg["frame_stack"],
                      egocentric=cfg.get("egocentric", False))
    probe = SymbolicDiggerEnv(**env_kwargs)
    in_ch, h, w = probe.obs_shape
    in_ch += 6 * bool(cfg.get("prev_action"))
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
    p.add_argument("--policy", choices=("ckpt", "random", "teacher", "search", "pixel"),
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
    p.add_argument("--search-horizon", type=int, default=24,
                   help="--policy search: rollout length in agent steps")
    p.add_argument("--search-commit", type=int, default=3,
                   help="--policy search: steps the candidate is held")
    p.add_argument("--search-rollouts", type=int, default=2,
                   help="--policy search: rollouts per candidate")
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
    pixel = None
    if args.policy == "pixel":
        # Pixel student (train_ppo.Agent). Same score/life/level-1 metrics
        # as the symbolic policies; frames come from the same emulator.
        import collections
        from digger_env import preprocess_uint8
        from tools.gen_search_trace import _load_student
        pixel, pcfg = _load_student(args.checkpoint)
        pixel.to(device)
        pstack = collections.deque(maxlen=pcfg["frame_stack"])

        def pixel_frame():
            raw = env._env._core.get_frame()
            if "obs_hw" in pcfg:   # native-aspect model (train_bc_traces)
                from tools.gen_search_trace import _resize_rgb
                return _resize_rgb(raw, *pcfg["obs_hw"])
            return preprocess_uint8(raw, pcfg["obs_size"], pcfg["color"])
    if args.policy == "search":
        from tools.search_teacher import SearchTeacher
        teacher = SearchTeacher(env, horizon=args.search_horizon,
                                commit=args.search_commit,
                                rollouts=args.search_rollouts)
    else:
        teacher = SmartHeuristic() if args.policy == "teacher" else None

    label = args.label or (args.policy if args.policy != "ckpt"
                           else str(args.checkpoint))
    game_scores: list[int] = []
    life_lengths: list[int] = []
    # The project goal: clear level 1 without losing a life. A level-up
    # shows as the emerald grid refilling (a jump of >= 4; CV occlusion
    # only flickers the count by 1-2) -- same rule as DeathBagLogger.
    clean_l1 = 0
    any_l1 = 0

    for ep in range(args.episodes):
        obs = env.reset()
        if teacher is not None:
            teacher.reset()
        steps_this_life = 0
        score = 0
        died_yet = False
        levelled = False
        clean_game = False
        prev_em = None
        refill = True   # (re)fill the pixel stack, as DiggerVecEnv.reset does
        prev_a = 0      # previous executed action (prev_action checkpoints)
        while True:
            if pixel is not None:
                fr = pixel_frame()
                if refill:
                    pstack.clear()
                    pstack.extend([fr] * pstack.maxlen)
                    refill = False
                else:
                    pstack.append(fr)
                x = np.concatenate([f.transpose(2, 0, 1) for f in pstack], 0)
                with torch.no_grad():
                    xt = torch.from_numpy(x).to(device).float().div_(255.0)
                    xt = add_prev(xt.unsqueeze(0), pcfg, prev_a)
                    logits = pixel.actor(pixel.encode(xt))
                    a = int(logits.argmax(-1).item() if args.argmax else
                            torch.distributions.Categorical(
                                logits=logits).sample().item())
            elif args.policy == "random":
                a = int(rng.integers(env.NUM_ACTIONS))
            elif teacher is not None:
                a = int(teacher(env._last_state))
            else:
                with torch.no_grad():
                    x = torch.from_numpy(obs).to(device).unsqueeze(0)
                    x = add_prev(x, cfg, prev_a)
                    logits = agent.actor(agent.encode(x))
                    a = int(logits.argmax(-1).item() if args.argmax else
                            torch.distributions.Categorical(
                                logits=logits).sample().item())
            obs, _, done, info = env_step_skipped(env, a, args.frame_skip)
            prev_a = a
            steps_this_life += 1
            score = int(info.get("score", score))
            # Lives are segmented at death onset (RAM death stage), so a
            # life's length excludes the ~100 input-ignored steps of the
            # death sequence. Lengths measured before 2026-10 counted
            # lives-drop to lives-drop and included them.
            n_em = int(env._last_state.emeralds.sum())
            if prev_em is not None and n_em - prev_em >= 4 and not levelled:
                levelled = True
                any_l1 += 1
                clean_game = not died_yet
                clean_l1 += clean_game
            prev_em = n_em
            if info.get("death_event", False):
                died_yet = True
                life_lengths.append(steps_this_life)
                steps_this_life = 0
            elif steps_this_life and info.get("dying", False):
                steps_this_life -= 1   # don't count dead time
            if info.get("dying", False):
                # The deployed env skips the death sequence and refills the
                # stack on reset; refill once the digger is back.
                refill = True
                prev_a = 0
            if done:
                break
        game_scores.append(score)
        print(f"  [{label}] game {ep + 1}/{args.episodes} "
              f"score={score} lives_recorded={len(life_lengths)} "
              f"L1={'clean' if clean_game else ('yes' if levelled else 'no')}",
              flush=True)
        if args.policy == "search":
            print(f"  [{label}]   {teacher.summary()}", flush=True)

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
    print(f"level 1    : cleared without dying {clean_l1}/{args.episodes}  "
          f"cleared at all {any_l1}/{args.episodes}")


if __name__ == "__main__":
    main()
