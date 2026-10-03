"""Render the figures used in the experiment report.

Every panel is a real 640x400 DOSBox frame or a real observation tensor
pushed through the real transform — nothing here is drawn by hand.

    python -m tools.make_report_figs --out docs/

Figures:
  play_strips.png       teacher / BC clone / survival-only policy, four
                        frames each at the same step offsets. The point
                        is the third row: the survival-trained policy
                        does not move.
  restore_divergence.png  one restored save-state stepped under LEFT and
                        under RIGHT. Proves input is live after restore
                        (the finding that rescued the curriculum work).
  obs_alloc_vs_ego.png  the same board as the allocentric (10x15) tensor
                        the old net saw and the egocentric (19x29)
                        canvas the new one sees.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tools.heuristic_agent import SmartHeuristic
from tools.symbolic_env import (BASE_OBS_CHANNELS, SymbolicDiggerEnv,
                                egocentric_view, state_to_tensor)
from train_dagger import env_step_skipped

REPO = Path(__file__).resolve().parent.parent
CKPT = REPO / "data" / "checkpoints"


def rgb(frame: np.ndarray) -> np.ndarray:
    """Frame -> RGB for imshow.

    The libretro core hands back RGBA in that order, not BGRA: the most
    common pixel is literally [130, 65, 0, 255], which is PAL_DIRT_A.
    Reversing the channels here paints the dirt blue and the gold bags
    teal, so don't.
    """
    return frame[..., :3]


def _ckpt_policy(path: Path):
    from eval_symbolic import build_agent
    agent, cfg = build_agent(path, "cpu")

    def act(env, obs):
        with torch.no_grad():
            logits = agent.actor(agent.encode(
                torch.from_numpy(obs).unsqueeze(0)))
            return int(torch.distributions.Categorical(
                logits=logits).sample().item())
    return act, cfg.get("egocentric", False)


def collect_strip(policy: str, ckpt: Path | None, offsets: list[int],
                  warmup: int, seed: int):
    """Play `policy` and grab the raw frame at each offset after warmup."""
    ego = False
    if policy == "teacher":
        teach = SmartHeuristic()
        act = lambda env, obs: int(teach(env._last_state))       # noqa: E731
    else:
        act, ego = _ckpt_policy(ckpt)
        teach = None
    env = SymbolicDiggerEnv(max_steps=10**9, clip_reward=False,
                            episodic_life=True, frame_stack=4, egocentric=ego)
    obs = env.reset()
    if teach is not None:
        teach.reset()
    frames, caps = [], []
    want = set(offsets)
    for t in range(warmup + max(offsets) + 1):
        a = act(env, obs)
        obs, _, done, info = env_step_skipped(env, a, 4)
        if done:
            obs = env.reset()
            if teach is not None:
                teach.reset()
        step = t - warmup
        if step in want:
            frames.append(rgb(env._env._core.get_frame()))
            d = env._last_state.digger
            caps.append(f"step {step}  score {info.get('score', 0)}  "
                        f"digger {'-' if d is None or not d.present else (d.row, d.col)}")
    env.close()
    return frames, caps


def fig_play_strips(out: Path, offsets: list[int]) -> None:
    rows = [
        ("SmartHeuristic v6.4 (teacher)", "teacher", None),
        ("Symbolic BC clone (1315 score)", "ckpt",
         CKPT / "ppo_sym_bc_v64" / "ppo_sym_final.pt"),
        ("Survival-only PPO (NOOP collapse)", "ckpt",
         CKPT / "ppo_sym_surv_iter" / "ppo_sym_best.pt"),
    ]
    fig, axes = plt.subplots(len(rows), len(offsets),
                             figsize=(4.0 * len(offsets), 2.9 * len(rows)))
    for r, (label, pol, ck) in enumerate(rows):
        frames, caps = collect_strip(pol, ck, offsets, warmup=60, seed=0)
        for c in range(len(offsets)):
            ax = axes[r, c]
            ax.imshow(frames[c])
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(caps[c], fontsize=8)
            if c == 0:
                ax.set_ylabel(label, fontsize=10, wrap=True)
    fig.suptitle("Same step offsets, three policies. Bottom row barely moves: "
                 "survival-only reward selects for standing still.",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_live_vs_sealed(out: Path, scenarios: Path, steps: int = 4) -> None:
    """The corpse detector: a live restore vs a death-sequence restore.

    Top pair is a state saved during ordinary play, replayed under LEFT
    and under RIGHT -- the trajectories diverge, so restore is
    input-live. Bottom pair is a state captured a few agent steps before
    the `lives` counter dropped, replayed the same two ways -- the
    frames come back pixel-identical, because the outcome is already
    sealed and input is ignored until the respawn.

    This is Lesson 12b in one picture, and the reason a save-state
    curriculum has to be probed before it is trained on.
    """
    env = SymbolicDiggerEnv(max_steps=10**9, clip_reward=False,
                            episodic_life=True, frame_stack=4)
    env.reset()
    teach = SmartHeuristic(); teach.reset()
    for _ in range(60):
        env_step_skipped(env, int(teach(env._last_state)), 4)
    live = env.save_state()

    blob = pickle.load(scenarios.open("rb"))
    sealed = min(blob["scenarios"], key=lambda s: s["meta"]["lookback"])

    def replay(state, act):
        env.load_state(state)
        fr, pos = [], []
        for _ in range(steps):
            _, _, done, _ = env_step_skipped(env, act, 4)
            fr.append(rgb(env._env._core.get_frame()))
            d = env._last_state.digger
            pos.append(None if d is None or not d.present else (d.row, d.col))
            if done:
                break
        return fr, pos

    panels, ident = [], {}
    for tag, st in (("live play", live),
                    (f"{sealed['meta']['lookback']} steps pre-death", sealed)):
        runs = {n: replay(st, a) for a, n in ((1, "LEFT"), (2, "RIGHT"))}
        k = min(len(runs["LEFT"][0]), len(runs["RIGHT"][0]))
        ident[tag] = all(np.array_equal(runs["LEFT"][0][i],
                                        runs["RIGHT"][0][i]) for i in range(k))
        for name in ("LEFT", "RIGHT"):
            panels.append((f"{tag}\nrestore + {name}", runs[name]))
    env.close()

    cols = max(len(p[1][0]) for p in panels)
    fig, axes = plt.subplots(len(panels), cols,
                             figsize=(4.0 * cols, 2.9 * len(panels)))
    for r, (label, (fr, pos)) in enumerate(panels):
        for c in range(cols):
            ax = axes[r, c]
            if c < len(fr):
                ax.imshow(fr[c])
                ax.set_title(f"after step {c + 1}  digger "
                             f"{pos[c] if pos[c] else '-'}", fontsize=8)
            else:
                ax.axis("off")
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=9)
    verdicts = "     ".join(
        f"{k}: LEFT vs RIGHT {'IDENTICAL - sealed' if v else 'DIVERGE - live'}"
        for k, v in ident.items())
    fig.suptitle("One save-state, two action sequences.\n" + verdicts,
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({verdicts})")


def fig_restore_divergence(out: Path, scenarios: Path, steps: int) -> None:
    """Replay one restored state under LEFT and under RIGHT."""
    blob = pickle.load(scenarios.open("rb"))
    env = SymbolicDiggerEnv(max_steps=10**9, clip_reward=False,
                            episodic_life=True, frame_stack=4)
    env.reset()

    # Pick a scenario where the two actions actually produce different
    # digger positions -- an unescapable (frozen) state would make the
    # figure say the opposite of what it should.
    # Frames are captured *after* each step, never straight out of
    # load_state: restoring twice back-to-back reproduces the frame
    # exactly, but restoring into an env that has since been stepped
    # does not (score and lives come back correct, the pixels don't).
    # The first post-restore frame is therefore not trustworthy.
    chosen, traces = None, None
    for sc in blob["scenarios"]:
        runs, alive = {}, True
        for act, name in ((1, "LEFT"), (2, "RIGHT")):
            env.load_state(sc)
            fr, pos = [], []
            for t in range(steps):
                _, _, done, _ = env_step_skipped(env, act, 4)
                fr.append(rgb(env._env._core.get_frame()))
                d = env._last_state.digger
                pos.append(None if d is None or not d.present
                           else (d.row, d.col))
                if done:
                    alive = False
                    break
            runs[name] = (fr, pos)
        # Require both runs to survive the whole window and to actually
        # end up somewhere different -- a run that died early "differs"
        # for the wrong reason, and an unescapable state would make the
        # figure claim the opposite of what it should.
        if alive and runs["LEFT"][1] != runs["RIGHT"][1] \
                and all(p is not None for p in runs["LEFT"][1]) \
                and all(p is not None for p in runs["RIGHT"][1]):
            chosen, traces = sc, runs
            break
    env.close()
    if chosen is None:
        print("no scenario diverged under LEFT vs RIGHT -- skipping figure")
        return

    n = min(len(traces["LEFT"][0]), len(traces["RIGHT"][0]), 4)
    fig, axes = plt.subplots(2, n, figsize=(4.0 * n, 6.0))
    for r, name in enumerate(("LEFT", "RIGHT")):
        fr, pos = traces[name]
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(fr[c])
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"after step {c+1}  digger {pos[c] if pos[c] else '-'}",
                         fontsize=8)
            if c == 0:
                ax.set_ylabel(f"restore + {name}", fontsize=11)
    fig.suptitle("One save-state, two action sequences. The trajectories "
                 "diverge, so restore is input-live -- the test that "
                 "separates a hard scenario from a corpse.", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_obs(out: Path) -> None:
    """Allocentric tensor vs the egocentric canvas, same real board."""
    env = SymbolicDiggerEnv(max_steps=10**9, clip_reward=False,
                            episodic_life=True, frame_stack=1)
    env.reset()
    teach = SmartHeuristic(); teach.reset()
    for _ in range(70):
        env_step_skipped(env, int(teach(env._last_state)), 4)
    state = env._last_state
    frame = rgb(env._env._core.get_frame())
    board = state_to_tensor(state)
    d = state.digger
    rc = (d.row, d.col) if d is not None and d.present else (5, 7)
    ego = egocentric_view(board, *rc)
    env.close()

    names = ["dirt", "emerald", "digger", "monster", "bag", "cherry"]
    # One flattened picture per view: channel index painted per cell.
    def paint(t, ch):
        img = np.zeros(t.shape[1:] + (3,))
        colors = [(0.35, 0.22, 0.10), (0.10, 0.85, 0.35), (1.0, 1.0, 0.2),
                  (0.95, 0.25, 0.25), (0.85, 0.7, 0.1), (0.9, 0.3, 0.7)]
        for i in range(min(ch, t.shape[0])):
            m = t[i] > 0.5
            for k in range(3):
                img[..., k][m] = colors[i][k]
        return img

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    axes[0].imshow(frame); axes[0].set_title("raw 640x400 frame", fontsize=11)
    axes[1].imshow(paint(board, BASE_OBS_CHANNELS), interpolation="nearest")
    axes[1].set_title(f"allocentric 10x15 -- digger at {rc}\n"
                      "flatten->FC gives every cell private weights",
                      fontsize=10)
    axes[2].imshow(paint(ego, BASE_OBS_CHANNELS), interpolation="nearest")
    axes[2].set_title("egocentric 19x29 -- digger pinned at centre\n"
                      "translation invariance by construction", fontsize=10)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    legend = "  ".join(f"{n}" for n in names)
    fig.suptitle(f"Observation transforms (channels: {legend})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=REPO / "docs")
    p.add_argument("--offsets", type=str, default="0,25,50,75")
    p.add_argument("--scenarios", type=Path,
                   default=REPO / "data/scenarios/death_bcclone_far.pkl")
    p.add_argument("--only", type=str, default="all",
                   choices=("all", "strips", "restore", "obs"))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    offsets = [int(x) for x in a.offsets.split(",")]
    if a.only in ("all", "obs"):
        fig_obs(a.out / "obs_alloc_vs_ego.png")
    if a.only in ("all", "strips"):
        fig_play_strips(a.out / "play_strips.png", offsets)
    if a.only in ("all", "restore") and a.scenarios.exists():
        fig_live_vs_sealed(a.out / "restore_live_vs_sealed.png",
                           a.scenarios, steps=4)


if __name__ == "__main__":
    main()
