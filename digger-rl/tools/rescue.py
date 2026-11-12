"""Rescue traces: how to survive the deaths a student actually runs into.

The student plays (from tools/level_states.py starts) while we keep a
save state for each of the last steps. When it starts to die, we roll
back `--rollback` steps (then further, if nothing works) and try
branches: hold one move for `--hold` steps, then let the teacher play,
plus the teacher alone. A branch that is still alive `--horizon` steps
later is a rescue; the best-scoring one is kept, with the frames and
actions it actually played, and play continues from its end so the
next death comes from further into the level.

Branches are recorded as they run, never replayed: repeated restores of
one state do not replay identically (input timing around unserialize),
so a found action sequence cannot be trusted to work twice.

Each rescue becomes a segment of a train_pixel.py trace: the 3 frames
before the rollback point (real history, so the first rescue step has
a full frame stack) and then the branch. Oversample these traces in
--bc-traces to weight them.

    python -m tools.rescue --student data/checkpoints/RUN/ppo_digger_bc_only.pt \\
        --out data/traces/px128_rescue_0.npz --deaths 200 --seed 0

Also reports what killed the student (bag or monster, from the game's
own deathbag field) and how often a rescue was found.
"""

from __future__ import annotations

import argparse
import collections
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from digger_env import DiggerEnv, preprocess_uint8
from tools.heuristic_agent import SmartHeuristic
from tools.policies import make_policy
from tools.symbolic_env import SymbolicDiggerEnv, env_step_skipped

MOVES = (DiggerEnv.NOOP, DiggerEnv.LEFT, DiggerEnv.RIGHT,
         DiggerEnv.UP, DiggerEnv.DOWN)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--student", required=True, help="pixel checkpoint")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--deaths", type=int, default=200,
                   help="stop after this many student deaths")
    p.add_argument("--rollback", type=int, nargs="+", default=[10, 15, 20],
                   help="steps to roll back before death onset, tried in "
                        "order until a branch survives")
    p.add_argument("--hold", type=int, default=3,
                   help="steps a branch holds its first move")
    p.add_argument("--horizon", type=int, default=20,
                   help="a rescue is alive this many steps after rollback")
    p.add_argument("--start-states", default="data/save/levels")
    p.add_argument("--start-prob", type=float, default=0.75)
    p.add_argument("--pool-dir", type=Path, default=None,
                   help="also save the --rollback states before each death "
                        "as start states (level_states.py format), for "
                        "PPO episodes that start just before a death")
    p.add_argument("--pool-rescued-only", action="store_true",
                   help="save a --pool-dir state only when a branch from it "
                        "survived (states a policy can learn to escape)")
    p.add_argument("--pool-only", action="store_true",
                   help="only save --pool-dir states; skip the branch search")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if (args.pool_only or args.pool_rescued_only) and args.pool_dir is None:
        p.error("--pool-only / --pool-rescued-only need --pool-dir")
    if args.pool_only and args.pool_rescued_only:
        p.error("--pool-rescued-only needs the branch search (no --pool-only)")
    if args.pool_dir is not None:
        args.pool_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    env = SymbolicDiggerEnv(max_steps=10**9, clip_reward=False,
                            episodic_life=False,
                            start_states=args.start_states,
                            start_prob=args.start_prob)
    student = make_policy(args.student, env, torch.device("cpu"),
                          seed=args.seed)
    size = student.cfg["obs_size"]
    teacher = SmartHeuristic()
    keep = max(args.rollback) + 4

    def frame() -> np.ndarray:
        return preprocess_uint8(env._env._core.get_frame(), size, True)

    out = collections.defaultdict(list)       # trace arrays

    def emit(frames, actions, scores, lives) -> None:
        n = len(frames)
        out["frames"] += frames
        out["actions"] += actions
        out["scores"] += scores
        out["lives"] += lives
        out["resets"] += [True] + [False] * (n - 1)

    def run_branch(state, first: int | None):
        """Play from `state`: `first` held for --hold steps (None = teacher
        throughout), then the teacher. Returns the recording plus the end
        state, or None if the digger dies within --horizon steps."""
        env.load_state(state)
        teacher.reset()
        lives0 = env._last_state.lives
        fr, ac, sc, lv = [], [], [], []
        for t in range(args.horizon):
            a = (first if first is not None and t < args.hold
                 else int(teacher(env._last_state)))
            fr.append(frame())
            ac.append(a)
            lv.append(env._last_state.lives)
            _, _, done, info = env_step_skipped(env, a, 4)
            sc.append(env._env._read_score())
            if done or info.get("dying", False) \
                    or env._last_state.lives < lives0:
                return None
        return fr, ac, sc, lv, env.save_state()

    stats = collections.Counter()
    t0 = time.monotonic()
    env.reset()
    student.reset()
    history = collections.deque(maxlen=keep)   # (state, frame, a, score, lives)
    obs = env.current_obs()
    while stats["deaths"] < args.deaths:
        st = env.save_state()
        f = frame()
        a = student(obs)
        history.append((st, f, a, env._env._read_score(),
                        env._last_state.lives))
        obs, _, done, info = env_step_skipped(env, a, 4)
        student.observe(info)
        if done:                                  # game over / timeout
            env.reset()
            student.reset()
            history.clear()
            obs = env.current_obs()
            continue
        if not info.get("dying", False):
            continue

        stats["deaths"] += 1
        cause = info.get("death_cause") or "monster"
        stats[f"death_{cause}"] += 1
        level = env._env.read_level()

        def save_pool(back: int) -> None:
            h = history[len(history) - back]
            name = (f"L{level}_nd{args.seed}_{stats['deaths']}"
                    f"_{cause}_b{back}.pkl")
            with open(args.pool_dir / name, "wb") as fh:
                pickle.dump(dict(state=h[0], level=level, step=-back,
                                 lives=h[4], score=h[3], cause=cause), fh)
            stats["pool_states"] += 1

        if args.pool_dir is not None and not args.pool_rescued_only:
            for back in args.rollback:
                if back <= len(history):
                    save_pool(back)
        if args.pool_only:
            env.reset()
            student.reset()
            history.clear()
            obs = env.current_obs()
            continue
        best = None
        for back in args.rollback:
            if back + 3 > len(history):
                break
            i = len(history) - back
            state = history[i][0]
            for first in (None,) + MOVES:
                rec = run_branch(state, first)
                if rec is not None and (best is None
                                        or rec[2][-1] > best[1][2][-1]):
                    best = (i, rec)
            if best is not None:
                stats[f"rescued_at_{back}"] += 1
                stats[f"rescued_{cause}"] += 1
                if args.pool_rescued_only:
                    save_pool(back)
                break
        if best is None:
            stats["unrescued"] += 1
            env.reset()                           # new game
        else:
            i, (fr, ac, sc, lv, end) = best
            pre = [history[j] for j in range(i - 3, i)]
            emit([h[1] for h in pre] + fr, [h[2] for h in pre] + ac,
                 [h[3] for h in pre] + sc, [h[4] for h in pre] + lv)
            # Play on from the end of the rescue, so the next death comes
            # from further into the level.
            env.load_state(end)
        student.reset()
        history.clear()
        obs = env.current_obs()
        if stats["deaths"] % 10 == 0:
            el = time.monotonic() - t0
            print(f"  deaths {stats['deaths']}  {dict(stats)}  "
                  f"segments {sum(out['resets'])}  {el:.0f}s", flush=True)

    env.close()
    if not out["frames"]:
        print(f"\n{dict(stats)}\nno rescue segments written")
        return
    frames = np.stack(out["frames"])
    scores = np.array(out["scores"], np.int32)
    prev = np.concatenate([[scores[0]], scores[:-1]])
    resets = np.array(out["resets"], bool)
    raw = np.where(resets, 0, scores - prev).astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, frames=frames, actions=np.array(out["actions"], np.int64),
        raw_rewards=raw, scores=scores,
        lives=np.array(out["lives"], np.int32), resets=resets,
        frame_skip=np.int32(4), obs_size=np.int32(size),
        color=np.bool_(True), teacher=np.array("rescue", dtype=object))
    print(f"\n{dict(stats)}\nwrote {len(frames)} steps "
          f"({int(resets.sum())} rescues) to {args.out}")


if __name__ == "__main__":
    main()
