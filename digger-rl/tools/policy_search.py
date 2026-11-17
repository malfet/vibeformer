"""Search-improved pixel policy: expert iteration's teacher.

The pixel policy plays; whenever a monster is within `--risk` tiles,
the emulator checks its choice first: restore the current state and
play the policy's greedy action for `--commit` steps, then the greedy
policy for the rest of `--horizon`. If the digger is still alive, the
choice stands. Otherwise every action is rolled out the same way and
the best one by (survived, steps survived, score gained) is played and
recorded as the label, preferring the policy's own on ties. Restores
are exact with the lockstep core (README Lesson 21), so one rollout per
action is enough.

It also searches when the digger is stuck: `--stuck` steps on at most 2
tiles without scoring. Then every action is tried and ranked by
(survived, score gained, new tiles), each held 8 steps, so the label is the way out of the loop.

The rollouts reuse the policy's own frame stack, saved before and put
back after, so the played game and the policy's view of it carry on as
if no search had happened.

Writes a train_pixel.py trace (labels = the greedy action, or the
search's override), to use as --bc-traces next to PPO:

    python -m tools.policy_search checkpoints/digger_px128_bag1.pt \\
        --out data/traces/px128_search_0.npz --steps 20000 --seed 0
"""

from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path

import numpy as np
import torch

from digger_env import DiggerEnv, preprocess_uint8
from tools.policies import make_policy
from tools.symbolic_env import SymbolicDiggerEnv, env_step_skipped


class PolicySearch:
    def __init__(self, env: SymbolicDiggerEnv, policy, horizon: int = 30,
                 commit: int = 3, risk: int = 4, stuck_commit: int = 8):
        self.env, self.p = env, policy
        self.horizon, self.commit, self.risk = horizon, commit, risk
        self.stuck_commit = stuck_commit
        self.stats = collections.Counter()

    def _frame(self) -> np.ndarray:
        cfg = self.p.cfg
        return preprocess_uint8(self.env._env._core.get_frame(),
                                cfg["obs_size"], cfg["color"])

    @torch.no_grad()
    def _logits(self) -> torch.Tensor:
        planes = [f.transpose(2, 0, 1) if f.ndim == 3 else f[None]
                  for f in (self.p.stack[-1 - o] for o in self.p.offs)]
        x = torch.from_numpy(np.concatenate(planes, 0)).to(self.p.device)
        x = x.float().div_(255.0).unsqueeze(0)
        return self.p.agent.actor(self.p.agent.encode(x))[0]

    def observe(self, fresh: bool) -> torch.Tensor:
        """Push the current frame (refilling after a reset or death);
        returns the policy's logits."""
        fr = self._frame()
        if fresh or not self.p.stack:
            self.p.stack.clear()
            self.p.stack.extend([fr] * self.p.stack.maxlen)
        else:
            self.p.stack.append(fr)
        return self._logits()

    def risky(self) -> bool:
        s = self.env._last_state
        d = s.digger
        return d is not None and any(
            abs(m.col - d.col) + abs(m.row - d.row) <= self.risk
            for m in s.monsters)

    def _rollout(self, snap, first: int,
                 commit: int | None = None) -> tuple[bool, int, int, int]:
        """(survived, steps survived, score gained, tiles visited)."""
        self.stats["rollouts"] += 1
        self.env.restore_snapshot(snap)
        score0 = self.env._env._last_score
        commit = self.commit if commit is None else commit
        a, info, tiles = first, {}, set()
        for t in range(self.horizon):
            if t >= commit:
                a = int(self.observe(False).argmax())
            _, _, done, info = env_step_skipped(self.env, a, 4)
            d = self.env._last_state.digger
            if d is not None:
                tiles.add((d.col, d.row))
            if info.get("dying", False):
                return False, t, info.get("score", score0) - score0, len(tiles)
            if done:
                break
        return (True, self.horizon, info.get("score", score0) - score0,
                len(tiles))

    def choose(self, greedy: int, stuck: bool = False) -> int:
        """The action to play and record for the current state. When
        `stuck` (looping without scoring), every action is tried and
        ranked by (survived, score gained) instead of by time survived."""
        self.stats["steps"] += 1
        if not stuck and not self.risky():
            return greedy
        self.stats["stuck" if stuck else "checked"] += 1
        snap = self.env.snapshot()
        stack = list(self.p.stack)
        # stuck: hold the action longer (the greedy policy would walk
        # straight back into the loop) and rank by score, then new tiles
        commit = self.stuck_commit if stuck else None
        key = ((lambda v: (v[0], v[2], v[3])) if stuck
               else (lambda v: v[:3]))
        best, best_v = greedy, key(self._rollout(snap, greedy, commit))
        if stuck or not best_v[0]:
            self.stats["searched"] += 1
            for a in range(DiggerEnv.NUM_ACTIONS):
                if a == greedy:
                    continue
                self.p.stack.clear()
                self.p.stack.extend(stack)
                v = key(self._rollout(snap, a, commit))
                if v > best_v:
                    best, best_v = a, v
            self.stats["unstuck" if stuck else "overruled"] += best != greedy
            self.stats["rescued"] += best_v[0] and not stuck
        self.env.restore_snapshot(snap)
        self.p.stack.clear()
        self.p.stack.extend(stack)
        return best


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("policy", help="pixel checkpoint")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--commit", type=int, default=3)
    p.add_argument("--risk", type=int, default=4,
                   help="search only with a monster within this many tiles "
                        "(Manhattan)")
    p.add_argument("--stuck", type=int, default=20,
                   help="also search after this many steps on <= 2 tiles "
                        "without scoring (0 = off)")
    p.add_argument("--start-states", default="data/save/levels")
    p.add_argument("--start-prob", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    env = SymbolicDiggerEnv(max_steps=10**9, clip_reward=False,
                            episodic_life=False,
                            start_states=args.start_states,
                            start_prob=args.start_prob)
    policy = make_policy(args.policy, env, torch.device("cpu"),
                         seed=args.seed)
    search = PolicySearch(env, policy, args.horizon, args.commit, args.risk)
    out = collections.defaultdict(list)
    t0 = time.monotonic()
    env.reset()
    fresh = True
    deaths = 0
    recent: collections.deque = collections.deque(maxlen=max(args.stuck, 1))
    for k in range(args.steps):
        frame = search._frame()
        logits = search.observe(fresh)
        greedy = int(logits.argmax())
        d = env._last_state.digger
        if fresh or d is None:
            recent.clear()
        else:
            recent.append(((d.col, d.row), env._env._last_score))
        stuck = (args.stuck > 0 and len(recent) == args.stuck
                 and len({t for t, _ in recent}) <= 2
                 and recent[0][1] == recent[-1][1])
        a = search.choose(greedy, stuck)
        out["frames"].append(frame)
        out["actions"].append(a)
        out["lives"].append(env._last_state.lives)
        out["resets"].append(fresh)
        _, _, done, info = env_step_skipped(env, a, 4)
        out["scores"].append(env._env._read_score())
        fresh = info.get("dying", False)
        deaths += bool(info.get("death_event", False))
        if done:
            env.reset()
            fresh = True
        if (k + 1) % 500 == 0:
            st = search.stats
            print(f"  step {k + 1}: deaths {deaths}, checked "
                  f"{st['checked']}, searched {st['searched']}, overruled "
                  f"{st['overruled']} (rescued {st['rescued']}), stuck "
                  f"{st['stuck']} (unstuck {st['unstuck']}), "
                  f"{(time.monotonic() - t0) / (k + 1):.2f} s/step",
                  flush=True)
    env.close()

    scores = np.array(out["scores"], np.int32)
    resets = np.array(out["resets"], bool)
    prev = np.concatenate([[scores[0]], scores[:-1]])
    raw = np.where(resets, 0, scores - prev).astype(np.float32)
    cfg = policy.cfg
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, frames=np.stack(out["frames"]),
        actions=np.array(out["actions"], np.int64), raw_rewards=raw,
        scores=scores, lives=np.array(out["lives"], np.int32),
        resets=resets, frame_skip=np.int32(4),
        obs_size=np.int32(cfg["obs_size"]), color=np.bool_(cfg["color"]),
        teacher=np.array("policy_search", dtype=object))
    print(f"\n{dict(search.stats)}, deaths {deaths}\n"
          f"wrote {len(scores)} steps to {args.out}")


if __name__ == "__main__":
    main()
