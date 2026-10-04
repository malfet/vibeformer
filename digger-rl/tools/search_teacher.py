"""Lookahead search teacher: SmartHeuristic, overruled by emulator rollouts.

Same-process save/restore is exact enough to use the emulator as its own
model (README 12c), and RAM gives a frame-exact death flag (12d). So at
each decision:

  1. Ask SmartHeuristic for its action a_h.
  2. Snapshot, then play a_h for `commit` agent steps followed by the
     heuristic for the rest of `horizon`, `rollouts` times. Restores are
     not byte-exact, so the rollouts naturally sample DOSBox's timing
     noise. If none of them dies, take a_h: this gate keeps the common
     case at `rollouts` rollouts per step.
  3. Otherwise roll out every action the same way and take the one with
     the best (survival rate, steps survived, score gained), preferring
     a_h on ties.

Rollouts step the raw DiggerEnv and run the CV extractor only when the
heuristic has to decide (every `decide_every` agent steps), since the
extractor costs ~3x an emulator frame.

    python eval_symbolic.py --policy search --episodes 10
"""

from __future__ import annotations

import copy
import time

from tools.game_state import extract_state_fast
from tools.heuristic_agent import SmartHeuristic
from tools.symbolic_env import SymbolicDiggerEnv


class SearchTeacher:
    def __init__(self, env: SymbolicDiggerEnv, horizon: int = 24,
                 commit: int = 3, rollouts: int = 2, decide_every: int = 2,
                 frame_skip: int = 4, gate: bool = True,
                 verbose: bool = True):
        if commit < 1:
            # The first rollout steps run on a stale framebuffer (no-refresh
            # restore); a committed action needs no observation.
            raise ValueError("commit must be >= 1")
        self.env = env
        self.horizon = horizon
        self.commit = commit
        self.rollouts = rollouts
        self.decide_every = decide_every
        self.frame_skip = frame_skip
        self.gate = gate
        self.verbose = verbose
        self.base = SmartHeuristic()
        self.stats = {"steps": 0, "searched": 0, "overruled": 0,
                      "rollouts": 0, "seconds": 0.0}

    def reset(self) -> None:
        self.base.reset()

    def _rollout(self, snap, first: int) -> tuple[bool, int, int]:
        """(died, agent steps survived, score gained) for one rollout."""
        self.stats["rollouts"] += 1
        self.env.restore_snapshot(snap)
        raw = self.env._env
        score0 = raw._last_score
        h = copy.deepcopy(self.base)
        a = first
        info = {}
        for t in range(self.horizon):
            if t >= self.commit and (t - self.commit) % self.decide_every == 0:
                a = int(h(extract_state_fast(raw._core.get_frame())))
            for _ in range(self.frame_skip):
                s = raw.step(a)
                info = s.info
                if info["death_event"] or info["dying"]:
                    return True, t, info["score"] - score0
                if s.done:   # truncation / game over without a death
                    return False, t, info["score"] - score0
        return False, self.horizon, info.get("score", score0) - score0

    def _evaluate(self, snap, a: int) -> tuple[float, float, float]:
        res = [self._rollout(snap, a) for _ in range(self.rollouts)]
        n = len(res)
        return (sum(not d for d, _, _ in res) / n,
                sum(t for _, t, _ in res) / n,
                sum(g for _, _, g in res) / n)

    def __call__(self, state) -> int:
        t0 = time.perf_counter()
        a_h = int(self.base(state))
        snap = self.env.snapshot()
        best = a_h
        v_h = self._evaluate(snap, a_h)
        self.stats["steps"] += 1
        if not (self.gate and v_h[0] == 1.0):
            self.stats["searched"] += 1
            best_v = v_h
            for a in range(self.env.NUM_ACTIONS):
                if a == a_h:
                    continue
                v = self._evaluate(snap, a)
                if v > best_v:
                    best, best_v = a, v
            self.stats["overruled"] += best != a_h
        self.env.restore_snapshot(snap)
        self.stats["seconds"] += time.perf_counter() - t0
        if self.verbose and self.stats["steps"] % 500 == 0:
            print(f"    {self.summary()}", flush=True)
        return best

    def summary(self) -> str:
        s = self.stats
        n = max(s["steps"], 1)
        return (f"search: {s['steps']} steps, searched {s['searched'] / n:.1%}, "
                f"overruled {s['overruled'] / n:.1%}, "
                f"{s['seconds'] / n * 1e3:.0f} ms/step")
