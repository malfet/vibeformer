"""Collect emulator save-states from a few steps *before* each death.

Motivation. The survival-reward runs (see README) failed for a credit-
assignment reason: paid `+c` per step, every action looks identical and
the only discriminating signal is the truncated tail at death, hundreds
of steps after the mistake that caused it. Starting an episode 3-5 agent
steps before a real death inverts that — the agent is dropped straight
into the dangerous state, and whether it survives the next few steps is
decided by what it does *now*.

Mechanically: drive the game with some policy, keep a ring buffer of the
last `lookback+1` save-states (one per agent step, ~1 ms and 678 KB
each), and when a death *starts* (RAM death stage leaves "alive") write
out the state from `lookback` steps earlier. Repeat.

Two caveats worth knowing before trusting the output:

  - Restores are gameplay-valid but not byte-exact (README lesson 12), so
    a restored scenario does not replay the same death deterministically.
    For a curriculum that is fine — arguably the point.
  - Some deaths are simply not escapable that late (a bag already falling
    onto the digger's head). `--probe` measures what fraction of the
    collected scenarios the *teacher* survives, which is the ceiling any
    student trained on them could reach.

    python -m tools.collect_death_states \\
        --out data/scenarios/death_v64_200.pkl --scenarios 200 --lookback 4
    python -m tools.collect_death_states \\
        --out data/scenarios/death_v64_200.pkl --probe --probe-steps 40
"""

from __future__ import annotations

import argparse
import collections
import pickle
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from tools.heuristic_agent import SmartHeuristic
from tools.symbolic_env import SymbolicDiggerEnv
from train_dagger import env_step_skipped


def _policy_fn(name: str, checkpoint: Path | None, env, rng):
    """Return a callable(obs) -> action for the chosen driver policy."""
    if name == "random":
        return lambda obs: int(rng.integers(env.NUM_ACTIONS))
    if name == "teacher":
        teacher = SmartHeuristic()
        fn = lambda obs: int(teacher(env._last_state))       # noqa: E731
        fn.reset = teacher.reset
        return fn
    import torch
    from eval_symbolic import build_agent
    agent, _ = build_agent(checkpoint, "cpu")

    def act(obs):
        with torch.no_grad():
            x = torch.from_numpy(obs).unsqueeze(0)
            logits = agent.actor(agent.encode(x))
            return int(torch.distributions.Categorical(
                logits=logits).sample().item())
    return act


def collect(out: Path, n_deaths: int, lookbacks: list[int], policy: str,
            checkpoint: Path | None, frame_skip: int, frame_stack: int,
            min_lives: int, seed: int) -> None:
    """Capture every requested lookback from the same ring buffer.

    Collection is dominated by the ~10 s it takes the driver policy to
    die, so pulling 3 / 5 / 10 / 20-steps-before out of one death costs
    nothing extra and lets `--probe` measure how escapability decays with
    lookback — which is the number that decides whether this curriculum
    is learnable at all.
    """
    lookback = max(lookbacks)
    rng = np.random.default_rng(seed)
    env = SymbolicDiggerEnv(max_steps=10**9, clip_reward=False,
                            episodic_life=True, frame_stack=frame_stack)
    act_fn = _policy_fn(policy, checkpoint, env, rng)

    obs = env.reset()
    if hasattr(act_fn, "reset"):
        act_fn.reset()
    # Ring of (state, meta) for the last lookback+1 agent steps. The
    # oldest entry is exactly `lookback` steps behind the newest.
    ring: collections.deque = collections.deque(maxlen=lookback + 1)
    scenarios: list[dict] = []
    deaths = 0
    deaths_kept = 0
    skipped_lives = 0
    t0 = time.monotonic()

    while deaths_kept < n_deaths:
        st = env.save_state()
        state = env._last_state
        ring.append({
            "state": st,
            "lives": int(env._env._read_lives()),
            "digger": (None if state.digger is None or not state.digger.present
                       else (int(state.digger.row), int(state.digger.col))),
            "monster_min_dist": _min_monster_dist(state),
        })

        a = act_fn(obs)
        obs, _, done, info = env_step_skipped(env, a, frame_skip)

        # Anchor on death *onset* (RAM death stage leaving "alive"), not on
        # the `lives` decrement: the death sequence runs ~100 agent steps
        # with input ignored, so lives-anchored lookbacks of up to 60 all
        # landed inside it (README lesson 12b). Then let the sequence play
        # out without snapshotting until the life is actually gone.
        onset = info.get("dying", False)
        if onset:
            while not done:
                obs, _, done, info = env_step_skipped(env, 0, frame_skip)

        if onset or done:
            deaths += 1
            if len(ring) == ring.maxlen:
                # ring[-1] is the state at the step that started the
                # death, so the snapshot k steps earlier is ring[-1 - k].
                kept = False
                for k in lookbacks:
                    cand = ring[-1 - k]
                    # A scenario with one life left ends the *game* on
                    # death, which forces a slow DOSBox reboot on every
                    # reset during training. Keep only spare-life states.
                    if cand["lives"] < min_lives:
                        skipped_lives += 1
                        continue
                    # cand["state"] is already a SymbolicDiggerEnv snapshot
                    # ({"env": <DiggerEnv dict>, "prev_dist": ...}), which
                    # is exactly what load_state() wants. Wrapping it again
                    # would bury the "core" key one level too deep.
                    sc = dict(cand["state"])
                    sc["meta"] = {
                        "lookback": k,
                        "lives": cand["lives"],
                        "digger": cand["digger"],
                        "monster_min_dist": cand["monster_min_dist"],
                        "death_score": int(info.get("score", 0)),
                        "policy": policy,
                    }
                    scenarios.append(sc)
                    kept = True
                if kept:
                    deaths_kept += 1
            ring.clear()
            obs = env.reset()
            if hasattr(act_fn, "reset"):
                act_fn.reset()

            if deaths_kept and deaths_kept % 10 == 0:
                el = time.monotonic() - t0
                print(f"  {deaths_kept:>4d}/{n_deaths} deaths kept "
                      f"({len(scenarios)} scenarios, {deaths} deaths seen, "
                      f"{skipped_lives} skipped for lives<{min_lives})  "
                      f"{el:.0f}s", flush=True)

    env.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        pickle.dump({"scenarios": scenarios,
                     "lookback": lookbacks,
                     "policy": policy}, fh)
    mb = out.stat().st_size / 1e6
    print(f"\nWrote {len(scenarios)} scenarios from {deaths_kept} deaths "
          f"({mb:.1f} MB) to {out}")
    for k in lookbacks:
        dists = [s["meta"]["monster_min_dist"] for s in scenarios
                 if s["meta"]["lookback"] == k
                 and s["meta"]["monster_min_dist"] is not None]
        if dists:
            print(f"  lookback {k:>3d}: n={len(dists):>4d}  monster distance "
                  f"mean {statistics.mean(dists):.2f} "
                  f"median {statistics.median(dists):.0f} min {min(dists)}")


def _min_monster_dist(state) -> int | None:
    if state.digger is None or not state.digger.present or not state.monsters:
        return None
    return min(abs(m.row - state.digger.row) + abs(m.col - state.digger.col)
               for m in state.monsters)


def probe(path: Path, policy: str, checkpoint: Path | None, probe_steps: int,
          frame_skip: int, frame_stack: int, seed: int,
          limit: int = 0) -> None:
    """What fraction of the collected scenarios can `policy` survive?

    This is the escapability ceiling. If the teacher itself dies from
    most of them within `probe_steps`, the states were captured too late
    to be learnable and `--lookback` should go up.
    """
    with path.open("rb") as fh:
        blob = pickle.load(fh)
    scenarios = blob["scenarios"]
    rng = np.random.default_rng(seed)
    # The probe window has to outlast the lookback, otherwise the death
    # the state was captured from happens *after* the window closes and
    # every scenario scores a meaningless 100% "survived".
    max_lb = max(sc["meta"]["lookback"] for sc in scenarios)
    if probe_steps <= max_lb:
        raise SystemExit(
            f"--probe-steps {probe_steps} must exceed the largest lookback "
            f"in {path.name} ({max_lb}); otherwise the original death falls "
            f"outside the probe window and the survival rate is vacuous. "
            f"Try --probe-steps {max_lb + 30}.")
    if limit and limit < len(scenarios):
        # Stratify so every lookback keeps roughly equal representation;
        # a plain head-slice would bias toward whichever came first.
        by_k: dict[int, list[dict]] = {}
        for sc in scenarios:
            by_k.setdefault(sc["meta"]["lookback"], []).append(sc)
        per = max(1, limit // len(by_k))
        scenarios = [sc for k in sorted(by_k) for sc in by_k[k][:per]]
        print(f"probing a {len(scenarios)}-scenario stratified subset "
              f"({per} per lookback)", flush=True)
    env = SymbolicDiggerEnv(max_steps=10**9, clip_reward=False,
                            episodic_life=True, frame_stack=frame_stack)
    env.reset()
    act_fn = _policy_fn(policy, checkpoint, env, rng)

    by_lookback: dict[int, list[int | None]] = {}
    for i, sc in enumerate(scenarios):
        obs = env.load_state(sc)
        if hasattr(act_fn, "reset"):
            act_fn.reset()
        outcome: int | None = None      # steps to death, or None = survived
        for t in range(probe_steps):
            a = act_fn(obs)
            obs, _, done, info = env_step_skipped(env, a, frame_skip)
            # Death onset, not the lives decrement ~100 steps later.
            if done or info.get("dying", False):
                outcome = t + 1
                break
        by_lookback.setdefault(sc["meta"]["lookback"], []).append(outcome)
        if (i + 1) % 25 == 0:
            done_n = sum(len(v) for v in by_lookback.values())
            alive = sum(1 for v in by_lookback.values() for o in v if o is None)
            print(f"  probed {done_n}/{len(scenarios)}  survived {alive}",
                  flush=True)
    env.close()

    print(f"\n=== probe: {policy} on {path.name} "
          f"({len(scenarios)} scenarios, {probe_steps} agent steps) ===")
    print(f"{'lookback':>9s} {'n':>5s} {'survived':>9s} {'rate':>6s}  "
          f"{'steps-to-death (when it died)':s}")
    for k in sorted(by_lookback):
        v = by_lookback[k]
        alive = sum(1 for o in v if o is None)
        died = [o for o in v if o is not None]
        det = (f"mean {statistics.mean(died):.1f} "
               f"median {statistics.median(died):.0f}") if died else "-"
        print(f"{k:>9d} {len(v):>5d} {alive:>9d} {alive / len(v):>5.0%}  {det}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--deaths", type=int, default=150,
                   help="how many deaths to harvest; each yields one "
                        "scenario per --lookbacks entry")
    p.add_argument("--lookbacks", type=str, default="3,5,10,20",
                   help="comma-separated agent-step distances before the "
                        "death to snapshot. All come from one ring buffer, "
                        "so extra entries are free.")
    p.add_argument("--policy", choices=("teacher", "random", "ckpt"),
                   default="teacher",
                   help="who drives the game while collecting / probing")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--min-lives", type=int, default=2,
                   help="skip captures with fewer lives left than this, so "
                        "training resets don't hit a full DOSBox reboot")
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probe", action="store_true",
                   help="don't collect; measure survivability of --out")
    p.add_argument("--probe-steps", type=int, default=40)
    p.add_argument("--probe-limit", type=int, default=0,
                   help="probe only this many scenarios, stratified across "
                        "lookbacks (0 = all)")
    a = p.parse_args()

    if a.probe:
        probe(a.out, a.policy, a.checkpoint, a.probe_steps,
              a.frame_skip, a.frame_stack, a.seed, a.probe_limit)
    else:
        lookbacks = sorted({int(x) for x in a.lookbacks.split(",") if x})
        collect(a.out, a.deaths, lookbacks, a.policy, a.checkpoint,
                a.frame_skip, a.frame_stack, a.min_lives, a.seed)


if __name__ == "__main__":
    main()
