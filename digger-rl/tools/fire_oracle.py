"""Would FIRE kill a monster right now? Ground truth from the emulator.

The bullet stops at any non-black pixel (digger.c updatefire), which a
tile-level line-of-sight check cannot see, so this asks the game itself:
from a save state, play FIRE then NOOPs, and only NOOPs, for `horizon`
agent steps; the shot "would kill" if the first kills more monsters.

The probes run in a separate process. Restores are not timing-exact
(each shifts the game's tick phase ~1 frame, README Lesson 20), so the
played game must never be restored, and only one core fits a process.

    oracle = FireOracle()
    if state.digger.can_fire and state.monsters:
        kill = oracle(env._env.save_state())
    oracle.close()

As a script, measures how a policy uses FIRE:

    python -m tools.fire_oracle teacher --games 8
"""

from __future__ import annotations

import argparse
import collections
import multiprocessing as mp

import torch

FIRE, NOOP = 5, 0


def _worker(conn, horizon: int, frame_skip: int) -> None:
    from tools.symbolic_env import SymbolicDiggerEnv, env_step_skipped
    env = SymbolicDiggerEnv(max_steps=10**9, clip_reward=False,
                            episodic_life=False)
    env.reset()

    def kills(state: dict, first: int) -> int:
        env._env.load_state(state, refresh=False)
        env._last_state = env._read_state()
        prev, k = len(env._last_state.monsters), 0
        for i in range(horizon):
            _, _, done, info = env_step_skipped(
                env, first if i == 0 else NOOP, frame_skip)
            cur = len(env._last_state.monsters)
            k += max(0, prev - cur)
            prev = cur
            if done or info.get("dying", False):
                break
        return k

    while (state := conn.recv()) is not None:
        conn.send(kills(state, FIRE) > kills(state, NOOP))


class FireOracle:
    def __init__(self, horizon: int = 10, frame_skip: int = 4):
        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_worker, daemon=True,
                                 args=(child, horizon, frame_skip))
        self._proc.start()

    def __call__(self, digger_env_state: dict) -> bool:
        """`digger_env_state` is DiggerEnv.save_state() of the played game."""
        self._conn.send(digger_env_state)
        return self._conn.recv()

    def close(self) -> None:
        self._conn.send(None)
        self._proc.join()


def main() -> None:
    from tools.policies import env_kwargs, make_policy
    from tools.symbolic_env import SymbolicDiggerEnv, env_step_skipped
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("policy", help="teacher | checkpoint")
    p.add_argument("--games", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    oracle = FireOracle()
    env = SymbolicDiggerEnv(max_steps=10**9, clip_reward=False,
                            episodic_life=False, **env_kwargs(args.policy))
    policy = make_policy(args.policy, env, torch.device("cpu"),
                         seed=args.seed)
    n = collections.Counter()
    for g in range(args.games):
        obs = env.reset()
        policy.reset()
        while True:
            s = env._last_state
            kill = None
            if s.digger.can_fire and s.monsters and s.digger.alive:
                kill = oracle(env._env.save_state())
            a = policy(obs)
            if kill is not None:
                n["ready"] += 1
                n["would_kill"] += kill
                n["fired"] += a == FIRE
                n["fired_killed"] += kill and a == FIRE
                n["missed"] += kill and a != FIRE
            obs, _, done, info = env_step_skipped(env, a, 4)
            policy.observe(info)
            if done:
                break
        print(f"game {g + 1}: score {info.get('score')} {dict(n)}",
              flush=True)
    oracle.close()
    env.close()
    print(f"\nfire-ready steps {n['ready']}, would kill {n['would_kill']}, "
          f"fired {n['fired']} (killed {n['fired_killed']}), "
          f"missed kills {n['missed']}")


if __name__ == "__main__":
    main()
