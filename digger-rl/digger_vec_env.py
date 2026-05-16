"""Subprocess-based vectorised env for Digger.

LibretroCore is a per-process singleton: only one DOSBox can be alive in any
one Python interpreter. To run multiple Digger envs in parallel for PPO, each
env has to live in its own subprocess. This is the standard SubprocVecEnv
pattern (StableBaselines / CleanRL) adapted for our env protocol.

Architecture:

    main process                          N worker processes
    ─────────────────                     ─────────────────────────────
    SubprocVecEnv                         worker_main(child_conn, ...)
      ├── n×Pipe(parent_conn, child_conn) ├── DiggerEnv (LibretroCore)
      ├── n×subprocess.Process            ├── FrameStack (k stacks of 84×84)
      ├── reset()  → send "reset" to all  ├── while True:
      │              recv obs, stack      │     cmd, payload = conn.recv()
      ├── step(a) → send ("step", a_i)    │     dispatch reset/step/close
      │              recv (obs, r, d, i)  │     send result back
      └── close() → send "close"; join    └── on done: auto-reset, send fresh obs

Why subprocesses, not threads:
  - LibretroCore singleton check rejects a second instance in the same process.
  - DOSBox Pure uses globals and threads internally; reentrancy is unsafe.
  - macOS GIL/fork interactions favour `spawn` start method; `fork` can hang.

Throughput budget (rough, single-env baseline = 250 sps on MPS):
  - 8 workers theoretical max: 2000 sps. Realistic: ~1500 (IPC + GIL).
  - 1M frames at 1500 sps = ~11 min wall-clock (vs ~67 min serial).
  - Per-worker memory: ~250 MB (Python + DOSBox). 8 workers ~ 2 GB.
  - Per-step IPC: obs_stack uint8 (4×84×84) = 28 KB each direction × n_envs.
    At 250 step/s × 8 envs × 56 KB = ~110 MB/s. Pipes handle this fine.

Observation pipeline:
  - Worker side: env.step → raw RGBA 640×400 → preprocess() to 84×84
    grayscale uint8 → push into local FrameStack → ship k×84×84 uint8 stack.
  - Sending uint8 (not float32) cuts IPC bandwidth 4×.
  - Main batches uint8 stacks (N, k, 84, 84) → .to(device).float()/255 once
    per rollout step, amortising the H2D copy.

Auto-reset semantics (CleanRL-compatible):
  - When env.step returns done=True, the worker immediately calls env.reset
    and ships back the FIRST observation of the new episode along with the
    LAST reward/info from the dead episode. Main sees done=True, processes
    the bootstrap correctly, and obs is the new-episode start.

Status: PROTOTYPE. The classes below are functional but train_ppo.py hasn't
been wired to use them yet -- that requires reshaping the rollout buffer to
(num_steps, num_envs, ...) and flattening for minibatches.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from typing import Any

import numpy as np

# Imported lazily inside the worker -- train_ppo / digger_env pull in torch,
# which spawn-mode subprocesses re-import.


def _worker_main(child_conn, env_kwargs: dict, frame_skip: int,
                 obs_size: int, frame_stack: int, worker_id: int) -> None:
    """Entry point for each subprocess.

    Owns one DiggerEnv + one FrameStack. Receives commands over `child_conn`
    and ships back (obs, reward, done, info). Returns uint8 stacks to keep
    IPC bandwidth in check; main converts to float32 on the device.
    """
    # Local imports so the parent process doesn't pay these costs.
    from digger_env import DiggerEnv
    from train_ppo import FrameStack, env_step_skipped, preprocess

    env = DiggerEnv(**env_kwargs)
    stack = FrameStack(k=frame_stack, size=obs_size)

    def _preprocess_uint8(raw):
        # train_ppo.preprocess returns float32 [0,1]; we want uint8 in the
        # pipe to save 4x bandwidth.
        gray_f = preprocess(raw, obs_size)
        return (gray_f * 255.0).clip(0, 255).astype(np.uint8)

    def _reset_and_stack():
        raw = env.reset()
        return stack.reset(_preprocess_uint8(raw))

    obs = _reset_and_stack()
    try:
        while True:
            cmd, payload = child_conn.recv()
            if cmd == "reset":
                obs = _reset_and_stack()
                child_conn.send(obs)
            elif cmd == "step":
                action = int(payload)
                raw, reward, done, info = env_step_skipped(
                    env, action, frame_skip)
                if done:
                    # Auto-reset: ship back the FIRST obs of the new
                    # episode but the LAST reward/done/info from the
                    # finished one. Main treats done=True correctly for
                    # GAE bootstrap and obs is already the new start.
                    obs = _reset_and_stack()
                else:
                    obs = stack.push(_preprocess_uint8(raw))
                child_conn.send((obs, float(reward), bool(done), info))
            elif cmd == "close":
                break
            else:
                child_conn.send(("error", f"unknown cmd: {cmd}"))
    except KeyboardInterrupt:
        pass
    except Exception as exc:  # surface worker crashes so they don't deadlock the parent
        import traceback
        traceback.print_exc()
        try:
            child_conn.send(("error", repr(exc)))
        except Exception:
            pass
    finally:
        try:
            env.close()
        except Exception:
            pass
        child_conn.close()


class SubprocVecEnv:
    """N parallel DiggerEnv workers driven over multiprocessing Pipes.

    API mirrors gym-style vec envs without depending on gym:
        vec = SubprocVecEnv(num_envs=8, ...)
        obs = vec.reset()                  # (N, frame_stack, H, W) uint8
        for _ in range(steps):
            obs, rewards, dones, infos = vec.step(actions)  # actions: (N,) int
        vec.close()
    """

    def __init__(self, num_envs: int, env_kwargs: dict | None = None,
                 frame_skip: int = 4, obs_size: int = 84, frame_stack: int = 4,
                 ctx: str = "spawn") -> None:
        # `spawn` is the safe default on macOS (and required if the parent
        # already imported torch / matplotlib). `fork` deadlocks with some
        # Mac frameworks.
        mp_ctx = mp.get_context(ctx)
        env_kwargs = dict(env_kwargs or {})
        self.num_envs = num_envs
        self.frame_skip = frame_skip
        self.obs_size = obs_size
        self.frame_stack = frame_stack

        self._parent_conns: list = []
        self._procs: list = []
        for i in range(num_envs):
            parent_conn, child_conn = mp_ctx.Pipe(duplex=True)
            p = mp_ctx.Process(
                target=_worker_main,
                args=(child_conn, env_kwargs, frame_skip, obs_size,
                      frame_stack, i),
                daemon=True,
            )
            p.start()
            child_conn.close()  # only the worker keeps its end open
            self._parent_conns.append(parent_conn)
            self._procs.append(p)

    # -- main API --------------------------------------------------------

    def reset(self) -> np.ndarray:
        for c in self._parent_conns:
            c.send(("reset", None))
        return self._gather_obs()

    def step(self, actions) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        if len(actions) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} actions, got {len(actions)}")
        for c, a in zip(self._parent_conns, actions):
            c.send(("step", int(a)))
        results = [c.recv() for c in self._parent_conns]
        # Sanity: a worker can ship ("error", msg) on failure -- raise.
        for i, r in enumerate(results):
            if isinstance(r, tuple) and len(r) == 2 and r[0] == "error":
                raise RuntimeError(f"worker {i} failed: {r[1]}")
        obs = np.stack([r[0] for r in results], axis=0)
        rewards = np.array([r[1] for r in results], dtype=np.float32)
        dones = np.array([r[2] for r in results], dtype=bool)
        infos = [r[3] for r in results]
        return obs, rewards, dones, infos

    def close(self) -> None:
        for c in self._parent_conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        for c in self._parent_conns:
            try:
                c.close()
            except Exception:
                pass

    # -- internals -------------------------------------------------------

    def _gather_obs(self) -> np.ndarray:
        results = [c.recv() for c in self._parent_conns]
        return np.stack(results, axis=0)


def _smoke_test(num_envs: int = 2, steps: int = 8) -> None:
    """Spawn N workers, do a few random steps, print throughput. Sanity only."""
    import time
    rng = np.random.default_rng(0)
    vec = SubprocVecEnv(num_envs=num_envs)
    print(f"vec: num_envs={vec.num_envs}, obs_shape after reset:")
    obs = vec.reset()
    print(f"  {obs.shape} {obs.dtype}")

    t0 = time.monotonic()
    for t in range(steps):
        actions = rng.integers(0, 6, size=num_envs)
        obs, rewards, dones, infos = vec.step(actions)
        if t == 0:
            print(f"  step shape: {obs.shape}, rewards.shape={rewards.shape}")
        if dones.any():
            print(f"  step {t}: dones={dones.tolist()}, infos="
                  f"{[(i['score'], i['lives']) for i in infos]}")
    elapsed = time.monotonic() - t0
    total = steps * num_envs
    print(f"{total} policy steps in {elapsed:.2f}s = {total/elapsed:.0f} sps "
          f"({total/elapsed * 4:.0f} emu fps incl. frame_skip)")
    vec.close()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    s = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    _smoke_test(num_envs=n, steps=s)
