"""RL env wrapping DOSBox Pure + Digger.

Two layers, no gymnasium dependency:

  DiggerEnv      — single env, raw RGBA frames. LibretroCore is a per-process
                   singleton so only one per process.
  DiggerVecEnv   — N parallel envs with frame_skip + frame_stack + preprocess
                   baked in. num_envs=1 runs in-process (no IPC overhead).
                   num_envs>1 spawns subprocesses (one DiggerEnv each), pipes
                   uint8 stacks to keep IPC bandwidth modest.

Both expose the same vec-style API for the trainer:

    vec = DiggerVecEnv(num_envs=N)
    obs = vec.reset()                    # (N, frame_stack, H, W) uint8
    obs, rewards, dones, infos = vec.step(actions)   # actions: (N,) int
"""

from __future__ import annotations

import collections
import gc
import multiprocessing as mp
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import _libretro

REPO = Path(__file__).parent.resolve()
CORE_PATH = REPO / "vendor" / "dosbox-pure" / "dosbox_pure_libretro.dylib"
GAME_PATH = REPO / "data" / "DIGGER.EXE"
SYS_DIR = REPO / "data" / "system"
SAVE_DIR = REPO / "data" / "save"

# RAM offsets located by diffing memory across gameplay events; see commit
# history. Both live in memory region 0 (the DOS GAME segment exposed via
# DOSBox Pure's SET_MEMORY_MAPS).
SCORE_OFFSET = 0x282E0  # int32 LE
LIVES_OFFSET = 0x259F2  # uint8


@dataclass
class StepResult:
    obs: np.ndarray  # (400, 640, 4) RGBA uint8 -- raw DOSBox Pure framebuffer
    reward: float    # score delta this step
    done: bool       # lives == 0 after gameplay has started, or step cap hit
    info: dict       # {"score": int, "lives": int, "truncated": bool}


class DiggerEnv:
    """Discrete-action env over the raw Digger pixel stream.

    Action space (Discrete(6)):
        0 NOOP    1 LEFT    2 RIGHT    3 UP    4 DOWN    5 FIRE

    Observation: (400, 640, 4) RGBA uint8 -- the raw framebuffer. We leave
    downsampling / palette quantization to the agent's encoder.

    Reward: per-step score delta in points (emerald=+25, kill=+250, etc.).

    Episode end: lives drops to 0 (true game-over) OR step count hits
    `max_steps` (truncation; info["truncated"] = True).
    """

    NOOP, LEFT, RIGHT, UP, DOWN, FIRE = range(6)
    NUM_ACTIONS = 6
    OBS_SHAPE = (400, 640, 4)
    OBS_DTYPE = np.uint8

    # action -> libretro keycode that, while held, causes that movement.
    # FIRE is F1 in Digger Remastered's keyboard layout, not SPACE.
    _ACTION_TO_KEY = {
        LEFT:  _libretro.RETROK.LEFT,
        RIGHT: _libretro.RETROK.RIGHT,
        UP:    _libretro.RETROK.UP,
        DOWN:  _libretro.RETROK.DOWN,
        FIRE:  _libretro.RETROK.F1,
    }

    def __init__(self, max_steps: int = 36000,
                 clip_reward: bool = False,
                 episodic_life: bool = False,
                 death_penalty: float = 0.0):
        # 36000 frames ~= 8.5 minutes of in-game time at 70 fps. Plenty for
        # a Digger run; cap exists so an immortal idle policy doesn't loop
        # forever.
        #
        # clip_reward: report sign(reward) to the agent so a +1000 bonus
        # doesn't dominate value-loss gradients (Atari PPO standard).
        # episodic_life: emit done=True on every life-loss instead of only
        # on full game-over, so the agent sees three short episodes per
        # life-pool. The actual game keeps running underneath -- reset()
        # short-circuits to the current frame until lives truly hit 0.
        # death_penalty: subtract this from reward on any life loss (each
        # transition where lives decreased). Useful in addition to (or
        # instead of) episodic_life because the value head learns to
        # discount states that lead to death, propagating back via GAE.
        self.max_steps = max_steps
        self.clip_reward = clip_reward
        self.episodic_life = episodic_life
        self.death_penalty = float(death_penalty)
        self._core: _libretro.LibretroCore | None = None
        self._last_action = self.NOOP
        self._last_score = 0
        self._seen_alive = False
        self._steps = 0
        self._prev_lives = -1
        self._real_game_over = True  # forces a hard reset on first call

    # -- lifecycle ---------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Start a fresh episode. Returns the initial observation.

        With episodic_life=True and the game still alive (i.e. we just
        emitted done=True on a life-loss), this is a soft reset: we don't
        reboot DIGGER.EXE, we just re-baseline the reward tracker and
        return the current frame. The game has already auto-respawned the
        digger and continued running. On true game-over (or truncation, or
        first call), do a full hard reset.
        """
        SYS_DIR.mkdir(parents=True, exist_ok=True)
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

        if (self.episodic_life and not self._real_game_over
                and self._core is not None):
            self._last_action = self.NOOP
            self._last_score = self._read_score()
            self._prev_lives = self._read_lives()
            self._steps = 0
            return self._core.get_frame()

        # LibretroCore is a singleton per process. Drop the old one first so
        # the new constructor doesn't see another live instance.
        if self._core is not None:
            self._core = None
            gc.collect()

        self._core = _libretro.LibretroCore(
            str(CORE_PATH), str(SYS_DIR), str(SAVE_DIR))
        self._core.load_game(str(GAME_PATH))
        self._advance_to_gameplay()

        self._last_action = self.NOOP
        self._last_score = self._read_score()
        self._seen_alive = self._read_lives() > 0
        self._prev_lives = self._read_lives()
        self._real_game_over = False
        self._steps = 0
        return self._core.get_frame()

    def step(self, action: int) -> StepResult:
        """Apply `action`, advance one emulator frame, return the result."""
        if self._core is None:
            raise RuntimeError("step() called before reset()")
        if not (0 <= action < self.NUM_ACTIONS):
            raise ValueError(f"action out of range: {action}")

        self._apply_action(action)
        self._core.run()
        self._steps += 1

        score = self._read_score()
        raw_reward = float(score - self._last_score)
        self._last_score = score
        reward = float(np.sign(raw_reward)) if self.clip_reward else raw_reward

        lives = self._read_lives()
        if lives > 0:
            self._seen_alive = True

        real_game_over = self._seen_alive and lives == 0
        if real_game_over:
            self._real_game_over = True

        # Death = lives went down this step (covers both intermediate deaths
        # and the final game-over death).
        death_event = self._prev_lives > 0 and lives < self._prev_lives
        self._prev_lives = lives

        if death_event and self.death_penalty != 0.0:
            reward -= self.death_penalty

        if self.episodic_life:
            agent_done = death_event or real_game_over
        else:
            agent_done = real_game_over

        truncated = self._steps >= self.max_steps
        if truncated:
            # Force a hard reset on next reset() so we don't resume into a
            # weird state.
            self._real_game_over = True
        done = agent_done or truncated

        return StepResult(
            obs=self._core.get_frame(),
            reward=reward,
            done=done,
            info={"score": score, "lives": int(lives),
                  "truncated": truncated, "real_done": real_game_over,
                  "raw_reward": raw_reward},
        )

    def close(self) -> None:
        if self._core is not None:
            self._core = None
            gc.collect()

    # -- helpers -----------------------------------------------------------

    def _advance_to_gameplay(self, title=20, hold=10, settle=20) -> None:
        for _ in range(title):
            self._core.run()
        self._core.set_key(_libretro.RETROK.RETURN, True)
        for _ in range(hold):
            self._core.run()
        self._core.set_key(_libretro.RETROK.RETURN, False)
        for _ in range(settle):
            self._core.run()

    def _apply_action(self, action: int) -> None:
        if action == self._last_action:
            return
        old_key = self._ACTION_TO_KEY.get(self._last_action)
        if old_key is not None:
            self._core.set_key(old_key, False)
        new_key = self._ACTION_TO_KEY.get(action)
        if new_key is not None:
            self._core.set_key(new_key, True)
        self._last_action = action

    def _read_score(self) -> int:
        m = self._core.read_memory_region(0)
        return struct.unpack_from("<i", m, SCORE_OFFSET)[0]

    def _read_lives(self) -> int:
        m = self._core.read_memory_region(0)
        return m[LIVES_OFFSET]


# -- Vectorised wrapper -----------------------------------------------------

def preprocess_uint8(rgba: np.ndarray, size: int = 84) -> np.ndarray:
    """Raw (H, W, 4) RGBA uint8 -> (size, size) uint8 grayscale.

    Mirrors the original train_ppo.preprocess float pipeline; emits uint8
    so workers can pipe small frames over multiprocessing and the trainer
    converts to float once on the device. Lazily imports torch.
    """
    import torch
    import torch.nn.functional as F
    rgb = rgba[..., :3].astype(np.float32) * (1.0 / 255.0)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    t = torch.from_numpy(gray)[None, None]
    t = F.interpolate(t, size=(size, size), mode="area")
    return (t[0, 0].numpy() * 255.0).clip(0, 255).astype(np.uint8)


class FrameStack:
    """Rolling stack of the last k preprocessed frames."""

    def __init__(self, k: int = 4, size: int = 84):
        self.k = k
        self.size = size
        self.frames: collections.deque[np.ndarray] = collections.deque(maxlen=k)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(frame)
        return np.stack(self.frames, axis=0)

    def push(self, frame: np.ndarray) -> np.ndarray:
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


def _env_step_skipped(env: DiggerEnv, action: int, skip: int):
    """Hold action across `skip` emulator frames; return last raw obs."""
    total_r = 0.0
    info: dict = {}
    obs = None
    done = False
    for _ in range(skip):
        s = env.step(action)
        total_r += s.reward
        obs = s.obs
        info = s.info
        if s.done:
            done = True
            break
    return obs, total_r, done, info


def _vec_worker(conn, env_kwargs, frame_skip, obs_size, frame_stack) -> None:
    """One DiggerEnv per subprocess; pipe protocol = (cmd, payload)."""
    env = DiggerEnv(**env_kwargs)
    stack = FrameStack(k=frame_stack, size=obs_size)
    obs = stack.reset(preprocess_uint8(env.reset(), obs_size))
    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "reset":
                obs = stack.reset(preprocess_uint8(env.reset(), obs_size))
                conn.send(obs)
            elif cmd == "step":
                raw, r, done, info = _env_step_skipped(
                    env, int(payload), frame_skip)
                if done:
                    # CleanRL-style auto-reset: ship the new episode's first
                    # obs alongside the dead episode's reward/done.
                    obs = stack.reset(preprocess_uint8(env.reset(), obs_size))
                else:
                    obs = stack.push(preprocess_uint8(raw, obs_size))
                conn.send((obs, float(r), bool(done), info))
            elif cmd == "close":
                break
    except (KeyboardInterrupt, EOFError):
        pass
    except Exception as exc:
        import traceback; traceback.print_exc()
        try: conn.send(("error", repr(exc)))
        except Exception: pass
    finally:
        try: env.close()
        except Exception: pass
        conn.close()


class DiggerVecEnv:
    """Parallel DiggerEnv with frame_skip + frame_stack + preprocess built in.

    num_envs=1 runs in-process (no IPC); num_envs>1 spawns workers. Both
    return identically shaped (N, frame_stack, obs_size, obs_size) uint8
    observations so the trainer doesn't branch.
    """

    def __init__(self, num_envs: int = 1, frame_skip: int = 4,
                 frame_stack: int = 4, obs_size: int = 84,
                 env_kwargs: dict | None = None,
                 ctx: str = "spawn"):
        self.num_envs = num_envs
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.obs_size = obs_size
        env_kwargs = dict(env_kwargs or {})

        self._inproc = num_envs == 1
        if self._inproc:
            self._env = DiggerEnv(**env_kwargs)
            self._stack = FrameStack(k=frame_stack, size=obs_size)
            self._conns = self._procs = []
        else:
            mp_ctx = mp.get_context(ctx)
            self._conns, self._procs = [], []
            for _ in range(num_envs):
                parent, child = mp_ctx.Pipe(duplex=True)
                p = mp_ctx.Process(target=_vec_worker, daemon=True,
                                   args=(child, env_kwargs, frame_skip,
                                         obs_size, frame_stack))
                p.start()
                child.close()
                self._conns.append(parent)
                self._procs.append(p)

    def reset(self) -> np.ndarray:
        if self._inproc:
            obs = self._stack.reset(
                preprocess_uint8(self._env.reset(), self.obs_size))
            return obs[None]  # (1, k, H, W)
        for c in self._conns:
            c.send(("reset", None))
        return np.stack([c.recv() for c in self._conns], axis=0)

    def step(self, actions):
        if self._inproc:
            raw, r, done, info = _env_step_skipped(
                self._env, int(actions[0]), self.frame_skip)
            if done:
                obs = self._stack.reset(
                    preprocess_uint8(self._env.reset(), self.obs_size))
            else:
                obs = self._stack.push(preprocess_uint8(raw, self.obs_size))
            return (obs[None],
                    np.array([r], dtype=np.float32),
                    np.array([done], dtype=bool),
                    [info])
        for c, a in zip(self._conns, actions):
            c.send(("step", int(a)))
        results = [c.recv() for c in self._conns]
        for i, r in enumerate(results):
            if isinstance(r, tuple) and len(r) == 2 and r[0] == "error":
                raise RuntimeError(f"vec worker {i} failed: {r[1]}")
        obs = np.stack([r[0] for r in results], axis=0)
        rewards = np.array([r[1] for r in results], dtype=np.float32)
        dones = np.array([r[2] for r in results], dtype=bool)
        infos = [r[3] for r in results]
        return obs, rewards, dones, infos

    def close(self) -> None:
        if self._inproc:
            self._env.close()
            return
        for c in self._conns:
            try: c.send(("close", None))
            except (BrokenPipeError, EOFError): pass
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        for c in self._conns:
            try: c.close()
            except Exception: pass


def _smoke_test() -> None:
    """Random-action sanity check -- meant for ad-hoc verification, not RL."""
    import random
    rng = random.Random(0)
    env = DiggerEnv(max_steps=2000)
    obs = env.reset()
    print(f"reset: obs shape={obs.shape} dtype={obs.dtype}", flush=True)

    total = 0.0
    last_score = 0
    last_lives = -1
    for t in range(2000):
        a = rng.randrange(DiggerEnv.NUM_ACTIONS)
        step = env.step(a)
        total += step.reward
        if step.info["score"] != last_score or step.info["lives"] != last_lives:
            print(f"  t={t:>4d} a={a} r={step.reward:+.0f}  "
                  f"score={step.info['score']} lives={step.info['lives']}",
                  flush=True)
            last_score = step.info["score"]
            last_lives = step.info["lives"]
        if step.done:
            print(f"  done at t={t}: truncated={step.info['truncated']}",
                  flush=True)
            break
    print(f"total reward: {total}", flush=True)
    env.close()


if __name__ == "__main__":
    _smoke_test()
