"""RL env over a faithful Python port of QBasic NIBBLES.BAS.

Two layers, mirroring ../digger-rl/digger_env.py:

  NibblesEnv     - single env, raw RGB frames + score/lives via NibblesGame.
  NibblesVecEnv  - N parallel envs with frame_stack + preprocess baked in.
                   num_envs=1 runs in-process (no IPC). num_envs>1 spawns
                   subprocess workers (one NibblesGame each).

The Python sim is deterministic enough that frame_skip is unnecessary (each
step advances the snake one cell). Trainers can still pass frame_skip>1 if
they want to model a slower control rate.

Native frame: arena rendered at 2 pixels per cell -> (100, 160, 3) RGB uint8.
That is the env's `step.obs`. The vec env preprocesses to 84x84 by default,
matching the NatureCNN input shape used in digger-rl.
"""

from __future__ import annotations

import collections
import multiprocessing as mp
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

from nibbles_sim import (
    NibblesGame,
    ARENA_ROWS, ARENA_COLS,
    EMPTY, WALL, SNAKE,
    NOOP, UP, DOWN, LEFT, RIGHT, NUM_ACTIONS,
)


# QBasic NIBBLES.BAS uses CGA/EGA palette indices. Map to RGB triples so the
# rendered frames look like the original (snake bright yellow on blue bg).
# normal: snake1=14, snake2=13, walls=12, background=1, dialog-fg=15, bg=4.
_EGA = {
    0:  (0,   0,   0),
    1:  (0,   0,   170),  # background blue
    4:  (170, 0,   0),
    12: (255, 85,  85),   # walls (bright red)
    13: (255, 85,  255),  # snake2 (bright magenta)
    14: (255, 255, 85),   # snake1 (bright yellow)
    15: (255, 255, 255),  # number digit
}

COLOR_BG      = np.array(_EGA[1],  dtype=np.uint8)
COLOR_WALL    = np.array(_EGA[12], dtype=np.uint8)
COLOR_SNAKE   = np.array(_EGA[14], dtype=np.uint8)
COLOR_NUMBER  = np.array(_EGA[15], dtype=np.uint8)


# Native render: each arena cell is RENDER_SCALE x RENDER_SCALE pixels.
# 2 -> (100, 160) frame, big enough that a 1-cell snake is visible after
# downsampling to 84x84.
RENDER_SCALE = 2
FRAME_H = ARENA_ROWS * RENDER_SCALE  # 100
FRAME_W = ARENA_COLS * RENDER_SCALE  # 160


@dataclass
class StepResult:
    obs: np.ndarray  # (FRAME_H, FRAME_W, 3) RGB uint8
    reward: float    # score delta this step
    done: bool       # game_over (lives=0) or step cap hit
    info: dict       # {"score", "lives", "level", "number_value", "ate",
                     #  "died", "level_complete", "truncated"}


class NibblesEnv:
    """Discrete-action env over the rendered Nibbles frame.

    Action space (Discrete(5)):
        0 NOOP   1 UP   2 DOWN   3 LEFT   4 RIGHT

    Observation: (FRAME_H, FRAME_W, 3) RGB uint8.

    Reward: per-step score delta. Eat number n -> +n. Die -> -10. The agent
    sees the same reward signal the original game tracks.

    Episode end: lives drops to 0 (true game-over) OR step count hits the
    sim's `max_steps` (truncation; info["truncated"] = True).
    """

    NOOP, UP, DOWN, LEFT, RIGHT = NOOP, UP, DOWN, LEFT, RIGHT
    NUM_ACTIONS = NUM_ACTIONS
    OBS_SHAPE = (FRAME_H, FRAME_W, 3)
    OBS_DTYPE = np.uint8

    def __init__(self,
                 start_level: int = 1,
                 max_steps: int = 50_000,
                 episodic_life: bool = False,
                 death_penalty_extra: float = 0.0,
                 rng_seed: Optional[int] = None):
        # episodic_life: emit done=True on every life-loss (Atari convention).
        #   The underlying game continues; reset() short-circuits until the
        #   true game-over happens.
        # death_penalty_extra: subtract this from reward on any death, in
        #   addition to the BAS's built-in -10 score penalty. Useful when
        #   tuning the value head's death aversion separately from the game
        #   score signal.
        self.start_level = start_level
        self.max_steps = max_steps
        self.episodic_life = episodic_life
        self.death_penalty_extra = float(death_penalty_extra)
        # `rng_seed` seeds an *episode-seed sampler* — each call to reset()
        # draws a fresh per-game seed from this generator. Without this, every
        # NibblesGame would share the same number-spawn sequence and the
        # whole training set would collapse onto one trajectory.
        self._seeder = random.Random(rng_seed)

        self._game: Optional[NibblesGame] = None
        self._last_score = 0
        self._prev_lives = -1
        self._real_game_over = True

    # -- lifecycle ----------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Start a fresh episode. Returns initial obs.

        With episodic_life=True and the game still has lives, this is a soft
        reset (the game already auto-respawned at the level start when the
        snake died). We just re-baseline the reward tracker and return the
        current frame.
        """
        if self._game is None or self._real_game_over:
            self._game = NibblesGame(
                start_level=self.start_level,
                max_steps=self.max_steps,
                rng_seed=self._seeder.randrange(2 ** 31),
            )
            self._real_game_over = False

        self._last_score = self._game.snake.score
        self._prev_lives = self._game.snake.lives
        return self._render()

    def step(self, action: int) -> StepResult:
        assert self._game is not None, "call reset() first"
        before_score = self._game.snake.score
        sim_info = self._game.step(int(action))

        reward = float(self._game.snake.score - before_score)
        died = sim_info.died
        if died:
            reward -= self.death_penalty_extra

        if sim_info.game_over:
            self._real_game_over = True
            done = True
        elif self.episodic_life and died:
            done = True
        elif sim_info.truncated:
            self._real_game_over = True
            done = True
        else:
            done = False

        info = {
            "score": int(sim_info.score),
            "lives": int(sim_info.lives),
            "level": int(sim_info.level),
            "number_value": int(sim_info.number_value),
            "ate": bool(sim_info.ate),
            "died": bool(died),
            "level_complete": bool(sim_info.level_complete),
            "truncated": bool(sim_info.truncated),
        }
        return StepResult(self._render(), reward, done, info)

    def close(self) -> None:
        self._game = None

    # -- rendering ----------------------------------------------------------

    def _render(self) -> np.ndarray:
        """Render the arena as a (FRAME_H, FRAME_W, 3) RGB uint8 frame."""
        g = self._game
        # Build a (50, 80, 3) base image, then upscale by RENDER_SCALE.
        small = np.empty((ARENA_ROWS, ARENA_COLS, 3), dtype=np.uint8)
        small[:] = COLOR_BG

        # Convert arena (list-of-lists, 1-indexed) to numpy once.
        # arena[r][c] for r in 1..50, c in 1..80.
        for r in range(1, ARENA_ROWS + 1):
            row = g.arena[r]
            for c in range(1, ARENA_COLS + 1):
                v = row[c]
                if v == WALL:
                    small[r - 1, c - 1] = COLOR_WALL
                elif v == SNAKE:
                    small[r - 1, c - 1] = COLOR_SNAKE

        # Draw the active number digit (occupies real-screen-row -> 2 sub-rows).
        if g.number_row != 0:
            r1 = 2 * g.number_row - 1
            r2 = 2 * g.number_row
            c = g.number_col
            small[r1 - 1, c - 1] = COLOR_NUMBER
            small[r2 - 1, c - 1] = COLOR_NUMBER

        if RENDER_SCALE == 1:
            return small
        # np.kron(small, ones((s, s, 1))) does NN upsample but is slow; faster:
        return np.repeat(np.repeat(small, RENDER_SCALE, axis=0),
                         RENDER_SCALE, axis=1)


# -- preprocess + FrameStack (copy of digger-rl shapes) -----------------------

def preprocess_uint8(rgb: np.ndarray, size: int = 84,
                     color: bool = False,
                     mode: str = "nearest") -> np.ndarray:
    """Raw (H, W, 3) RGB uint8 -> downscaled uint8.

    Grayscale: (size, size) via ITU-R 601 luma.
    Color:     (size, size, 3).

    `mode` controls F.interpolate behavior. The default "nearest" preserves
    cell colors — important when a sprite (e.g. the snake at 2 native pixels
    per cell) would get blended to a sub-pixel smear under "area".
    """
    import torch
    import torch.nn.functional as F
    rgb_f = rgb.astype(np.float32) * (1.0 / 255.0)
    if color:
        t = torch.from_numpy(rgb_f).permute(2, 0, 1)[None]
        t = F.interpolate(t, size=(size, size), mode=mode)
        arr = t[0].permute(1, 2, 0).numpy()
    else:
        gray = 0.299 * rgb_f[..., 0] + 0.587 * rgb_f[..., 1] + 0.114 * rgb_f[..., 2]
        t = torch.from_numpy(gray)[None, None]
        t = F.interpolate(t, size=(size, size), mode=mode)
        arr = t[0, 0].numpy()
    return (arr * 255.0).clip(0, 255).astype(np.uint8)


class FrameStack:
    """Rolling stack of the last k preprocessed frames."""

    def __init__(self, k: int = 4, size: int = 84, color: bool = False):
        self.k = k
        self.size = size
        self.color = color
        self.frames: collections.deque[np.ndarray] = collections.deque(maxlen=k)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(frame)
        return self._stacked()

    def push(self, frame: np.ndarray) -> np.ndarray:
        self.frames.append(frame)
        return self._stacked()

    def _stacked(self) -> np.ndarray:
        arr = np.stack(self.frames, axis=0)
        if not self.color:
            return arr  # (k, H, W)
        return arr.transpose(0, 3, 1, 2).reshape(self.k * 3, self.size, self.size)


def _env_step_skipped(env: NibblesEnv, action: int, skip: int):
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


def _vec_worker(conn, env_kwargs, frame_skip, obs_size, frame_stack, color):
    env = NibblesEnv(**env_kwargs)
    stack = FrameStack(k=frame_stack, size=obs_size, color=color)
    obs = stack.reset(preprocess_uint8(env.reset(), obs_size, color))
    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "reset":
                obs = stack.reset(preprocess_uint8(env.reset(), obs_size, color))
                conn.send(obs)
            elif cmd == "step":
                raw, r, done, info = _env_step_skipped(
                    env, int(payload), frame_skip)
                if done:
                    obs = stack.reset(preprocess_uint8(env.reset(), obs_size, color))
                else:
                    obs = stack.push(preprocess_uint8(raw, obs_size, color))
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


class NibblesVecEnv:
    """Parallel NibblesEnv with frame_skip + frame_stack + preprocess.

    num_envs=1 runs in-process. num_envs>1 spawns workers. Both return
    (N, channels, H, W) uint8 observations.
    """

    def __init__(self, num_envs: int = 1, frame_skip: int = 1,
                 frame_stack: int = 4, obs_size: int = 84,
                 color: bool = False,
                 env_kwargs: dict | None = None,
                 ctx: str = "spawn"):
        self.num_envs = num_envs
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.obs_size = obs_size
        self.color = color
        env_kwargs = dict(env_kwargs or {})

        self._inproc = num_envs == 1
        if self._inproc:
            self._env = NibblesEnv(**env_kwargs)
            self._stack = FrameStack(k=frame_stack, size=obs_size, color=color)
            self._conns = self._procs = []
        else:
            mp_ctx = mp.get_context(ctx)
            self._conns, self._procs = [], []
            for i in range(num_envs):
                # Stagger worker seeds so subprocesses don't all draw the
                # same number-spawn sequence.
                worker_kwargs = dict(env_kwargs)
                seed = env_kwargs.get("rng_seed")
                if seed is not None:
                    worker_kwargs["rng_seed"] = seed + i
                parent, child = mp_ctx.Pipe(duplex=True)
                p = mp_ctx.Process(target=_vec_worker, daemon=True,
                                   args=(child, worker_kwargs, frame_skip,
                                         obs_size, frame_stack, color))
                p.start()
                child.close()
                self._conns.append(parent)
                self._procs.append(p)

    def reset(self) -> np.ndarray:
        if self._inproc:
            obs = self._stack.reset(
                preprocess_uint8(self._env.reset(), self.obs_size, self.color))
            return obs[None]
        for c in self._conns:
            c.send(("reset", None))
        return np.stack([c.recv() for c in self._conns], axis=0)

    def step(self, actions):
        if self._inproc:
            raw, r, done, info = _env_step_skipped(
                self._env, int(actions[0]), self.frame_skip)
            if done:
                obs = self._stack.reset(
                    preprocess_uint8(self._env.reset(), self.obs_size, self.color))
            else:
                obs = self._stack.push(
                    preprocess_uint8(raw, self.obs_size, self.color))
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


# -- symbolic obs (alternative to RGB pixels) ---------------------------------

SYM_EMPTY = 0
SYM_WALL = 1
SYM_BODY = 2
SYM_HEAD = 3
SYM_NUMBER = 4
SYM_NUM_TYPES = 5


def extract_symbolic_obs(game: NibblesGame) -> np.ndarray:
    """Return a (50, 80) uint8 grid encoding the arena's cell types.

    Codes: 0=empty 1=wall 2=snake body 3=snake head 4=number cell.
    The head is distinguished from the body so a small CNN can immediately
    locate the agent without inferring it from frame-stack motion.
    """
    arr = np.empty((ARENA_ROWS, ARENA_COLS), dtype=np.uint8)
    for r in range(1, ARENA_ROWS + 1):
        row = game.arena[r]
        for c in range(1, ARENA_COLS + 1):
            arr[r - 1, c - 1] = row[c]  # 0=EMPTY 1=WALL 2=SNAKE
    head_r, head_c = game.head
    arr[head_r - 1, head_c - 1] = SYM_HEAD
    if game.number_row != 0:
        r1 = 2 * game.number_row - 1
        r2 = 2 * game.number_row
        c = game.number_col
        arr[r1 - 1, c - 1] = SYM_NUMBER
        arr[r2 - 1, c - 1] = SYM_NUMBER
    return arr


class SymbolicVecEnv:
    """1-env wrapper that yields symbolic obs (no pixel pipeline).

    Same API surface as the parts of NibblesVecEnv that train_bc.py touches
    (reset → obs[None], step → (obs[None], reward, done, info), `_env` attr
    so the heuristic can read `vec._env._game`). Frame-stacking is skipped
    because symbolic obs is already fully observable.
    """

    OBS_SHAPE = (ARENA_ROWS, ARENA_COLS)
    NUM_ACTIONS = NUM_ACTIONS

    def __init__(self, env_kwargs: dict | None = None):
        env_kwargs = dict(env_kwargs or {})
        self._env = NibblesEnv(**env_kwargs)

    def reset(self) -> np.ndarray:
        self._env.reset()
        return extract_symbolic_obs(self._env._game)[None]

    def step(self, actions):
        s = self._env.step(int(actions[0]))
        if s.done:
            self._env.reset()
        obs = extract_symbolic_obs(self._env._game)
        return (obs[None],
                np.array([s.reward], dtype=np.float32),
                np.array([s.done], dtype=bool),
                [s.info])

    def close(self) -> None:
        self._env.close()


def _smoke_test() -> None:
    """Random-action sanity check."""
    import random
    rng = random.Random(0)
    env = NibblesEnv(max_steps=2000, rng_seed=0)
    obs = env.reset()
    print(f"reset: obs shape={obs.shape} dtype={obs.dtype}", flush=True)
    total_r = 0.0
    n_ate = 0
    n_died = 0
    for t in range(500):
        s = env.step(rng.randrange(NibblesEnv.NUM_ACTIONS))
        total_r += s.reward
        n_ate += int(s.info["ate"])
        n_died += int(s.info["died"])
        if s.done:
            print(f"  done at step {t}: {s.info}")
            break
    print(f"final: total_r={total_r:.1f} ate={n_ate} died={n_died}")


if __name__ == "__main__":
    _smoke_test()
