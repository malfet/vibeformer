"""Single-process RL env wrapping DOSBox Pure + Digger.

No gymnasium dependency. Just a minimal protocol:

    env = DiggerEnv()
    obs = env.reset()
    while not done:
        action = policy(obs)
        obs, reward, done, info = env.step(action)

LibretroCore is a per-process singleton (libretro callbacks have no userdata),
so a single DiggerEnv per process. For vectorized rollouts spawn subprocesses.
"""

from __future__ import annotations

import gc
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

    def __init__(self, max_steps: int = 36000):
        # 36000 frames ~= 8.5 minutes of in-game time at 70 fps. Plenty for
        # a Digger run; cap exists so an immortal idle policy doesn't loop
        # forever.
        self.max_steps = max_steps
        self._core: _libretro.LibretroCore | None = None
        self._last_action = self.NOOP
        self._last_score = 0
        self._seen_alive = False
        self._steps = 0

    # -- lifecycle ---------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Start a fresh episode. Returns the initial observation."""
        SYS_DIR.mkdir(parents=True, exist_ok=True)
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

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
        reward = float(score - self._last_score)
        self._last_score = score

        lives = self._read_lives()
        if lives > 0:
            self._seen_alive = True

        game_over = self._seen_alive and lives == 0
        truncated = self._steps >= self.max_steps
        done = game_over or truncated

        return StepResult(
            obs=self._core.get_frame(),
            reward=reward,
            done=done,
            info={"score": score, "lives": int(lives), "truncated": truncated},
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
