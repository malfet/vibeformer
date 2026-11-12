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
import os
import pickle
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import _libretro

REPO = Path(__file__).parent.resolve()
# Lockstep core (patches/dosbox-pure-lockstep.patch, built by `make core`):
# stock DOSBox Pure starts emulating the next frame on its own thread and
# returns from retro_run, so RAM reads, key events and serialize() land at
# a host-timing-dependent point mid-frame. That made restores drift ~1
# frame and repeated restores of one state diverge (README Lesson 21).
# With DBP_LOCKSTEP set, each run() emulates exactly one whole frame and
# restores replay byte-identically, for ~20% less emulator throughput.
# DIGGER_LOCKSTEP=0 opts out; DIGGER_CORE overrides the library.
_CORE_DIR = REPO / "vendor" / "dosbox-pure"
LOCKSTEP = (os.environ.get("DIGGER_LOCKSTEP", "1") != "0"
            and (_CORE_DIR / "dosbox_pure_libretro_lockstep.dylib").exists())
if LOCKSTEP:
    os.environ["DBP_LOCKSTEP"] = "1"           # read by the core at run()
CORE_PATH = (Path(os.environ["DIGGER_CORE"]) if "DIGGER_CORE" in os.environ
             else _CORE_DIR / ("dosbox_pure_libretro_lockstep.dylib"
                               if LOCKSTEP else "dosbox_pure_libretro.dylib"))
GAME_PATH = REPO / "data" / "DIGGER.EXE"
SYS_DIR = REPO / "data" / "system"
SAVE_DIR = REPO / "data" / "save"

# RAM offsets located by diffing memory across gameplay events; see commit
# history. Both live in memory region 0 (the DOS GAME segment exposed via
# DOSBox Pure's SET_MEMORY_MAPS).
SCORE_OFFSET = 0x282E0  # int32 LE
LIVES_OFFSET = 0x259F2  # uint8
# gamedat[curplayer].level (main.c), int16, 1-based. Found as the only
# byte equal to the level at every level start of 8 games (levels 1-3).
LEVEL_OFFSET = 0x2599E

# The digger's own record sits right next to LIVES_OFFSET (same struct).
# Located by scanning for the spawn values (x=152, y=180, col=7, row=9)
# and keeping the words that tracked controlled moves; the death fields by
# logging the struct through a monster kill. All int16 LE.
DIGGER_X_OFFSET = 0x259C8      # pixel x (col * 20 + 12 when tile-aligned)
DIGGER_Y_OFFSET = 0x259CA      # pixel y (row * 18 + 18 when tile-aligned)
DIGGER_COL_OFFSET = 0x259CC    # tile column h, 0..14
DIGGER_ROW_OFFSET = 0x259CE    # tile row v, 0..9
DEATH_STAGE_OFFSET = 0x259E4   # 1 = alive; 3/5/2/4 = death sequence
DEATH_TIMER_OFFSET = 0x259EA   # tombstone countdown (59 -> 0)
DEATH_STAGE_ALIVE = 1
# The stage alone misses bag deaths: DGR_DEATH_BAG is also 1, the value
# the digger lives with, so a bag kill (killdigger(n, 1, bag)) leaves it
# at 1 for the ~4 steps the bag takes to land. digdat.dob.alive (bool,
# -1 / 0) and deathbag (the bag that did it, 0 for a monster) catch the
# hit itself; at onset, stage 1 means a bag and 3 a monster.
DIGGER_ALIVE_OFFSET = 0x259F8  # -1 alive, 0 from the hit on
DEATH_BAG_OFFSET = 0x259E6     # nonzero: killed by that bag
# A death begins ~100 agent steps (frame_skip=4) before `lives` drops:
# bag/monster hit, animation, then the tombstone countdown. Input is
# ignored for that whole stretch, so anything that keys off the lives
# decrement (episodic-life dones, near-death snapshots) is late by that
# much. Key off the alive flag instead.


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
                 clip_reward: bool = True,
                 episodic_life: bool = False,
                 death_penalty: float = 0.0,
                 cycles: str | None = "7800",
                 start_states: str | None = None,
                 start_prob: float = 0.0):
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
        # DOSBox Pure "dosbox_pure_cycles" (None = core default, "auto").
        # Digger paces itself off the PIT timer, so emulating a slower CPU
        # mostly just makes frames cheaper: 7800 (386DX-33) matches
        # auto-mode gameplay timing (NOOP death onsets within a few frames)
        # at ~6x less emulation cost. 2750 (286-12) is cheaper still but
        # starves the game: monsters arrive ~30-80 frames late.
        self.cycles = cycles
        # start_states: directories of tools/level_states.py-format
        # pickles, "DIR" or "DIR=W,DIR=W" (weights pick the directory, then
        # a file uniformly; e.g. level starts vs tools/rescue.py near-death
        # states). On a hard reset, with probability start_prob the game
        # continues from one of them instead of level 1, so training sees
        # later levels and the situations a student dies in.
        self.start_prob = float(start_prob)
        self._start_pools: list[list[Path]] = []
        weights = []
        for part in (start_states or "").split(","):
            if not part:
                continue
            d, _, w = part.partition("=")
            files = sorted(Path(d).glob("L*.pkl"))
            if not files:
                raise ValueError(f"no L*.pkl start states in {d!r}")
            self._start_pools.append(files)
            weights.append(float(w or 1.0))
        if self.start_prob > 0 and not self._start_pools:
            raise ValueError("start_prob > 0 needs start_states")
        self._pool_p = np.array(weights) / sum(weights) if weights else None
        self._rng = np.random.default_rng()
        self._core: _libretro.LibretroCore | None = None
        self._last_action = self.NOOP
        self._last_score = 0
        self._seen_alive = False
        self._steps = 0
        self._prev_lives = -1
        self._prev_dying = False
        self._death_counted = False  # onset already reported for this life
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
            # Release any key the dying life had held. If we just set
            # _last_action=NOOP without releasing, the next agent action
            # of NOOP early-returns in _apply_action and the prior key
            # (e.g. RIGHT) keeps driving the emulator while the trainer
            # records NOOP -- silent action mismatch every episodic
            # death.
            prior_key = self._ACTION_TO_KEY.get(self._last_action)
            if prior_key is not None:
                self._core.set_key(prior_key, False)
            self._last_action = self.NOOP
            # The episode ended at death *onset*; play the input-ignored
            # death sequence out here so the next episode starts at the
            # respawn instead of inside ~100 steps of dead time.
            self._skip_death_sequence()
        if (self.episodic_life and not self._real_game_over
                and self._core is not None):
            self._last_score = self._read_score()
            self._prev_lives = self._read_lives()
            self._prev_dying = False
            self._death_counted = False
            self._steps = 0
            return self._core.get_frame()

        # LibretroCore is a singleton per process. Drop the old one first so
        # the new constructor doesn't see another live instance.
        if self._core is not None:
            self._core = None
            gc.collect()

        self._core = _libretro.LibretroCore(
            str(CORE_PATH), str(SYS_DIR), str(SAVE_DIR))
        if self.cycles is not None:
            self._core.set_option("dosbox_pure_cycles", str(self.cycles))
        self._core.load_game(str(GAME_PATH))
        self._advance_to_gameplay()

        self._last_action = self.NOOP
        self._last_score = self._read_score()
        self._seen_alive = self._read_lives() > 0
        self._prev_lives = self._read_lives()
        self._prev_dying = False
        self._death_counted = False
        self._real_game_over = False
        self._steps = 0
        if self._start_pools and self._rng.random() < self.start_prob:
            pool = self._start_pools[self._rng.choice(len(self._start_pools),
                                                      p=self._pool_p)]
            f = pool[self._rng.integers(len(pool))]
            with open(f, "rb") as fh:
                d = pickle.load(fh)
            frame = self.load_state(d["state"]["env"])
            self._steps = 0
            return frame
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

        # Death = the RAM death stage leaving "alive" (onset), ~100 agent
        # steps before `lives` drops. The lives drop still counts as a death
        # when no onset was seen for this life (e.g. a state restored from
        # inside the death sequence), so every life loss is reported once.
        dig = self.read_digger()
        onset = dig["dying"] and not self._prev_dying
        self._prev_dying = dig["dying"]
        lives_dropped = self._prev_lives > 0 and lives < self._prev_lives
        self._prev_lives = lives
        onset_event = onset and not self._death_counted
        death_event = onset_event or (
            lives_dropped and not self._death_counted)
        if death_event:
            self._death_counted = True
        if lives_dropped:
            self._death_counted = False
        # Onset on the last life seals the game: report game over now
        # rather than ~100 steps later when `lives` reaches 0.
        if onset_event and lives <= 1:
            real_game_over = self._real_game_over = True

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
                  "death_event": bool(death_event),
                  **dig,
                  "truncated": truncated, "real_done": real_game_over,
                  "raw_reward": raw_reward},
        )

    def close(self) -> None:
        if self._core is not None:
            self._core = None
            gc.collect()

    # -- helpers -----------------------------------------------------------

    def _advance_to_gameplay(self, max_frames: int = 3000) -> None:
        """Get from boot to the first frame of gameplay: tap RETURN every 60
        frames until RAM shows lives > 0 and the digger alive. A fixed frame
        count is not enough, because boot time depends on the emulated
        CPU speed (`cycles`)."""
        key = _libretro.RETROK.RETURN
        for f in range(max_frames):
            if f >= 20 and self._in_gameplay():
                self._core.set_key(key, False)
                return
            if f % 60 == 20:
                self._core.set_key(key, True)
            elif f % 60 == 30:
                self._core.set_key(key, False)
            self._core.run()
        raise RuntimeError(f"game did not start within {max_frames} frames")

    def _in_gameplay(self) -> bool:
        m = self._core.read_memory_region(0)
        stage = struct.unpack_from("<h", m, DEATH_STAGE_OFFSET)[0]
        return 0 < m[LIVES_OFFSET] <= 5 and stage == DEATH_STAGE_ALIVE

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

    # -- save / restore ----------------------------------------------------

    def save_state(self) -> dict:
        """Snapshot the full env state so a later load_state() resumes here.

        Returned dict has:
          - "core": opaque bytes from retro_serialize (DOSBox Pure memory,
            registers, peripherals).
          - The Python-side tracking that lives in this wrapper: step count,
            reward-baseline score, life tracking, and the last action key
            we pressed.

        Pickle the dict (the "core" entry is already plain bytes) to
        write to disk; nothing here is non-pickleable.
        """
        if self._core is None:
            raise RuntimeError("save_state() called before reset()")
        return {
            "core": self._core.serialize(),
            "held_keys": list(self._core.get_held_keys()),
            "last_action": int(self._last_action),
            "last_score": int(self._last_score),
            "seen_alive": bool(self._seen_alive),
            "steps": int(self._steps),
            "prev_lives": int(self._prev_lives),
            "real_game_over": bool(self._real_game_over),
            "death_counted": bool(self._death_counted),
        }

    def load_state(self, state: dict, refresh: bool = True) -> np.ndarray:
        """Restore a snapshot from save_state(). Returns the current frame.

        We release any frontend-held key before unserialize so the C++
        keyboard mirror doesn't disagree with the core's just-restored
        internal state. Held-key continuity is sacrificed: at the next
        step() the agent's chosen action drives the key press fresh.
        For RL where actions are re-chosen each step that's fine.
        """
        if self._core is None:
            raise RuntimeError("load_state() called before reset()")
        # Release any keys held in the frontend mirror; events get sent
        # to the core *before* we unserialize, so the new state isn't
        # touched.
        self._core.clear_keys()
        self._core.unserialize(state["core"])
        # Restore the held-key mirror to whatever was held at save time,
        # *without* firing keyboard events (set_held_keys_raw). The core's
        # internal state was reverted by unserialize, so the polled input
        # now agrees with what the core remembers.
        # That is not enough on its own: clear_keys()'s release events
        # reach the guest on the next run(), *after* the restore, so the
        # game sees the key go up while the mirror says it is held, and
        # _apply_action() never presses it again (the digger stalled
        # after every restore). So start with nothing held and let the
        # next step press its key fresh.
        self._core.set_held_keys_raw([])
        # The framebuffer is not part of the serialized state: right after
        # unserialize, get_frame() still returns whatever the core last
        # drew (the pre-restore screen, or the spawn screen in a freshly
        # booted process), and the game needs ~3 frames to redraw. Run
        # them so the returned observation matches the restored game.
        # refresh=False skips this for callers that restore many times per
        # step (search rollouts) and must not advance game time; they get a
        # stale frame back and have to cope with it themselves.
        if refresh:
            for _ in range(3):
                self._core.run()
        # Wrapper fields are tolerated as missing: a state snapshot from
        # play.py --live S only contains {core, held_keys} because
        # the live viewer talks to the raw libretro core, not DiggerEnv.
        # When the field is absent, recompute from the now-restored
        # emulator RAM so the env stays consistent.
        live_score = self._read_score()
        live_lives = self._read_lives()
        self._last_action = self.NOOP          # nothing held; see above
        self._last_score = int(state.get("last_score", live_score))
        self._seen_alive = bool(state.get("seen_alive", live_lives > 0))
        self._steps = int(state.get("steps", 0))
        self._prev_lives = int(state.get("prev_lives", live_lives))
        self._real_game_over = bool(state.get("real_game_over", False))
        self._prev_dying = self.read_digger()["dying"]
        self._death_counted = bool(state.get("death_counted",
                                             self._prev_dying))
        return self._core.get_frame()

    def _skip_death_sequence(self, max_frames: int = 3000) -> None:
        """Run the core with no input until the death sequence finishes:
        the digger respawns (stage back to alive, held for a few frames so
        a one-frame blip mid-sequence can't end it early) or the game is
        over. No-op when the digger isn't dying."""
        if not self.read_digger()["dying"]:
            return
        alive_run = 0
        for _ in range(max_frames):
            if self._read_lives() == 0:
                self._real_game_over = True
                return
            alive_run = 0 if self.read_digger()["dying"] else alive_run + 1
            if alive_run >= 8:
                return
            self._core.run()
        raise RuntimeError("death sequence did not finish in "
                           f"{max_frames} frames")

    def read_digger(self) -> dict:
        """Ground-truth digger state straight from game RAM (no CV)."""
        m = self._core.read_memory_region(0)
        x, y, col, row = struct.unpack_from("<4h", m, DIGGER_X_OFFSET)
        stage = struct.unpack_from("<h", m, DEATH_STAGE_OFFSET)[0]
        alive = struct.unpack_from("<h", m, DIGGER_ALIVE_OFFSET)[0] != 0
        bag = struct.unpack_from("<h", m, DEATH_BAG_OFFSET)[0]
        dying = stage != DEATH_STAGE_ALIVE or not alive
        return {"digger_xy": (x, y), "digger_rc": (row, col),
                "death_stage": stage, "dying": dying,
                "death_cause": ("bag" if bag else "monster") if dying
                               else None}

    def read_level(self) -> int:
        m = self._core.read_memory_region(0)
        return struct.unpack_from("<h", m, LEVEL_OFFSET)[0]

    def poke_level(self, level: int) -> None:
        """Overwrite the level counter in the running game.

        There is no memory-write API, so this patches the RAM image inside
        a serialize() blob (it is stored uncompressed, though not as one
        contiguous copy of region 0: the bytes around the counter are
        located by content) and unserializes it. The game only reads the
        counter when it builds a level (layout, background, monster count
        and speed: main.c levplan / levof10), so poke level N-1 shortly
        before the current level ends to get a properly built level N.
        Like any restore this costs ~1 frame of timing phase.
        """
        m = bytes(self._core.read_memory_region(0))
        lo, hi = LEVEL_OFFSET - 64, LEVEL_OFFSET + 64
        blob = bytearray(self._core.serialize())
        at = blob.find(m[lo:hi])
        if at < 0 or blob.find(m[lo:hi], at + 1) >= 0:
            raise RuntimeError("level counter not uniquely found in state")
        struct.pack_into("<h", blob, at + LEVEL_OFFSET - lo, level)
        held = list(self._core.get_held_keys())
        self._core.unserialize(bytes(blob))
        self._core.set_held_keys_raw(held)
        if self.read_level() != level:
            raise RuntimeError("level poke did not take")

    def _read_score(self) -> int:
        m = self._core.read_memory_region(0)
        return struct.unpack_from("<i", m, SCORE_OFFSET)[0]

    def _read_lives(self) -> int:
        m = self._core.read_memory_region(0)
        return m[LIVES_OFFSET]


def load_dos_ckpt(spec: str) -> tuple[dict, str]:
    """An emulator save state, as a DiggerEnv.load_state() dict, plus a
    description. `spec` is either

      - a pickle path: a play.py S slot ({core, held_keys}), a
        DiggerEnv.save_state(), a SymbolicDiggerEnv.save_state(), or a
        tools/level_states.py file; or
      - `L<n>` / `L<n>_<k>`: level n's start (k-th of them, default 0) in
        data/save/levels (tools/level_states.py).
    """
    m = re.fullmatch(r"[Ll](\d+)(?:_(\d+))?", spec)
    path = (SAVE_DIR / "levels" / f"L{m[1]}_{m[2] or 0}_0.pkl"
            if m else Path(spec))
    if not path.exists():
        raise FileNotFoundError(f"no emulator state {spec!r} ({path})")
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    desc = str(path)
    if "level" in d:                          # tools/level_states.py
        desc += (f" (level {d['level']}, step {d['step']}, "
                 f"lives {d['lives']}, score {d['score']})")
        d = d["state"]
    if "env" in d:                            # SymbolicDiggerEnv
        d = d["env"]
    return d, desc


# -- Vectorised wrapper -----------------------------------------------------

def preprocess_uint8(rgba: np.ndarray, size: int = 84,
                     color: bool = False) -> np.ndarray:
    """Raw (H, W, 4) RGBA uint8 -> downscaled uint8.

    Grayscale: returns (size, size) via ITU-R 601 luma transform.
    Color:     returns (size, size, 3) keeping per-channel RGB (drops alpha).

    uint8 so workers can pipe small frames over multiprocessing; the trainer
    divides by 255 on the device.
    """
    import torch
    import torch.nn.functional as F
    rgb = rgba[..., :3].astype(np.float32) * (1.0 / 255.0)
    if color:
        # (H, W, 3) -> (1, 3, H, W) -> resize -> (size, size, 3)
        t = torch.from_numpy(rgb).permute(2, 0, 1)[None]
        t = F.interpolate(t, size=(size, size), mode="area")
        arr = t[0].permute(1, 2, 0).numpy()
    else:
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        t = torch.from_numpy(gray)[None, None]
        t = F.interpolate(t, size=(size, size), mode="area")
        arr = t[0, 0].numpy()
    return (arr * 255.0).clip(0, 255).astype(np.uint8)


class FrameStack:
    """Rolling stack of the last k preprocessed frames.

    Output is (channels, H, W) where channels = k for grayscale or k*3 for
    color (so a 4-frame color stack is 12 channels, with RGB triplets laid
    out frame-by-frame: [r0 g0 b0 r1 g1 b1 ...]).
    """

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
            return arr                                     # (k, H, W)
        # (k, H, W, 3) -> (k, 3, H, W) -> (k*3, H, W)
        return arr.transpose(0, 3, 1, 2).reshape(
            self.k * 3, self.size, self.size)


def _env_step_skipped(env: DiggerEnv, action: int, skip: int):
    """Hold action across `skip` emulator frames; return last raw obs."""
    total_r = 0.0
    info: dict = {}
    obs = None
    done = False
    death = False
    for _ in range(skip):
        s = env.step(action)
        total_r += s.reward
        obs = s.obs
        # death_event is a one-frame flag; keep it if it fired on any
        # sub-frame, not just the last one.
        death = death or bool(s.info.get("death_event", False))
        info = dict(s.info, death_event=death)
        if s.done:
            done = True
            break
    return obs, total_r, done, info


def _vec_worker(conn, env_kwargs, frame_skip, obs_size, frame_stack,
                color) -> None:
    """One DiggerEnv per subprocess; pipe protocol = (cmd, payload)."""
    env = DiggerEnv(**env_kwargs)
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
                    # CleanRL-style auto-reset: ship the new episode's first
                    # obs alongside the dead episode's reward/done.
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


class DiggerVecEnv:
    """Parallel DiggerEnv with frame_skip + frame_stack + preprocess built in.

    num_envs=1 runs in-process (no IPC); num_envs>1 spawns workers. Both
    return identically shaped (N, channels, H, W) uint8 observations where
    channels = frame_stack for grayscale or frame_stack*3 for color.
    """

    def __init__(self, num_envs: int = 1, frame_skip: int = 4,
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
            self._env = DiggerEnv(**env_kwargs)
            self._stack = FrameStack(k=frame_stack, size=obs_size, color=color)
            self._conns = self._procs = []
        else:
            mp_ctx = mp.get_context(ctx)
            self._conns, self._procs = [], []
            for _ in range(num_envs):
                parent, child = mp_ctx.Pipe(duplex=True)
                p = mp_ctx.Process(target=_vec_worker, daemon=True,
                                   args=(child, env_kwargs, frame_skip,
                                         obs_size, frame_stack, color))
                p.start()
                child.close()
                self._conns.append(parent)
                self._procs.append(p)

    def reset(self) -> np.ndarray:
        if self._inproc:
            obs = self._stack.reset(
                preprocess_uint8(self._env.reset(), self.obs_size, self.color))
            return obs[None]  # (1, C, H, W)
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
