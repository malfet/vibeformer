"""Symbolic-observation DiggerEnv: returns a small (C, MHEIGHT, MWIDTH) tensor.

Wraps DiggerEnv and converts each raw RGBA frame into a stack of boolean
per-tile masks via the CV extractor in game_state.py. The result is a
tiny structured observation (10x15 grid x ~6 channels) so the agent
network can be a few-conv MLP instead of a multi-million-param CNN
chewing on 84x84 raw pixels.

Channel layout (each plane is a 10x15 float32 mask):
  0 dirt
  1 emerald
  2 digger
  3 monster (any nobbin / hobbin in that tile)
  4 bag intact
  5 cherry              (currently always zero; CV doesn't detect cherry)

Frame stacking: a single frame is non-Markov for everything that moves
(monster velocity, falling bag, fire cooldown, digger sub-tile pos).
`SymbolicDiggerEnv(frame_stack=N)` concatenates the last N single-frame
observations along the channel axis, producing a (BASE_OBS_CHANNELS*N,
MHEIGHT, MWIDTH) tensor. The agent learns velocity from inter-frame
differences. N=1 reproduces the original single-frame behaviour.

Plus a small per-step scalar tail (concatenated by the trainer if
desired): score, lives, frames_since_last_event, etc. For now we ship
just the masks; the agent can infer urgency from the digger/monster
spatial relationship.

Egocentric mode (`egocentric=True`) re-centres the whole stack on the
digger: the board is pasted into a (2*MHEIGHT-1, 2*MWIDTH-1) canvas so
the digger always sits at the middle cell, plus one extra plane marking
the off-board region. This is the snake project's finding #16 — a
flatten->FC head over an allocentric grid has to learn every board
position as a separate case, while head-centred obs makes translation
invariance structural. All stacked frames are re-centred on the
*current* digger tile (not each frame's own), so self-motion stays
visible in the inter-frame differences.
"""

from __future__ import annotations

import collections

import numpy as np

from digger_env import DiggerEnv
from tools.game_state import MHEIGHT, MWIDTH, extract_state_fast as extract_state

# Channels in a single symbolic frame.
BASE_OBS_CHANNELS = 6
BASE_OBS_SHAPE = (BASE_OBS_CHANNELS, MHEIGHT, MWIDTH)

# Backwards-compat aliases: default = no frame stacking, so old imports
# (`from tools.symbolic_env import OBS_CHANNELS, OBS_SHAPE`) still see
# what they saw before. New code should prefer env.obs_channels /
# env.obs_shape, which reflect the configured frame_stack.
OBS_CHANNELS = BASE_OBS_CHANNELS
OBS_SHAPE = BASE_OBS_SHAPE


# Egocentric canvas: the board pasted anywhere from "digger at (0,0)" to
# "digger at (MHEIGHT-1, MWIDTH-1)" still fits, with the digger pinned at
# the centre cell (MHEIGHT-1, MWIDTH-1).
EGO_HEIGHT = 2 * MHEIGHT - 1
EGO_WIDTH = 2 * MWIDTH - 1


def stacked_obs_shape(frame_stack: int,
                      egocentric: bool = False) -> tuple[int, int, int]:
    if egocentric:
        # +1 plane: 1 outside the board, 0 on it.
        return (BASE_OBS_CHANNELS * frame_stack + 1, EGO_HEIGHT, EGO_WIDTH)
    return (BASE_OBS_CHANNELS * frame_stack, MHEIGHT, MWIDTH)


def egocentric_view(board: np.ndarray, row: int, col: int) -> np.ndarray:
    """(C, MHEIGHT, MWIDTH) -> (C+1, EGO_HEIGHT, EGO_WIDTH), digger centred.

    `row`/`col` is the digger tile that should land on the centre cell.
    The appended plane is 1 outside the board and 0 on it, so the agent
    can tell "wall" from "empty" at the canvas edges.
    """
    c = board.shape[0]
    out = np.zeros((c + 1, EGO_HEIGHT, EGO_WIDTH), dtype=np.float32)
    out[c] = 1.0
    r0 = MHEIGHT - 1 - int(row)
    c0 = MWIDTH - 1 - int(col)
    out[:c, r0:r0 + MHEIGHT, c0:c0 + MWIDTH] = board
    out[c, r0:r0 + MHEIGHT, c0:c0 + MWIDTH] = 0.0
    return out


def state_to_tensor(state) -> np.ndarray:
    """GameState -> (BASE_OBS_CHANNELS, MHEIGHT, MWIDTH) float32 in [0, 1].

    Single frame; frame-stacking (if any) is handled by SymbolicDiggerEnv.
    """
    obs = np.zeros(BASE_OBS_SHAPE, dtype=np.float32)
    obs[0] = state.dirt.astype(np.float32)
    obs[1] = state.emeralds.astype(np.float32)
    if state.digger is not None and state.digger.present:
        obs[2, state.digger.row, state.digger.col] = 1.0
    for m in state.monsters:
        if 0 <= m.row < MHEIGHT and 0 <= m.col < MWIDTH:
            obs[3, m.row, m.col] = 1.0
    for b in state.bags:
        if 0 <= b.row < MHEIGHT and 0 <= b.col < MWIDTH:
            obs[4, b.row, b.col] = 1.0
    if state.cherry is not None:
        cc, cr = state.cherry
        if 0 <= cr < MHEIGHT and 0 <= cc < MWIDTH:
            obs[5, cr, cc] = 1.0
    return obs


def _nearest_emerald_distance(state) -> float | None:
    """Manhattan distance from the digger to the nearest emerald, or None."""
    if state.digger is None or not state.digger.present:
        return None
    em_rows, em_cols = np.where(state.emeralds)
    if em_rows.size == 0:
        return None
    dr = np.abs(em_rows - state.digger.row)
    dc = np.abs(em_cols - state.digger.col)
    return float((dr + dc).min())


class SymbolicDiggerEnv:
    """DiggerEnv that emits a small grid tensor instead of raw pixels.

    Same step/reset protocol as DiggerEnv (reset() -> obs;
    step(action) -> (obs, reward, done, info)) but `obs` is the symbolic
    tensor.

    `shaping_coef > 0` adds a per-step dense reward = shaping_coef *
    (prev_dist - cur_dist) using Manhattan distance to the nearest
    emerald. Classic potential-based shaping: rewards getting closer,
    penalises moving away, sums to zero over a complete trajectory so it
    doesn't bias the optimal policy.

    Reward composition, per emulator frame:

        score_weight * (game score delta)
      + shaping_coef * (manhattan distance closed toward an emerald)
      + survival_reward          (only on frames that didn't end a life)
      - time_penalty
      - death_penalty            (on any life loss)

    `score_weight=0, survival_reward>0, shaping_coef=0` gives the pure
    "maximise frames alive" objective: the game's own score is invisible
    to the agent and the only thing that matters is not dying.
    """

    NUM_ACTIONS = DiggerEnv.NUM_ACTIONS

    def __init__(self, shaping_coef: float = 0.0,
                 time_penalty: float = 0.0,
                 survival_reward: float = 0.0,
                 score_weight: float = 1.0,
                 death_penalty: float = 0.0,
                 frame_stack: int = 1,
                 egocentric: bool = False, **digger_kwargs):
        if frame_stack < 1:
            raise ValueError(f"frame_stack must be >=1, got {frame_stack}")
        # death_penalty is applied here, not inside DiggerEnv, so that
        # score_weight=0 doesn't silently scale it away along with the
        # score signal.
        self._env = DiggerEnv(**digger_kwargs)
        self._last_state = None
        self.shaping_coef = shaping_coef
        # Per-agent-step reward subtracted regardless of action. Pairs
        # naturally with death_penalty to push the policy out of the
        # "wander forever without scoring" basin: every step has a
        # small opportunity cost, so hiding is no longer a free lunch.
        self.time_penalty = time_penalty
        # The mirror image: paid for every frame the digger stays alive.
        self.survival_reward = survival_reward
        self.score_weight = score_weight
        self.death_penalty = death_penalty
        self.frame_stack = frame_stack
        self.egocentric = egocentric
        self._stack: collections.deque[np.ndarray] = collections.deque(
            maxlen=frame_stack)
        self._prev_dist: float | None = None
        self._prev_lives: int = -1
        # Last tile the digger was seen on; the egocentric transform needs
        # a centre even on frames where the CV extractor loses the sprite
        # (mid-death animation, level transition).
        self._digger_rc: tuple[int, int] = (MHEIGHT // 2, MWIDTH // 2)

    @property
    def obs_channels(self) -> int:
        return self.obs_shape[0]

    @property
    def obs_shape(self) -> tuple[int, int, int]:
        return stacked_obs_shape(self.frame_stack, self.egocentric)

    def _push_frame(self, state) -> np.ndarray:
        """Append a fresh single-frame obs and return the stacked obs."""
        self._stack.append(state_to_tensor(state))
        return self.current_obs()

    def current_obs(self) -> np.ndarray:
        """Return the current stacked obs without advancing the env.

        Useful for evaluation / playback paths that need to query the
        agent at the start of a new episode after a manual reset.
        """
        if len(self._stack) == 0:
            raise RuntimeError("env.reset() must be called before current_obs()")
        if self.frame_stack == 1:
            board = self._stack[0].copy()
        else:
            board = np.concatenate(self._stack, axis=0)
        if not self.egocentric:
            return board
        return egocentric_view(board, *self._digger_rc)

    def _note_digger(self, state) -> None:
        if state.digger is not None and state.digger.present:
            r, c = state.digger.row, state.digger.col
            if 0 <= r < MHEIGHT and 0 <= c < MWIDTH:
                self._digger_rc = (r, c)

    def reset(self) -> np.ndarray:
        raw = self._env.reset()
        state = extract_state(raw)
        self._last_state = state
        self._prev_dist = _nearest_emerald_distance(state)
        self._prev_lives = -1
        self._note_digger(state)
        # Fill the stack with copies of the initial frame so the very
        # first action sees a (C*N, H, W) tensor of consistent shape.
        frame0 = state_to_tensor(state)
        self._stack.clear()
        for _ in range(self.frame_stack):
            self._stack.append(frame0.copy())
        return self.current_obs()

    def step(self, action: int):
        s = self._env.step(action)
        state = extract_state(s.obs)
        self._last_state = state
        self._note_digger(state)
        reward = self.score_weight * float(s.reward)
        if self.shaping_coef > 0:
            cur_dist = _nearest_emerald_distance(state)
            if self._prev_dist is not None and cur_dist is not None:
                # Positive when distance decreased (we got closer).
                reward += self.shaping_coef * (self._prev_dist - cur_dist)
            self._prev_dist = cur_dist
        lives = int(s.info.get("lives", 0))
        died = self._prev_lives > 0 and lives < self._prev_lives
        self._prev_lives = lives
        if self.survival_reward and not died:
            reward += self.survival_reward
        if self.death_penalty and died:
            reward -= self.death_penalty
        if self.time_penalty > 0:
            reward -= self.time_penalty
        info = dict(s.info)
        info["score_reward"] = float(s.reward)  # original raw signal
        info["death"] = bool(died)
        return self._push_frame(state), reward, s.done, info

    def save_state(self) -> dict:
        """Snapshot a state that load_state() can replay. Wraps the
        underlying DiggerEnv snapshot and adds the symbolic shaping
        baseline so reward shaping continues consistently.
        """
        if self._last_state is None:
            raise RuntimeError("save_state() called before reset()")
        return {
            "env": self._env.save_state(),
            "prev_dist": self._prev_dist,
        }

    def load_state(self, state: dict) -> np.ndarray:
        """Restore a snapshot produced by save_state(). Returns the
        stacked observation at the restored state.

        The frame stack is refilled with the post-restore frame so the
        policy doesn't see a stack mixing frames from before and after
        the restore point.
        """
        raw = self._env.load_state(state["env"])
        parsed = extract_state(raw)
        self._last_state = parsed
        self._prev_lives = -1
        self._note_digger(parsed)
        # Prefer the saved shaping baseline so the very next step's
        # delta is consistent with the state that was saved. Fall back to
        # recomputing if the field is missing (e.g. old pickles).
        self._prev_dist = state.get(
            "prev_dist", _nearest_emerald_distance(parsed))
        frame0 = state_to_tensor(parsed)
        self._stack.clear()
        for _ in range(self.frame_stack):
            self._stack.append(frame0.copy())
        return self.current_obs()

    def close(self) -> None:
        self._env.close()
