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
  4 bag intact          (currently always zero; CV doesn't detect bags yet)
  5 cherry              (currently always zero; CV doesn't detect cherry)

Plus a small per-step scalar tail (concatenated by the trainer if
desired): score, lives, frames_since_last_event, etc. For now we ship
just the masks; the agent can infer urgency from the digger/monster
spatial relationship.
"""

from __future__ import annotations

import numpy as np

from digger_env import DiggerEnv
from tools.game_state import MHEIGHT, MWIDTH, extract_state_fast as extract_state

OBS_CHANNELS = 6
OBS_SHAPE = (OBS_CHANNELS, MHEIGHT, MWIDTH)


def state_to_tensor(state) -> np.ndarray:
    """GameState -> (OBS_CHANNELS, MHEIGHT, MWIDTH) float32 in [0, 1]."""
    obs = np.zeros(OBS_SHAPE, dtype=np.float32)
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
    """

    NUM_ACTIONS = DiggerEnv.NUM_ACTIONS

    def __init__(self, shaping_coef: float = 0.0, **digger_kwargs):
        self._env = DiggerEnv(**digger_kwargs)
        self._last_state = None
        self.shaping_coef = shaping_coef
        self._prev_dist: float | None = None

    def reset(self) -> np.ndarray:
        raw = self._env.reset()
        state = extract_state(raw)
        self._last_state = state
        self._prev_dist = _nearest_emerald_distance(state)
        return state_to_tensor(state)

    def step(self, action: int):
        s = self._env.step(action)
        state = extract_state(s.obs)
        self._last_state = state
        reward = float(s.reward)
        if self.shaping_coef > 0:
            cur_dist = _nearest_emerald_distance(state)
            if self._prev_dist is not None and cur_dist is not None:
                # Positive when distance decreased (we got closer).
                reward += self.shaping_coef * (self._prev_dist - cur_dist)
            self._prev_dist = cur_dist
        info = dict(s.info)
        info["score_reward"] = float(s.reward)  # original raw signal
        return state_to_tensor(state), reward, s.done, info

    def close(self) -> None:
        self._env.close()
