"""Symbolic GameState + classical-CV extractor for Digger frames.

The intuition: pixel-based RL has hit a wall (PPO and Dreamer both stall
around avg_ret ~70-225). Digger has a fixed 11-color CGA-class palette and
a 15x10 tile grid, so a hand-coded CV pipeline can recover the *true*
underlying game state directly from the framebuffer. RL on that compact
symbolic state has a far easier learning problem than RL on raw pixels.

Layout matches sobomax's Digger Remastered source:
  Field: MWIDTH=15 columns x MHEIGHT=10 rows.
  Sprites: 1 player digger, up to 6 monsters (nobbin/hobbin), up to 7
  money bags, 1 cherry bonus, plus emeralds occupying any field tile.

Frame layout (DOSBox Pure renders DIGGER.EXE at 640x400 RGBA):
  Top ~32 px: score / lives strip.
  Below that: 10 rows of play, each row ~36.8 px tall (368/10).
  Each column ~42.67 px wide (640/15).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---- Field constants (mirror sobomax/digger def.h) ------------------------

MWIDTH = 15
MHEIGHT = 10

# Direction encoding matches sobomax's def.h: DIR_NONE=-1, R=0, U=2, L=4, D=6.
DIR_NONE, DIR_RIGHT, DIR_UP, DIR_LEFT, DIR_DOWN = -1, 0, 2, 4, 6

# ---- Frame geometry -------------------------------------------------------

FRAME_W, FRAME_H = 640, 400
SCORE_BAR_H = 32              # measured from sample frames
PLAY_TOP = SCORE_BAR_H
PLAY_BOTTOM = FRAME_H
TILE_W = FRAME_W / MWIDTH           # 42.67
TILE_H = (PLAY_BOTTOM - PLAY_TOP) / MHEIGHT  # 36.80


def tile_center(col: int, row: int) -> tuple[int, int]:
    """Return (x, y) pixel of the centre of tile (col, row) in the play area."""
    x = int((col + 0.5) * TILE_W)
    y = int(PLAY_TOP + (row + 0.5) * TILE_H)
    return x, y


def pixel_to_tile(x: int, y: int) -> tuple[int, int]:
    """Inverse: which (col, row) does play-area pixel (x, y) fall into?"""
    col = max(0, min(MWIDTH - 1, int(x / TILE_W)))
    row = max(0, min(MHEIGHT - 1, int((y - PLAY_TOP) / TILE_H)))
    return col, row


# ---- Palette --------------------------------------------------------------
# Observed RGB triplets across many frames; exact CGA palette (every pixel
# is one of these 11 colours, no anti-aliasing).

PAL_BLACK     = (0,   0,   0)    # tunnel / empty / score-bar background
PAL_DIRT_A    = (130, 65,  0)    # dirt (lighter wavy band)
PAL_DIRT_B    = (130, 0,   0)    # dirt (darker wavy band)
PAL_GREEN     = (0,   255, 0)    # emerald body, score digits
PAL_DGREEN    = (0,   130, 0)    # emerald inner / digit shade / monster body
PAL_YELLOW    = (255, 255, 0)    # bag '$', monster head, digger trim
PAL_RED       = (255, 0,   0)    # digger body, monster eyes/feet
PAL_WHITE     = (255, 255, 255)  # text
PAL_GRAY      = (130, 130, 130)  # sprite outlines / dim
PAL_DGRAY     = (65,  65,  65)   # bag outline
PAL_DCYAN     = (0,   130, 130)  # title-screen text

DIRT_COLORS = {PAL_DIRT_A, PAL_DIRT_B}
SPRITE_COLORS = {PAL_GREEN, PAL_DGREEN, PAL_YELLOW, PAL_RED}


# ---- GameState dataclasses ------------------------------------------------

@dataclass
class DiggerPos:
    """Player digger. col/row are tile coords (0..14, 0..9); dir is DIR_*."""
    col: int
    row: int
    dir: int = DIR_NONE
    alive: bool = True
    can_fire: bool = True  # CV can't read this directly from a single frame;
                           # caller must track bullet presence to set it false

    @property
    def present(self) -> bool:
        return self.alive and 0 <= self.col < MWIDTH and 0 <= self.row < MHEIGHT


@dataclass
class MonsterPos:
    col: int
    row: int
    dir: int = DIR_NONE
    is_hobbin: bool = False        # hobbin = digger-monster (can dig dirt)
    alive: bool = True


@dataclass
class BagPos:
    col: int
    row: int
    moving: bool = False           # falling or being pushed
    broken: bool = False           # broken open into gold pile


@dataclass
class GameState:
    digger: Optional[DiggerPos] = None
    monsters: list[MonsterPos] = field(default_factory=list)
    bags: list[BagPos] = field(default_factory=list)
    cherry: Optional[tuple[int, int]] = None       # (col, row) if visible

    # 15x10 boolean grids. True iff that tile contains the named object.
    emeralds: np.ndarray = field(
        default_factory=lambda: np.zeros((MHEIGHT, MWIDTH), dtype=bool))
    dirt:     np.ndarray = field(
        default_factory=lambda: np.zeros((MHEIGHT, MWIDTH), dtype=bool))

    # Status overlay (read from score bar via simple template digit OCR or
    # from RAM if available). CV alone won't fill these reliably yet.
    score: int = 0
    lives: int = 0


# ---- CV extractor ---------------------------------------------------------
# Strategy:
#  1. Mask the play area (skip score bar).
#  2. Build per-pixel "is colour X" masks. These are exact equality tests
#     since the palette has no anti-aliasing.
#  3. Sprites (digger / monsters) are the only red+green or yellow+green
#     blobs roughly the size of one tile. Locate via connected components.
#  4. Static field objects (emeralds, bags) live tile-aligned; sample each
#     tile centre to classify it.

def _color_mask(frame_rgb: np.ndarray, rgb: tuple[int, int, int]) -> np.ndarray:
    r, g, b = rgb
    return (frame_rgb[..., 0] == r) & (frame_rgb[..., 1] == g) & (frame_rgb[..., 2] == b)


def _largest_components(mask: np.ndarray, min_area: int = 20) -> list[tuple[int, tuple[int, int]]]:
    """Return list of (area, (cy, cx)) for each connected blob >= min_area.

    Simple BFS over True pixels. Adequate for the few sprites we expect
    per frame; no scipy dependency.
    """
    H, W = mask.shape
    seen = np.zeros_like(mask)
    out: list[tuple[int, tuple[int, int]]] = []
    for y in range(H):
        for x in range(W):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            ys, xs = [], []
            while stack:
                cy, cx = stack.pop()
                ys.append(cy); xs.append(cx)
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(ys) >= min_area:
                cy = int(np.mean(ys)); cx = int(np.mean(xs))
                out.append((len(ys), (cy, cx)))
    out.sort(reverse=True)
    return out


def extract_state(frame_rgba: np.ndarray) -> GameState:
    """Decode a raw 640x400 RGBA frame into a GameState.

    The frame must be the unprocessed output of `core.get_frame()` (DOSBox
    Pure framebuffer). Downscaled / preprocessed frames will not classify
    correctly because we rely on exact palette colour matches.
    """
    assert frame_rgba.shape == (FRAME_H, FRAME_W, 4), \
        f"expected 400x640 RGBA, got {frame_rgba.shape}"
    rgb = frame_rgba[..., :3]
    play = rgb[PLAY_TOP:PLAY_BOTTOM]                   # (PLAY_H, 640, 3)

    state = GameState()

    # --- masks
    m_red   = _color_mask(play, PAL_RED)
    m_dgrn  = _color_mask(play, PAL_DGREEN)
    m_grn   = _color_mask(play, PAL_GREEN)
    m_yel   = _color_mask(play, PAL_YELLOW)
    m_blk   = _color_mask(play, PAL_BLACK)
    m_dirt  = _color_mask(play, PAL_DIRT_A) | _color_mask(play, PAL_DIRT_B)

    m_dgray = _color_mask(play, PAL_DGRAY)

    # --- digger: largest connected RED blob. Player sprite is dominated
    # by red; monster eyes/feet are also red but smaller (a few px each).
    red_blobs = _largest_components(m_red, min_area=30)
    if red_blobs:
        _, (cy, cx) = red_blobs[0]
        col, row = pixel_to_tile(cx, cy + PLAY_TOP)
        state.digger = DiggerPos(col=col, row=row, alive=True)

    # --- bags: each bag has a yellow '$' glyph (~15-30 px) AND a
    # dark-gray sack body (~30+ px). The dark-gray test is the key
    # discriminator vs emerald highlights (which can also have yellow
    # but never have the gray sack).
    bag_seen: set[tuple[int, int]] = set()
    half_w_b, half_h_b = int(TILE_W * 0.5), int(TILE_H * 0.5)
    for area, (cy, cx) in _largest_components(m_yel, min_area=8):
        if area > 50:
            continue
        y0 = max(0, cy - half_h_b); y1 = min(m_dgray.shape[0], cy + half_h_b)
        x0 = max(0, cx - half_w_b); x1 = min(m_dgray.shape[1], cx + half_w_b)
        if int(m_dgray[y0:y1, x0:x1].sum()) < 25:
            continue                  # no sack body -> not a bag
        col, row = pixel_to_tile(cx, cy + PLAY_TOP)
        if (col, row) in bag_seen:
            continue
        bag_seen.add((col, row))
        state.bags.append(BagPos(col=col, row=row))

    # --- emeralds: each is a small bright-green diamond (~15-40 px).
    # Bigger green blobs are emerald clusters that bled together; we
    # accept them and let each contribute its tile.
    em_seen: set[tuple[int, int]] = set()
    for area, (cy, cx) in _largest_components(m_grn, min_area=10):
        if area > 200:               # huge - probably score-text overflow
            continue
        col, row = pixel_to_tile(cx, cy + PLAY_TOP)
        em_seen.add((col, row))
    for col, row in em_seen:
        state.emeralds[row, col] = True

    # --- monsters: dark-green body blob with yellow head AND red eyes
    # in a sprite-sized window. Lower min_area than emerald clusters to
    # catch single monsters; the head+eye combo filters false positives
    # (emerald clusters have no yellow head + no red).
    half_w, half_h = int(TILE_W * 0.7), int(TILE_H * 0.7)
    monster_seen: set[tuple[int, int]] = set()
    for area, (cy, cx) in _largest_components(m_dgrn, min_area=15):
        y0 = max(0, cy - half_h); y1 = min(m_yel.shape[0], cy + half_h)
        x0 = max(0, cx - half_w); x1 = min(m_yel.shape[1], cx + half_w)
        n_yel = int(m_yel[y0:y1, x0:x1].sum())
        n_red = int(m_red[y0:y1, x0:x1].sum())
        # Bags have lots of yellow but no red eyes; emeralds have neither.
        if n_yel >= 6 and n_red >= 3 and n_red <= 40:
            col, row = pixel_to_tile(cx, cy + PLAY_TOP)
            if state.digger and (col, row) == (state.digger.col, state.digger.row):
                continue
            if (col, row) in monster_seen:
                continue
            monster_seen.add((col, row))
            state.monsters.append(MonsterPos(col=col, row=row, is_hobbin=False))

    # --- dirt grid: tile-aligned, sample tile centres for brown.
    half_w2, half_h2 = int(TILE_W * 0.4), int(TILE_H * 0.4)
    for r in range(MHEIGHT):
        for c in range(MWIDTH):
            cx, cy = tile_center(c, r)
            y0 = max(0, cy - half_h2 - PLAY_TOP); y1 = min(m_dirt.shape[0], cy + half_h2 - PLAY_TOP)
            x0 = max(0, cx - half_w2); x1 = min(m_dirt.shape[1], cx + half_w2)
            if int(m_dirt[y0:y1, x0:x1].sum()) > (y1 - y0) * (x1 - x0) * 0.3:
                state.dirt[r, c] = True

    return state


# ---- Visualization helper -------------------------------------------------

def render_overlay(frame_rgba: np.ndarray, state: GameState) -> np.ndarray:
    """Draw GameState markers on top of frame for visual validation."""
    out = frame_rgba[..., :3].copy()

    def cross(cx, cy, color, sz=4):
        H, W = out.shape[:2]
        for dx in range(-sz, sz + 1):
            if 0 <= cx + dx < W:
                out[max(0, cy - 1):min(H, cy + 2), cx + dx] = color
        for dy in range(-sz, sz + 1):
            if 0 <= cy + dy < H:
                out[cy + dy, max(0, cx - 1):min(W, cx + 2)] = color

    if state.digger is not None and state.digger.present:
        cx, cy = tile_center(state.digger.col, state.digger.row)
        cross(cx, cy, (255, 0, 255), sz=8)  # magenta cross on digger

    for m in state.monsters:
        cx, cy = tile_center(m.col, m.row)
        cross(cx, cy, (0, 0, 255), sz=6)    # blue cross on monster

    for b in state.bags:
        cx, cy = tile_center(b.col, b.row)
        cross(cx, cy, (255, 128, 0), sz=4)  # orange cross on bag

    for r in range(MHEIGHT):
        for c in range(MWIDTH):
            if state.emeralds[r, c]:
                cx, cy = tile_center(c, r)
                cross(cx, cy, (0, 255, 255), sz=2)  # cyan dot on emerald
    return out
