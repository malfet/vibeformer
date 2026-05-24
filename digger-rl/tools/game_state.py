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
PAL_DCYAN     = (0,   130, 130)  # title-screen text

DIRT_COLORS = {PAL_DIRT_A, PAL_DIRT_B}
SPRITE_COLORS = {PAL_GREEN, PAL_DGREEN, PAL_YELLOW, PAL_RED}


# ---- Bag sprite ----------------------------------------------------------
# Intact bag: a 32x30 pure-black sprite (circular outline + solid "$")
# drawn over dirt. The bag uses NO colour of its own -- the brown sack we
# see is the dirt showing through the open parts of the outline, and the
# "$" is just black carved out of the dirt. So bag detection is an exact
# template match against the framebuffer's black mask.
_BAG_TEMPLATE_STR = (
    '00000000000111111111100000000000',
    '00000000001000000000010000000000',
    '00000000001000000000010000000000',
    '00000000001100000000110000000000',
    '00000000000011000011000000000000',
    '00000000000011000011000000000000',
    '00000000001100000000110000000000',
    '00000000110000000000001100000000',
    '00000001000000000000000010000000',
    '00000010000000111100000001000000',
    '00000100000000111100000000100000',
    '00001000001111111111100000010000',
    '00010000111111111111111000001000',
    '00100001111111111111111110000100',
    '01000011111000111100111110000010',
    '01000011110000111100000000000010',
    '10000011111111111100000000000001',
    '10000001111111111111111000000001',
    '10000000011111111111111110000001',
    '10000000000000111111111111000001',
    '10000000000000111100001111000001',
    '10000001111100111100011111000001',
    '01000001111111111111111110000010',
    '01000000111111111111111100000010',
    '00100000000111111111100000000100',
    '00010000000000111100000000001000',
    '00001100000000111100000000110000',
    '00000011000000000000000011000000',
    '00000000111110000001111100000000',
    '00000000000001111110000000000000',
)
BAG_TEMPLATE: np.ndarray = np.array(
    [[c == '1' for c in row] for row in _BAG_TEMPLATE_STR], dtype=bool)
BAG_TPL_H, BAG_TPL_W = BAG_TEMPLATE.shape           # 30, 32
# Offset from the template's top-left to the bag's centre pixel.
BAG_CENTER_DY, BAG_CENTER_DX = BAG_TPL_H // 2, BAG_TPL_W // 2


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


def _find_bags_by_template(m_blk_play: np.ndarray) -> list[tuple[int, int]]:
    """Locate bags by exact template match against the play-area black mask.

    Returns a list of (col, row) tile coordinates -- one per bag -- whose
    centres fall on the corresponding tile centre. The template is matched
    exactly (black-where-template-says-black AND not-black-elsewhere), so
    tunnels and other large black regions do not false-match -- they fail
    the not-black check on the bag's internal-dirt pixels.

    Implementation: two-stage vectorised filter. The cheap stage AND-reduces
    one row of the bag's central "$" spine (must-be-black) with one row of
    the bag's interior gap (must-NOT-be-black). That rejects almost every
    non-bag pixel in the frame, including all of the tunnel interiors which
    are entirely black. The few survivors get a full template check.
    """
    H, W = m_blk_play.shape
    if H < BAG_TPL_H or W < BAG_TPL_W:
        return []

    TH, TW = BAG_TPL_H, BAG_TPL_W
    out_h, out_w = H - TH + 1, W - TW + 1  # valid top-left positions

    # Cheap pre-filter at template positions:
    #   (12, 14..17) = the $ spine, must be black (4 pixels)
    #   (14, 11..13) = the dirt gap just left of the spine, must NOT be black
    # In a tunnel, the "gap" positions are also black, so the filter rejects.
    # In dirt, the "spine" positions are not black, so the filter rejects.
    # Only an aligned bag passes both rows.
    spine = (m_blk_play[12:12+out_h, 14:14+out_w]
             & m_blk_play[12:12+out_h, 15:15+out_w]
             & m_blk_play[12:12+out_h, 16:16+out_w]
             & m_blk_play[12:12+out_h, 17:17+out_w])
    gap   = ~(m_blk_play[14:14+out_h, 11:11+out_w]
              | m_blk_play[14:14+out_h, 12:12+out_w]
              | m_blk_play[14:14+out_h, 13:13+out_w])
    candidates = spine & gap
    cand_ys, cand_xs = np.where(candidates)

    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for ty, tx in zip(cand_ys.tolist(), cand_xs.tolist()):
        if (ty, tx) in seen:
            continue
        if not np.array_equal(m_blk_play[ty:ty + TH, tx:tx + TW], BAG_TEMPLATE):
            continue
        seen.add((ty, tx))
        sprite_cy = ty + BAG_CENTER_DY
        sprite_cx = tx + BAG_CENTER_DX
        col, row = pixel_to_tile(sprite_cx, sprite_cy + PLAY_TOP)
        out.append((col, row))
    return out


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

    # --- digger: largest connected RED blob. Player sprite is dominated
    # by red; monster eyes/feet are also red but smaller (a few px each).
    red_blobs = _largest_components(m_red, min_area=30)
    if red_blobs:
        _, (cy, cx) = red_blobs[0]
        col, row = pixel_to_tile(cx, cy + PLAY_TOP)
        state.digger = DiggerPos(col=col, row=row, alive=True)

    # --- bags: intact bags are an exact-pixel template (circular black
    # outline + solid "$" carved from dirt). Use sprite-exact matching
    # against the framebuffer's black mask; see _find_bags_by_template.
    bag_seen: set[tuple[int, int]] = set()
    for col, row in _find_bags_by_template(m_blk):
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


# ---- Fast vectorized extractor --------------------------------------------

_TILE_INDEX_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _tile_index_for_play(play_h: int, play_w: int) -> np.ndarray:
    """Build a (play_h, play_w) int32 map: each pixel -> linear tile index 0..149.

    Cached because the play-area dimensions never change between frames.
    """
    key = (play_h, play_w)
    cached = _TILE_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    ys = np.arange(play_h, dtype=np.int32)
    xs = np.arange(play_w, dtype=np.int32)
    tile_row = (ys * MHEIGHT // play_h)            # (H,) in 0..MHEIGHT-1
    tile_col = (xs * MWIDTH // play_w)             # (W,) in 0..MWIDTH-1
    idx = tile_row[:, None] * MWIDTH + tile_col[None, :]
    _TILE_INDEX_CACHE[key] = idx
    return idx


def extract_state_fast(frame_rgba: np.ndarray) -> GameState:
    """Vectorised CV extractor, ~10x faster than extract_state.

    No Python loops over pixels and no connected-components BFS. Instead:
      1. Compute 6 boolean colour masks against the exact palette
         (one elementwise compare per colour).
      2. Use a precomputed (play_h, play_w) -> tile_index lookup and
         np.bincount to get per-tile per-colour counts in one pass.
      3. Classify each tile from those count vectors.

    Loses the sub-tile centroid precision of the BFS version but for a
    tile-aligned game like Digger this is fine.
    """
    assert frame_rgba.shape == (FRAME_H, FRAME_W, 4), \
        f"expected 400x640 RGBA, got {frame_rgba.shape}"
    rgb = frame_rgba[..., :3]
    play = rgb[PLAY_TOP:PLAY_BOTTOM]
    H, W = play.shape[:2]
    tile_idx_flat = _tile_index_for_play(H, W).ravel()
    n_tiles = MWIDTH * MHEIGHT

    # All boolean masks computed in one sweep through `play`.
    pr = play[..., 0]; pg = play[..., 1]; pb = play[..., 2]
    m_red   = (pr == PAL_RED[0])   & (pg == PAL_RED[1])   & (pb == PAL_RED[2])
    m_grn   = (pr == PAL_GREEN[0]) & (pg == PAL_GREEN[1]) & (pb == PAL_GREEN[2])
    m_dgrn  = (pr == PAL_DGREEN[0]) & (pg == PAL_DGREEN[1]) & (pb == PAL_DGREEN[2])
    m_yel   = (pr == PAL_YELLOW[0]) & (pg == PAL_YELLOW[1]) & (pb == PAL_YELLOW[2])
    m_blk   = (pr == 0) & (pg == 0) & (pb == 0)
    m_dirt  = (((pr == PAL_DIRT_A[0]) & (pg == PAL_DIRT_A[1]) & (pb == PAL_DIRT_A[2])) |
               ((pr == PAL_DIRT_B[0]) & (pg == PAL_DIRT_B[1]) & (pb == PAL_DIRT_B[2])))

    def per_tile(mask: np.ndarray) -> np.ndarray:
        return np.bincount(tile_idx_flat[mask.ravel()],
                           minlength=n_tiles).reshape(MHEIGHT, MWIDTH)

    red_c   = per_tile(m_red)
    grn_c   = per_tile(m_grn)
    dgrn_c  = per_tile(m_dgrn)
    yel_c   = per_tile(m_yel)
    dirt_c  = per_tile(m_dirt)

    state = GameState()

    # Monsters first so we can excise them from the digger search.
    # Nobbin signature: large dark-green body + yellow head + a few red
    # eye/foot pixels. Digger signature: bright-red body dominates, so
    # red_c will be much higher in the digger tile than the floor of
    # 3-5 red eye pixels a nobbin contributes.
    monster_mask = ((dgrn_c >= 20) & (yel_c >= 6) &
                    (red_c >= 2) & (red_c < 25))
    # Digger: tile with the most red pixels among NON-monster tiles.
    # The exclusion avoids the corner case where a nobbin's red eye-
    # pixels would otherwise win argmax during a frame where the
    # digger sprite is mid-animation with reduced red mass.
    red_for_digger = red_c.copy()
    red_for_digger[monster_mask] = 0
    flat_idx = int(red_for_digger.argmax())
    if red_for_digger.flat[flat_idx] >= 30:
        d_row, d_col = divmod(flat_idx, MWIDTH)
        state.digger = DiggerPos(col=int(d_col), row=int(d_row), alive=True)

    # Emeralds: bright-green pixels in tile, no red (avoids the digger
    # body and monster eyes contaminating).
    state.emeralds = (grn_c >= 25) & (red_c < 4)

    # Dirt: enough brown pixels in tile to call it "covered". Tile area
    # ~1500 px; >300 dirt pixels = clearly undisturbed.
    state.dirt = dirt_c > 300

    # Monsters: emit positions from the mask we computed above. Exclude
    # the digger tile in case both signatures fire there.
    if state.digger is not None:
        monster_mask[state.digger.row, state.digger.col] = False
    for r, c in zip(*np.where(monster_mask)):
        state.monsters.append(MonsterPos(col=int(c), row=int(r), is_hobbin=False))

    # Bags: intact bags are a pure-black sprite (circular outline + "$")
    # over dirt. Exact-match the template against the black mask.
    bag_seen: set[tuple[int, int]] = set()
    for col, row in _find_bags_by_template(m_blk):
        if (col, row) in bag_seen:
            continue
        bag_seen.add((col, row))
        state.bags.append(BagPos(col=col, row=row))

    return state


# ---- Visualization helper -------------------------------------------------

def render_overlay(frame_rgba: np.ndarray, state: GameState,
                    show_digger: bool = False,
                    show_monsters: bool = True,
                    show_bags: bool = True,
                    show_emeralds: bool = False) -> np.ndarray:
    """Draw GameState markers on top of frame for visual validation.

    Defaults: only mark monsters (blue) and bags (orange) -- the digger
    sprite is already obvious to a human, and emerald markers tend to
    clutter the dense level-1 layout. Pass show_emeralds=True / show_
    digger=True to enable them.
    """
    out = frame_rgba[..., :3].copy()
    H, W = out.shape[:2]

    def cross(cx, cy, color, sz):
        # Single-pixel-thick cross of total span 2*sz+1.
        for dx in range(-sz, sz + 1):
            if 0 <= cx + dx < W and 0 <= cy < H:
                out[cy, cx + dx] = color
        for dy in range(-sz, sz + 1):
            if 0 <= cy + dy < H and 0 <= cx < W:
                out[cy + dy, cx] = color

    if show_digger and state.digger is not None and state.digger.present:
        cx, cy = tile_center(state.digger.col, state.digger.row)
        cross(cx, cy, (255, 0, 255), sz=3)
    if show_monsters:
        for m in state.monsters:
            cx, cy = tile_center(m.col, m.row)
            cross(cx, cy, (0, 0, 255), sz=4)
    if show_bags:
        for b in state.bags:
            cx, cy = tile_center(b.col, b.row)
            cross(cx, cy, (255, 128, 0), sz=3)
    if show_emeralds:
        for r in range(MHEIGHT):
            for c in range(MWIDTH):
                if state.emeralds[r, c]:
                    cx, cy = tile_center(c, r)
                    cross(cx, cy, (0, 255, 255), sz=1)
    return out
