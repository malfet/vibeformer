"""Tile-patch classifier for Digger.

Per-tile sprite detection. The CGA-class palette is exact (11 unique RGB
triples, no antialiasing), so a nearest-prototype classifier on raw pixel
L2 distance is sufficient -- when sprites match, distance is literally
zero; when they differ, distance is large.

Workflow:
  1. Build a Bank of (label, patch) prototypes by capturing the live game
     at known states (e.g. just-spawned digger position; the top-left
     dirt cell; an emerald tile, etc.) See `bootstrap_from_env`.
  2. classify(patch) returns the nearest-label vote.
  3. game_state.extract_state(frame, classifier=bank) tiles the playfield
     and labels each tile.

Each prototype is the centre 42x37 RGB region of a tile (tile pitch is
640/15 x 368/10 = 42.67 x 36.8). Some sprites move sub-tile; we store
multiple positional prototypes per label and let the nearest-distance
voting pick the right one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from tools.game_state import PLAY_TOP, TILE_H, TILE_W


class TileClass(Enum):
    EMPTY = "empty"          # black tunnel, nothing on top
    DIRT = "dirt"            # undisturbed brown dirt
    EMERALD = "emerald"      # green diamond
    BAG_INTACT = "bag"       # yellow $ on sack
    BAG_SPILLED = "gold"     # broken bag, gold pile
    DIGGER = "digger"        # player sprite (any direction/anim)
    NOBBIN = "nobbin"        # passive monster
    HOBBIN = "hobbin"        # digger monster
    CHERRY = "cherry"        # bonus item


# Window we crop per tile. A bit wider than one tile so sub-tile sprite
# motion still falls inside.
PATCH_W = 44
PATCH_H = 40


def crop_tile_patch(frame_rgb: np.ndarray, col: int, row: int) -> np.ndarray:
    """Extract the (PATCH_H, PATCH_W, 3) uint8 patch centred on tile (col, row)."""
    cx = int((col + 0.5) * TILE_W)
    cy = int(PLAY_TOP + (row + 0.5) * TILE_H)
    y0 = max(0, cy - PATCH_H // 2)
    y1 = min(frame_rgb.shape[0], y0 + PATCH_H)
    x0 = max(0, cx - PATCH_W // 2)
    x1 = min(frame_rgb.shape[1], x0 + PATCH_W)
    out = np.zeros((PATCH_H, PATCH_W, 3), dtype=np.uint8)
    h, w = y1 - y0, x1 - x0
    out[:h, :w] = frame_rgb[y0:y1, x0:x1, :3]
    return out


@dataclass
class PrototypeBank:
    """Holds (label, patch) prototypes for nearest-neighbour classification."""
    protos: list[tuple[TileClass, np.ndarray]] = field(default_factory=list)

    def add(self, label: TileClass, patch: np.ndarray) -> None:
        assert patch.shape == (PATCH_H, PATCH_W, 3) and patch.dtype == np.uint8
        self.protos.append((label, patch.copy()))

    def classify(self, patch: np.ndarray) -> tuple[TileClass, float]:
        """Return (best_label, distance) for the given patch."""
        if not self.protos:
            return TileClass.EMPTY, float("inf")
        best_lbl, best_dist = self.protos[0][0], float("inf")
        for lbl, p in self.protos:
            # Sum of squared per-pixel-channel diffs.
            d = float(np.sum((p.astype(np.int32) - patch.astype(np.int32)) ** 2))
            if d < best_dist:
                best_dist = d
                best_lbl = lbl
        return best_lbl, best_dist

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            labels=np.array([p[0].name for p in self.protos]),
            patches=np.stack([p[1] for p in self.protos], axis=0))

    @classmethod
    def load(cls, path: Path) -> "PrototypeBank":
        d = np.load(path, allow_pickle=False)
        bank = cls()
        for name, patch in zip(d["labels"], d["patches"]):
            bank.protos.append((TileClass[str(name)], patch))
        return bank


# --- Bootstrap helpers -----------------------------------------------------

def harvest_tile_patches(frame_rgba: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    """Return {(col, row): patch} for every tile in the play area of a frame."""
    from tools.game_state import MWIDTH, MHEIGHT
    rgb = frame_rgba[..., :3]
    out = {}
    for r in range(MHEIGHT):
        for c in range(MWIDTH):
            out[(c, r)] = crop_tile_patch(rgb, c, r)
    return out


def bootstrap_from_env() -> PrototypeBank:
    """Run the live emulator, navigate to known states, snapshot prototypes.

    Trustworthy because each prototype is captured at a moment we KNOW the
    tile's class -- spawn frame for the digger, dirt cells in the corners,
    after letting nasties spawn for the monster prototypes, etc.
    """
    import _libretro
    REPO = Path(__file__).parent.resolve()
    core = _libretro.LibretroCore(
        str(REPO / "vendor/dosbox-pure/dosbox_pure_libretro.dylib"),
        str(REPO / "data/system"), str(REPO / "data/save"))
    core.load_game(str(REPO / "data/DIGGER.EXE"))

    def advance(n):
        for _ in range(n): core.run()
    def press(key, n):
        core.set_key(key, True); advance(n); core.set_key(key, False); advance(2)

    advance(20); press(_libretro.RETROK.RETURN, 10); advance(30)
    bank = PrototypeBank()

    # --- Spawn frame: digger is at row 9 (bottom carved tunnel); dirt
    # and emeralds are in their initial level-1 layout.
    f0 = core.get_frame()[..., :3]
    bank.add(TileClass.DIRT,    crop_tile_patch(f0, col=7, row=0))
    bank.add(TileClass.DIRT,    crop_tile_patch(f0, col=2, row=0))
    bank.add(TileClass.EMERALD, crop_tile_patch(f0, col=3, row=2))
    bank.add(TileClass.EMERALD, crop_tile_patch(f0, col=11, row=4))
    bank.add(TileClass.EMPTY,   crop_tile_patch(f0, col=9, row=9))
    bank.add(TileClass.EMPTY,   crop_tile_patch(f0, col=5, row=9))
    bank.add(TileClass.DIGGER,  crop_tile_patch(f0, col=7, row=9))

    # --- Move down + right to dig a tunnel; that gives us 'just-dug-empty'
    # tile prototypes (which may render slightly differently from the
    # original carved bottom tunnel).
    press(_libretro.RETROK.DOWN, 30)
    f1 = core.get_frame()[..., :3]
    bank.add(TileClass.EMPTY, crop_tile_patch(f1, col=7, row=8))   # cell just above digger
    bank.add(TileClass.DIGGER, crop_tile_patch(f1, col=7, row=8))   # digger moved here

    # --- Let nasties spawn. In Digger they appear ~80 frames in, from
    # the top-right corner; first nasty is a NOBBIN.
    advance(120)
    f2 = core.get_frame()[..., :3]
    # Use the CV partial detector to find the largest dark-green blob;
    # that's the nobbin. We crop the surrounding tile.
    from tools.game_state import (_color_mask, _largest_components, PAL_DGREEN,
                            PLAY_TOP, pixel_to_tile)
    play = f2[PLAY_TOP:]
    dgreen_blobs = _largest_components(_color_mask(play, PAL_DGREEN),
                                       min_area=15)
    for area, (cy, cx) in dgreen_blobs[:3]:
        col, row = pixel_to_tile(cx, cy + PLAY_TOP)
        bank.add(TileClass.NOBBIN, crop_tile_patch(f2, col=col, row=row))

    return bank


def label_image(frame_rgba: np.ndarray, bank: PrototypeBank) -> np.ndarray:
    """Classify every tile of the frame; return (MHEIGHT, MWIDTH) of TileClass enum values (by name)."""
    from tools.game_state import MWIDTH, MHEIGHT
    out = np.empty((MHEIGHT, MWIDTH), dtype=object)
    rgb = frame_rgba[..., :3]
    for r in range(MHEIGHT):
        for c in range(MWIDTH):
            patch = crop_tile_patch(rgb, c, r)
            lbl, _ = bank.classify(patch)
            out[r, c] = lbl
    return out
