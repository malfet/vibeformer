"""Boot DIGGER.EXE in DOSBox Pure via libretro, run 20 frames, dump a PPM."""

from pathlib import Path

import _libretro

REPO = Path(__file__).parent.resolve()
CORE = REPO / "vendor" / "dosbox-pure" / "dosbox_pure_libretro.dylib"
GAME = REPO / "data" / "DIGGER.EXE"
SYS_DIR = REPO / "data" / "system"
SAVE_DIR = REPO / "data" / "save"
OUT = REPO / "data" / "frame_20.ppm"

SYS_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

core = _libretro.LibretroCore(str(CORE), str(SYS_DIR), str(SAVE_DIR))
print("Loaded core:", core.get_system_info())

core.load_game(str(GAME))
print("AV info (base_w, base_h, max_w, max_h, fps, sample_rate):", core.get_av_info())

N = 20
for _ in range(N):
    core.run()

frame = core.get_frame()
print(f"Frame after {N} steps: shape={frame.shape}, dtype={frame.dtype}")

# Write a binary PPM (P6). frame is (H, W, 4) RGBA uint8.
h, w = frame.shape[:2]
with OUT.open("wb") as f:
    f.write(f"P6\n{w} {h}\n255\n".encode())
    f.write(frame[:, :, :3].tobytes())
print(f"Wrote {OUT}")
