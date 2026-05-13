"""Boot DIGGER.EXE in DOSBox Pure via libretro.

Default: run N frames headlessly and dump a PPM screenshot.
--live:  open a matplotlib window, run continuously at ~70 fps emulator time,
         and forward keyboard events from the window into the emulator.
"""

import argparse
import struct
import time
from pathlib import Path

import _libretro

REPO = Path(__file__).parent.resolve()
CORE = REPO / "vendor" / "dosbox-pure" / "dosbox_pure_libretro.dylib"
GAME = REPO / "data" / "DIGGER.EXE"
SYS_DIR = REPO / "data" / "system"
SAVE_DIR = REPO / "data" / "save"

# Offset of the int32 score variable inside memory region 0 (the DOS GAME
# segment as exposed by DOSBox Pure's SET_MEMORY_MAPS). Located by diffing
# RAM snapshots before/after eating emeralds; see find_score.py.
SCORE_OFFSET = 0x282E0


def read_score(core) -> int:
    """Read the current 1P score as a signed 32-bit LE integer."""
    region = core.read_memory_region(0)
    return struct.unpack_from("<i", region, SCORE_OFFSET)[0]

# Map matplotlib key-event names to libretro RETROK_* values. Single printable
# characters are handled separately (ASCII matches RETROK for [a-z] and a few
# others) so we only list the named/special keys here.
_NAMED_KEYS = {
    "left":      _libretro.RETROK.LEFT,
    "right":     _libretro.RETROK.RIGHT,
    "up":        _libretro.RETROK.UP,
    "down":      _libretro.RETROK.DOWN,
    "enter":     _libretro.RETROK.RETURN,
    "return":    _libretro.RETROK.RETURN,
    " ":         _libretro.RETROK.SPACE,
    "space":     _libretro.RETROK.SPACE,
    "escape":    _libretro.RETROK.ESCAPE,
    "tab":       _libretro.RETROK.TAB,
    "backspace": _libretro.RETROK.BACKSPACE,
    "shift":     _libretro.RETROK.LSHIFT,
    "control":   _libretro.RETROK.LCTRL,
    "alt":       _libretro.RETROK.LALT,
}
for _i in range(1, 13):
    _NAMED_KEYS[f"f{_i}"] = getattr(_libretro.RETROK, f"F{_i}")


def matplotlib_key_to_retro(key: str | None) -> int | None:
    if key is None:
        return None
    k = key.lower()
    if k in _NAMED_KEYS:
        return _NAMED_KEYS[k]
    if len(k) == 1 and "a" <= k <= "z":
        return getattr(_libretro.RETROK, k.upper())
    if len(k) == 1 and "0" <= k <= "9":
        return getattr(_libretro.RETROK, f"N{k}")
    return None


def init_core() -> "_libretro.LibretroCore":
    SYS_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    core = _libretro.LibretroCore(str(CORE), str(SYS_DIR), str(SAVE_DIR))
    core.load_game(str(GAME))
    print("Loaded core:", core.get_system_info())
    print("AV info (base_w, base_h, max_w, max_h, fps, sample_rate):",
          core.get_av_info())
    return core


def advance_to_gameplay(core, title_frames: int = 20, hold_frames: int = 10, advance_frames: int = 20) -> None:
    """Run past the title screen so the agent starts in actual gameplay.

    DIGGER.EXE boots into a title/menu that waits on a keypress. Pressing
    RETURN starts a one-player game. We let the title render for a few
    frames, then hold RETURN across `hold_frames` to make sure the press is
    seen by the keyboard interrupt handler, then release, and finally run
    `advance_frames` more so the screen transition into the level settles.
    """
    for _ in range(title_frames):
        core.run()
    core.set_key(_libretro.RETROK.RETURN, True)
    for _ in range(hold_frames):
        core.run()
    core.set_key(_libretro.RETROK.RETURN, False)
    for _ in range(advance_frames):
        core.run()


def run_headless(core, frames: int) -> None:
    out = REPO / "data" / f"frame_{frames}.ppm"
    for _ in range(frames):
        core.run()
    frame = core.get_frame()
    print(f"Frame after {frames} steps: shape={frame.shape}, dtype={frame.dtype}")
    h, w = frame.shape[:2]
    with out.open("wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(frame[:, :, :3].tobytes())
    print(f"Wrote {out}")


def run_live(core) -> None:
    import matplotlib.pyplot as plt

    target_fps = core.get_av_info()[4] or 70.0
    target_dt = 1.0 / target_fps

    core.run()  # populate first frame
    fig, ax = plt.subplots()
    ax.set_axis_off()
    img = ax.imshow(core.get_frame())
    fig.tight_layout(pad=0)

    held: set[int] = set()

    def on_press(event):
        retro = matplotlib_key_to_retro(event.key)
        if retro is not None and retro not in held:
            core.set_key(retro, True)
            held.add(retro)

    def on_release(event):
        retro = matplotlib_key_to_retro(event.key)
        if retro is not None and retro in held:
            core.set_key(retro, False)
            held.discard(retro)

    fig.canvas.mpl_connect("key_press_event", on_press)
    fig.canvas.mpl_connect("key_release_event", on_release)

    plt.ion()
    plt.show()
    print("Live mode. Arrows + Enter + Space drive the game. Close the window to quit.")

    last = time.monotonic()
    fps_ema = target_fps
    frame_no = 0
    while plt.fignum_exists(fig.number):
        now = time.monotonic()
        elapsed = now - last
        last = now

        # Catch up real-time emulation if matplotlib falls behind. Cap at 5
        # frames per render to avoid runaway after a stall.
        steps = max(1, min(5, int(round(elapsed / target_dt))))
        for _ in range(steps):
            core.run()
        frame_no += steps

        img.set_data(core.get_frame())
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        if elapsed > 0:
            fps_ema = 0.9 * fps_ema + 0.1 * (steps / elapsed)
        if frame_no % 30 == 0:
            fig.canvas.manager.set_window_title(
                f"DIGGER -- score {read_score(core):6d} -- "
                f"emu {fps_ema:5.1f} fps -- frame {frame_no}"
            )

        # If we got ahead of real time, sleep the rest of the budget.
        slack = target_dt - (time.monotonic() - now)
        if slack > 0:
            time.sleep(slack)

    core.clear_keys()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--live", action="store_true",
                   help="open a matplotlib window and run interactively")
    p.add_argument("--frames", type=int, default=20,
                   help="frames to run in headless mode (default: 20)")
    args = p.parse_args()

    core = init_core()
    advance_to_gameplay(core)
    if args.live:
        run_live(core)
    else:
        run_headless(core, args.frames)


if __name__ == "__main__":
    main()
