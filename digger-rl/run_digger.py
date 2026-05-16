"""Boot DIGGER.EXE in DOSBox Pure via libretro.

Default: run N frames headlessly and dump a PPM screenshot.
--live:  open a matplotlib window, run continuously at ~70 fps emulator time,
         and forward keyboard events from the window into the emulator.
--record-playtrace PATH: while in --live, snapshot at frame-skip cadence and
         save (frames, actions, rewards, lives) to PATH.npz for behavioural
         cloning warmup in train_ppo.py.
"""

import argparse
import struct
import time
from pathlib import Path

import numpy as np

import _libretro
from digger_env import DiggerEnv

REPO = Path(__file__).parent.resolve()
CORE = REPO / "vendor" / "dosbox-pure" / "dosbox_pure_libretro.dylib"
GAME = REPO / "data" / "DIGGER.EXE"
SYS_DIR = REPO / "data" / "system"
SAVE_DIR = REPO / "data" / "save"

# Offsets inside memory region 0 (the DOS GAME segment as exposed by DOSBox
# Pure's SET_MEMORY_MAPS). Located by diffing RAM snapshots across gameplay:
# eating emeralds for the score, observing deaths for lives. Lives goes
# 3 -> 2 -> 1 -> 0 in 1P mode; 0 == game over.
SCORE_OFFSET = 0x282E0  # int32 LE
LIVES_OFFSET = 0x259F2  # uint8


def read_score(core) -> int:
    """Read the current 1P score as a signed 32-bit LE integer."""
    region = core.read_memory_region(0)
    return struct.unpack_from("<i", region, SCORE_OFFSET)[0]


def read_lives(core) -> int:
    """Read the remaining-lives count (0 == game over once gameplay started)."""
    region = core.read_memory_region(0)
    return region[LIVES_OFFSET]

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
    "tab":       _libretro.RETROK.F1,
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


# Map a libretro arrow key to the DiggerEnv discrete action that represents it.
_ARROW_TO_ACTION = {
    _libretro.RETROK.LEFT:  DiggerEnv.LEFT,
    _libretro.RETROK.RIGHT: DiggerEnv.RIGHT,
    _libretro.RETROK.UP:    DiggerEnv.UP,
    _libretro.RETROK.DOWN:  DiggerEnv.DOWN,
}


def dominant_action(held_order: list[int]) -> int:
    """Collapse the human's current multi-key state into one discrete action.

    DiggerEnv's action space is exclusive (NOOP / 4 arrows / FIRE) but a human
    naturally holds LEFT and taps F1 at the same time. Priority: FIRE wins if
    F1 is held; otherwise the most-recently-pressed arrow; otherwise NOOP.
    """
    if _libretro.RETROK.F1 in held_order:
        return DiggerEnv.FIRE
    for k in reversed(held_order):
        if k in _ARROW_TO_ACTION:
            return _ARROW_TO_ACTION[k]
    return DiggerEnv.NOOP


def preprocess_uint8(frame_rgba: np.ndarray, size: int = 84) -> np.ndarray:
    """raw (H, W, 4) RGBA uint8 -> (size, size) uint8 grayscale.

    Byte-identical (modulo /255 normalisation at load) to train_ppo.preprocess
    so trace frames match what the policy sees during PPO. Imports torch
    lazily so the headless code path doesn't pay the import cost.
    """
    import torch
    import torch.nn.functional as F
    rgb = frame_rgba[..., :3].astype(np.float32) * (1.0 / 255.0)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    t = torch.from_numpy(gray)[None, None]
    t = F.interpolate(t, size=(size, size), mode="area")
    return (t[0, 0].numpy() * 255.0).clip(0, 255).astype(np.uint8)


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


def run_live(core, record_path: Path | None = None,
             frame_skip: int = 4, obs_size: int = 84) -> None:
    import matplotlib.pyplot as plt

    target_fps = core.get_av_info()[4] or 70.0
    target_dt = 1.0 / target_fps

    core.run()  # populate first frame
    fig, ax = plt.subplots()
    ax.set_axis_off()
    img = ax.imshow(core.get_frame())
    fig.tight_layout(pad=0)

    # Press-order list so dominant_action() can pick the most-recent arrow
    # when several are held at once.
    held_order: list[int] = []

    # --- recording state. Snapshot every frame_skip emulator frames.
    rec_frames: list[np.ndarray] = []
    rec_actions: list[int] = []
    rec_rewards: list[float] = []
    rec_scores: list[int] = []
    rec_lives: list[int] = []
    rec_resets: list[bool] = []
    sub_counter = 0
    prev_score = read_score(core) if record_path is not None else 0
    window_reward = 0.0
    pending_reset = False
    seen_alive = False
    notified_game_over = False

    def reset_emulator():
        """Soft-reset the libretro core so the user respawns at level 1.

        Uses retro_reset (a virtual power-cycle) rather than tearing down
        and reconstructing LibretroCore, which would have to fight the
        per-process singleton and any lingering refs from this very
        callback's closure. Recording continues across the reset; the next
        snapshot is flagged in rec_resets so the trainer treats it as an
        episode boundary for V.
        """
        nonlocal prev_score, sub_counter, window_reward
        nonlocal pending_reset, seen_alive, notified_game_over
        print("R: resetting emulator to title screen...", flush=True)
        for k in list(held_order):
            core.set_key(k, False)
        held_order.clear()
        core.reset()
        advance_to_gameplay(core)
        img.set_data(core.get_frame())
        prev_score = read_score(core)
        sub_counter = 0
        window_reward = 0.0
        seen_alive = read_lives(core) > 0
        notified_game_over = False
        pending_reset = True
        print("Reset complete.", flush=True)

    def on_press(event):
        if event.key and event.key.lower() == "r":
            reset_emulator()
            return
        retro = matplotlib_key_to_retro(event.key)
        if retro is not None and retro not in held_order:
            core.set_key(retro, True)
            held_order.append(retro)

    def on_release(event):
        retro = matplotlib_key_to_retro(event.key)
        if retro is not None and retro in held_order:
            core.set_key(retro, False)
            held_order.remove(retro)

    fig.canvas.mpl_connect("key_press_event", on_press)
    fig.canvas.mpl_connect("key_release_event", on_release)

    plt.ion()
    plt.show()
    print("Live mode. Arrows + Enter + Space drive the game. "
          "R restarts at level 1. Close the window to quit.")
    if record_path is not None:
        print(f"Recording trace to {record_path} "
              f"(frame_skip={frame_skip}, obs_size={obs_size}).")

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
            if record_path is not None:
                sub_counter += 1
                score_now = read_score(core)
                window_reward += float(score_now - prev_score)
                prev_score = score_now
                if sub_counter >= frame_skip:
                    rec_frames.append(preprocess_uint8(core.get_frame(), obs_size))
                    rec_actions.append(dominant_action(held_order))
                    rec_rewards.append(window_reward)
                    rec_scores.append(score_now)
                    rec_lives.append(int(read_lives(core)))
                    rec_resets.append(pending_reset)
                    pending_reset = False
                    sub_counter = 0
                    window_reward = 0.0
        frame_no += steps

        img.set_data(core.get_frame())
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        lives = read_lives(core)
        if lives > 0:
            seen_alive = True
            notified_game_over = False
        elif seen_alive and not notified_game_over:
            print(f"Game over (final score {read_score(core)}). "
                  f"Press R to restart, close window to save & quit.",
                  flush=True)
            notified_game_over = True

        if elapsed > 0:
            fps_ema = 0.9 * fps_ema + 0.1 * (steps / elapsed)
        if frame_no % 30 == 0:
            rec_tag = f" -- rec {len(rec_frames)}" if record_path is not None else ""
            fig.canvas.manager.set_window_title(
                f"DIGGER -- score {read_score(core):6d} -- lives {lives} -- "
                f"emu {fps_ema:5.1f} fps -- frame {frame_no}{rec_tag}"
            )

        # If we got ahead of real time, sleep the rest of the budget.
        slack = target_dt - (time.monotonic() - now)
        if slack > 0:
            time.sleep(slack)

    core.clear_keys()
    plt.close(fig)

    if record_path is not None and rec_frames:
        save_trace(record_path, rec_frames, rec_actions, rec_rewards,
                   rec_scores, rec_lives, rec_resets, frame_skip, obs_size)
    elif record_path is not None:
        print(f"No frames recorded; skipping save to {record_path}.")


def save_trace(path: Path, frames: list[np.ndarray], actions: list[int],
               rewards: list[float], scores: list[int], lives: list[int],
               resets: list[bool], frame_skip: int, obs_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frames=np.stack(frames, axis=0).astype(np.uint8),
        actions=np.array(actions, dtype=np.uint8),
        raw_rewards=np.array(rewards, dtype=np.float32),
        scores=np.array(scores, dtype=np.int32),
        lives=np.array(lives, dtype=np.uint8),
        resets=np.array(resets, dtype=bool),
        frame_skip=np.int32(frame_skip),
        obs_size=np.int32(obs_size),
    )
    final_score = scores[-1] if scores else 0
    n_resets = sum(resets)
    print(f"Wrote {len(frames)} samples (final score {final_score}, "
          f"{n_resets} R-resets) to {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--live", action="store_true",
                   help="open a matplotlib window and run interactively")
    p.add_argument("--frames", type=int, default=20,
                   help="frames to run in headless mode (default: 20)")
    p.add_argument("--record-playtrace", type=Path, default=None,
                   help="while in --live, snapshot every frame_skip emulator "
                        "frames and save to this .npz path on exit")
    p.add_argument("--frame-skip", type=int, default=4,
                   help="emulator frames per recorded sample (must match the "
                        "frame_skip used in train_ppo; default 4)")
    p.add_argument("--obs-size", type=int, default=84,
                   help="recorded frame resolution; matches train_ppo obs_size")
    args = p.parse_args()

    if args.record_playtrace is not None and not args.live:
        p.error("--record-playtrace requires --live")

    core = init_core()
    advance_to_gameplay(core)
    if args.live:
        run_live(core, record_path=args.record_playtrace,
                 frame_skip=args.frame_skip, obs_size=args.obs_size)
    else:
        run_headless(core, args.frames)


if __name__ == "__main__":
    main()
