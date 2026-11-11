"""Frame viewer shared by run_digger.py, run_ppo.py and the heuristic agent.

Two backends behind one interface:

  KittyDisplay  draws frames inline in the terminal with the Kitty graphics
                protocol (Kitty, Ghostty, WezTerm) and reads keys with the
                Kitty keyboard protocol, which also reports key releases.
                Works over SSH; no GUI toolkit needed.
  MplDisplay    a matplotlib window (the original viewer).

    with open_display("auto", title="DIGGER") as d:
        while d.is_open:
            d.show(frame)                  # (H, W, 3|4) uint8
            d.set_title("score 150")
            for key, pressed in d.poll():  # matplotlib key names
                ...

"auto" picks Kitty when the terminal advertises support and stdin/stdout
are a TTY, else matplotlib. Key names follow matplotlib ("left", "enter",
" ", "f1", "a"), so callers map keys the same way for both backends.
Ctrl+C or Ctrl+Q closes the Kitty viewer; closing the window closes the
matplotlib one. While the Kitty viewer is up, print() output goes to a
message line under the frame, and all of it is replayed to the terminal on
close.
"""

from __future__ import annotations

import base64
import fcntl
import os
import re
import select
import struct
import sys
import termios
import zlib

import numpy as np


def kitty_supported() -> bool:
    """Best-effort check, from the environment, for Kitty graphics support."""
    if not (sys.stdout.isatty() and sys.stdin.isatty()):
        return False
    term = os.environ.get("TERM", "")
    prog = os.environ.get("TERM_PROGRAM", "").lower()
    return ("KITTY_WINDOW_ID" in os.environ or "kitty" in term
            or "ghostty" in term or prog in ("ghostty", "wezterm"))


def open_display(kind: str = "auto", title: str = "DIGGER"):
    if kind == "auto":
        kind = "kitty" if kitty_supported() else "mpl"
    if kind == "kitty":
        return KittyDisplay(title)
    if kind == "mpl":
        return MplDisplay(title)
    raise ValueError(f"unknown display {kind!r} (auto | kitty | mpl)")


class _Display:
    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---- Kitty ------------------------------------------------------------------

_CHUNK = 4096
# CSI [code[:alt...]] [; mods[:event]] [; text] final -- kitty keyboard
# protocol; also matches legacy arrows (ESC [ A) and ESC [ 1 ; 5 A.
_CSI_KEY = re.compile(
    rb"\x1b\[(\d*)(?::\d*)*(?:;(\d*)(?::(\d+))?)?(?:;[\d:]*)?([u~ABCDHFPQRS])")
_LETTER_KEYS = {b"A": "up", b"B": "down", b"C": "right", b"D": "left",
                b"H": "home", b"F": "end", b"P": "f1", b"Q": "f2",
                b"R": "f3", b"S": "f4"}
_TILDE_KEYS = {2: "insert", 3: "delete", 5: "pageup", 6: "pagedown",
               13: "f3", 15: "f5", 17: "f6", 18: "f7", 19: "f8", 20: "f9",
               21: "f10", 23: "f11", 24: "f12"}
_CODEPOINT_KEYS = {13: "enter", 27: "escape", 9: "tab", 127: "backspace",
                   32: " ", 57441: "shift", 57442: "control", 57443: "alt",
                   57447: "shift", 57448: "control", 57449: "alt"}
_MOD_CTRL = 4


class _MessageLine:
    """stdout stand-in: shows the latest line under the frame, keeps all."""

    def __init__(self, display: "KittyDisplay"):
        self._display = display
        self.lines: list[str] = []
        self._partial = ""

    def write(self, text: str) -> int:
        self._partial += text
        *done, self._partial = self._partial.split("\n")
        self.lines.extend(done)
        if done and done[-1]:
            self._display._message(done[-1])
        return len(text)

    def flush(self) -> None:
        pass


class KittyDisplay(_Display):
    """Inline terminal viewer: Kitty graphics + Kitty keyboard protocol."""

    def __init__(self, title: str = "DIGGER"):
        self._out = sys.stdout.buffer
        self._fd = sys.stdin.fileno()
        self._saved_tty = termios.tcgetattr(self._fd)
        attrs = termios.tcgetattr(self._fd)
        attrs[3] &= ~(termios.ICANON | termios.ECHO | termios.ISIG)
        attrs[6][termios.VMIN], attrs[6][termios.VTIME] = 0, 0
        termios.tcsetattr(self._fd, termios.TCSANOW, attrs)
        self._buf = b""
        self._title = title
        self._rows = 24
        self.is_open = True
        self._stdout = sys.stdout
        self._messages = _MessageLine(self)
        sys.stdout = self._messages
        # Alt screen, hidden cursor, keyboard flags 1|2|8: disambiguate,
        # report press/repeat/release, report every key as an escape code.
        self._write(b"\x1b[?1049h\x1b[?25l\x1b[2J\x1b[>11u")

    def _write(self, data: bytes) -> None:
        self._out.write(data)
        self._out.flush()

    def _fit(self, h: int, w: int) -> tuple[int, int]:
        """Columns x rows to scale an h x w image into, below the title."""
        try:
            rows, cols, xpix, ypix = struct.unpack(
                "HHHH", fcntl.ioctl(1, termios.TIOCGWINSZ, b"\0" * 8))
        except OSError:
            rows, cols, xpix, ypix = 24, 80, 0, 0
        if not rows or not cols:
            rows, cols = 24, 80
        self._rows = rows
        cw = xpix / cols if xpix else 8.0
        ch = ypix / rows if ypix else 16.0
        avail_rows = max(1, rows - 3)    # title, gap, message line
        c = cols
        r = int(round(c * cw * h / w / ch))
        if r > avail_rows:
            r = avail_rows
            c = max(1, int(round(r * ch * w / h / cw)))
        return c, r

    def show(self, frame: np.ndarray) -> None:
        if not self.is_open:
            return
        rgb = np.ascontiguousarray(frame[..., :3], dtype=np.uint8)
        h, w = rgb.shape[:2]
        c, r = self._fit(h, w)
        payload = base64.standard_b64encode(zlib.compress(rgb.tobytes(), 1))
        parts = [b"\x1b[2;1H"]
        for i in range(0, len(payload), _CHUNK):
            more = int(i + _CHUNK < len(payload))
            if i == 0:
                ctl = (f"a=T,f=24,o=z,s={w},v={h},i=1,p=1,c={c},r={r},"
                       f"C=1,q=2,m={more}")
            else:
                ctl = f"m={more}"
            parts.append(b"\x1b_G" + ctl.encode() + b";"
                         + payload[i:i + _CHUNK] + b"\x1b\\")
        self._write(b"".join(parts))

    def set_title(self, text: str) -> None:
        if self.is_open:
            line = f"{self._title} -- {text}"
            self._write(b"\x1b[1;1H\x1b[2K" + line.encode())

    def _message(self, text: str) -> None:
        if self.is_open:
            self._write(f"\x1b[{self._rows};1H\x1b[2K".encode()
                        + text[:400].encode())

    def poll(self) -> list[tuple[str, bool]]:
        """Key events since the last call: [(name, pressed), ...]."""
        while select.select([self._fd], [], [], 0)[0]:
            chunk = os.read(self._fd, 4096)
            if not chunk:
                break
            self._buf += chunk
        events: list[tuple[str, bool]] = []
        buf = self._buf
        while buf:
            if buf.startswith(b"\x1b"):
                m = _CSI_KEY.match(buf)
                if m is None:
                    if len(buf) < 32 and buf.startswith(b"\x1b["):
                        break                  # incomplete sequence
                    buf = buf[1:]
                    events.append(("escape", True))
                    events.append(("escape", False))
                    continue
                buf = buf[m.end():]
                ev = self._decode(m)
                if ev is not None:
                    events.append(ev)
            else:
                ch, buf = buf[:1], buf[1:]     # legacy byte: no release
                if ch in (b"\x03", b"\x11"):
                    self.is_open = False
                    continue
                key = _CODEPOINT_KEYS.get(ch[0], ch.decode("latin-1").lower())
                events.append((key, True))
                events.append((key, False))
        self._buf = buf
        return events

    def _decode(self, m: re.Match) -> tuple[str, bool] | None:
        code = int(m.group(1) or 1)
        mods = int(m.group(2) or 1) - 1
        event = int(m.group(3) or 1)
        final = m.group(4)
        if final == b"u":
            if mods & _MOD_CTRL and code in (ord("c"), ord("q")):
                self.is_open = False
                return None
            key = _CODEPOINT_KEYS.get(code)
            if key is None:
                if code > 0x10FFFF:
                    return None
                key = chr(code).lower()
        elif final == b"~":
            key = _TILDE_KEYS.get(code)
        else:
            key = _LETTER_KEYS.get(final)
        if key is None:
            return None
        return key, event != 3                # 1 press, 2 repeat, 3 release

    def close(self) -> None:
        if getattr(self, "_saved_tty", None) is None:
            return
        self.is_open = False
        self._write(b"\x1b_Ga=d,d=A,q=2\x1b\\\x1b[<u\x1b[?25h\x1b[?1049l")
        termios.tcsetattr(self._fd, termios.TCSANOW, self._saved_tty)
        self._saved_tty = None
        sys.stdout = self._stdout
        for line in self._messages.lines + [self._messages._partial]:
            if line:
                print(line)


# ---- matplotlib -------------------------------------------------------------

class MplDisplay(_Display):
    """matplotlib window with the same interface as KittyDisplay."""

    def __init__(self, title: str = "DIGGER"):
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        # Default keymaps would eat game keys (s, l, r, arrows, f) before
        # our handlers see them.
        for k in ("keymap.save", "keymap.home", "keymap.yscale",
                  "keymap.back", "keymap.forward", "keymap.fullscreen",
                  "keymap.quit"):
            mpl.rcParams[k] = []
        self._plt = plt
        self._title = title
        self._fig, self._ax = plt.subplots()
        self._ax.set_axis_off()
        self._fig.tight_layout(pad=0)
        self._img = None
        self._events: list[tuple[str, bool]] = []
        self._fig.canvas.mpl_connect(
            "key_press_event", lambda e: self._key(e, True))
        self._fig.canvas.mpl_connect(
            "key_release_event", lambda e: self._key(e, False))
        self._fig.canvas.manager.set_window_title(title)
        plt.ion()
        plt.show()

    def _key(self, event, pressed: bool) -> None:
        if event.key:
            self._events.append((event.key.lower(), pressed))

    @property
    def is_open(self) -> bool:
        return self._plt.fignum_exists(self._fig.number)

    def show(self, frame: np.ndarray) -> None:
        rgb = frame[..., :3]
        if self._img is None or self._img.get_array().shape != rgb.shape:
            self._ax.clear()
            self._ax.set_axis_off()
            self._img = self._ax.imshow(rgb)
        else:
            self._img.set_data(rgb)
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def set_title(self, text: str) -> None:
        self._fig.canvas.manager.set_window_title(f"{self._title} -- {text}")

    def poll(self) -> list[tuple[str, bool]]:
        self._fig.canvas.flush_events()
        events, self._events = self._events, []
        return events

    def close(self) -> None:
        self._plt.close(self._fig)
