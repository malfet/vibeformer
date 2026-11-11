"""Play the emulator's audio (Digger's PC-speaker tunes and effects).

The libretro core hands its samples to the binding, which buffers them
while `core.enable_audio()` is on; `feed()` drains that buffer into a
sounddevice output stream after each emulated frame (or batch of frames).

    snd = open_sound(core)          # None if sounddevice is unavailable
    ...
    core.run(); snd and snd.feed()
    ...
    snd and snd.close()

The stream plays at real time, so when the caller runs the game faster
than that (watch.py --speed 2) the backlog is capped at `max_lag`
seconds and the oldest audio is dropped; slower than real time, the gaps
are silent.
"""

from __future__ import annotations

import collections
import threading

import numpy as np


class Sound:
    def __init__(self, core, max_lag: float = 0.15):
        import sounddevice as sd
        self._core = core
        self.rate = int(core.get_av_info()[5]) or 48000
        self._max = int(max_lag * self.rate)
        self._chunks: collections.deque[np.ndarray] = collections.deque()
        self._queued = 0
        self._lock = threading.Lock()
        core.take_audio()                  # drop anything from before
        core.enable_audio(True)
        self._stream = sd.OutputStream(
            samplerate=self.rate, channels=2, dtype="int16",
            latency="low", callback=self._callback)
        self._stream.start()

    def feed(self) -> None:
        """Queue the audio the core produced since the last call."""
        pcm = np.frombuffer(self._core.take_audio(), np.int16)
        if not pcm.size:
            return
        with self._lock:
            self._chunks.append(pcm.reshape(-1, 2))
            self._queued += len(self._chunks[-1])
            while self._queued > self._max and len(self._chunks) > 1:
                self._queued -= len(self._chunks.popleft())

    def flush(self) -> None:
        """Drop queued and buffered audio (after a pause or a restore)."""
        self._core.take_audio()
        with self._lock:
            self._chunks.clear()
            self._queued = 0

    def _callback(self, out, frames, _time, _status) -> None:
        i = 0
        with self._lock:
            while i < frames and self._chunks:
                c = self._chunks[0]
                n = min(frames - i, len(c))
                out[i:i + n] = c[:n]
                i += n
                if n == len(c):
                    self._chunks.popleft()
                else:
                    self._chunks[0] = c[n:]
                self._queued -= n
        out[i:] = 0

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()
        self._core.enable_audio(False)
        self._core = None             # let the env drop and replace its core


def open_sound(core) -> Sound | None:
    """A Sound for `core`, or None (with a hint) if audio can't play."""
    try:
        return Sound(core)
    except ImportError:
        print("--sound needs sounddevice: pip install sounddevice",
              flush=True)
    except Exception as e:                 # no output device, etc.
        print(f"--sound: audio unavailable ({e})", flush=True)
    return None
