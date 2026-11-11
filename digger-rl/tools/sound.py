"""Play the emulator's audio (Digger's PC-speaker tunes and effects).

The libretro core hands its samples to the binding, which buffers them
while `core.enable_audio()` is on; `feed()` drains that buffer into a
sounddevice output stream after each emulated frame (or batch of frames).

    snd = open_sound(core)          # None if sounddevice is unavailable
    ...
    core.run(); snd and snd.feed()
    ...
    snd and snd.close()

The emulator delivers audio in bursts (a policy step runs 4 frames,
~2700 samples, at once, then nothing for ~57 ms), so the stream keeps a
jitter buffer: it starts playing only once `prefill` seconds are queued,
and if the queue still runs dry it goes silent and re-buffers, rather
than playing every late burst the moment it lands. With only the
device's ~7 ms queued, each bit of jitter in the caller's loop was an
audible click.

The stream plays at real time, so when the caller runs the game faster
than that (watch.py --speed 2) the backlog is cut back to `prefill` once
it passes `max_lag` seconds; slower than real time, it re-buffers.
`underruns` and `trims` count both events.
"""

from __future__ import annotations

import collections
import threading

import numpy as np


class Sound:
    def __init__(self, core, prefill: float = 0.08, max_lag: float = 0.3):
        import sounddevice as sd
        self._core = core
        self.rate = int(core.get_av_info()[5]) or 48000
        self._prefill = int(prefill * self.rate)
        self._max = int(max_lag * self.rate)
        self._buffering = True
        self.underruns = self.trims = 0
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
            if self._queued > self._max:
                # One cut back to the prefill level (one discontinuity),
                # not a chunk at a time.
                self.trims += 1
                while self._queued - len(self._chunks[0]) >= self._prefill:
                    self._queued -= len(self._chunks.popleft())

    def flush(self) -> None:
        """Drop queued and buffered audio (after a pause or a restore)."""
        self._core.take_audio()
        with self._lock:
            self._chunks.clear()
            self._queued = 0
            self._buffering = True

    def _callback(self, out, frames, _time, _status) -> None:
        i = 0
        with self._lock:
            if self._buffering:
                if self._queued < self._prefill:
                    out[:] = 0
                    return
                self._buffering = False
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
            if i < frames:
                self.underruns += 1
                self._buffering = True
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
