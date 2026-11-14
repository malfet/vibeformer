"""Record what watch.py shows, at every emulator frame, with the game's
sound, to an mp4 for upload (H.264 + AAC, nearest-neighbour upscaled).

Video frames go straight to an ffmpeg pipe; the audio is collected
alongside and muxed in when the recording closes. Both come from the
emulator's own clock (70.087 frames/s, the core's sample rate), so they
stay in sync however fast or slow the playback ran.

ffmpeg is taken from PATH, else from the imageio-ffmpeg wheel
(`pip install imageio-ffmpeg`, which bundles a binary).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

FPS = 70.087


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
    except ImportError:
        raise SystemExit("--record needs ffmpeg: brew install ffmpeg, or "
                         "pip install imageio-ffmpeg") from None
    return imageio_ffmpeg.get_ffmpeg_exe()


class Recorder:
    def __init__(self, path: Path, scale: int = 2, crf: int = 18):
        self.path = Path(path)
        self.scale, self.crf = scale, crf
        self.exe = ffmpeg_exe()
        self._tmp = Path(tempfile.mkdtemp(prefix="digger_rec_"))
        self._proc: subprocess.Popen | None = None
        self._size: tuple[int, int] | None = None
        self._pcm: list[np.ndarray] = []
        self.rate = 0
        self.frames = 0

    def frame(self, rgb: np.ndarray) -> None:
        h, w = rgb.shape[:2]
        if self._proc is None:
            self._size = (w, h)
            self._proc = subprocess.Popen(
                [self.exe, "-loglevel", "error", "-y",
                 "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                 "-r", str(FPS), "-i", "-",
                 "-vf", f"scale=iw*{self.scale}:ih*{self.scale}:flags=neighbor",
                 "-c:v", "libx264", "-preset", "medium", "-crf", str(self.crf),
                 "-pix_fmt", "yuv420p", str(self._tmp / "video.mp4")],
                stdin=subprocess.PIPE)
        elif (w, h) != self._size:
            raise ValueError(f"frame size changed {self._size} -> {(w, h)}")
        self._proc.stdin.write(np.ascontiguousarray(rgb, np.uint8).tobytes())
        self.frames += 1

    def audio(self, pcm: np.ndarray, rate: int) -> None:
        """Interleaved stereo int16 samples."""
        self.rate = rate
        if pcm.size:
            self._pcm.append(pcm)

    def close(self) -> Path | None:
        if self._proc is None:
            return None
        self._proc.stdin.close()
        self._proc.wait()
        video = self._tmp / "video.mp4"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._pcm and self.rate:
            wav = self._tmp / "audio.wav"
            with wave.open(str(wav), "wb") as f:
                f.setnchannels(2)
                f.setsampwidth(2)
                f.setframerate(self.rate)
                f.writeframes(np.concatenate(self._pcm).tobytes())
            subprocess.run(
                [self.exe, "-loglevel", "error", "-y", "-i", str(video),
                 "-i", str(wav), "-c:v", "copy", "-c:a", "aac", "-b:a",
                 "192k", "-shortest", "-movflags", "+faststart",
                 str(self.path)], check=True)
        else:
            shutil.move(str(video), self.path)
        shutil.rmtree(self._tmp, ignore_errors=True)
        return self.path
