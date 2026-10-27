"""Video in and out through ffmpeg pipes (no OpenCV dependency)."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int | None


def probe(path: str) -> VideoInfo:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate,nb_frames:stream_side_data=rotation",
         "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["avg_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    w, h = int(s["width"]), int(s["height"])
    # ffmpeg auto-rotates on decode, so report the displayed size
    rot = next((abs(int(d.get("rotation", 0))) for d in s.get("side_data_list", [])), 0)
    if rot in (90, 270):
        w, h = h, w
    n = s.get("nb_frames")
    return VideoInfo(w, h, fps, int(n) if n and n.isdigit() else None)


def read_frames(path: str, width: int | None = None, gray: bool = False) -> Iterator[np.ndarray]:
    """Yield uint8 frames (RGB, or luma if `gray`), optionally downscaled to `width`.

    Every decoded frame is yielded once (`-vsync passthrough`), so frame i lines up
    with timestamp i from `timing.frame_times`.
    """
    info = probe(path)
    w, h = info.width, info.height
    args = []
    if width and width < w:
        h = int(round(h * width / w / 2) * 2)
        w = width
        args = ["-vf", f"scale={w}:{h}"]
    ch, fmt = (1, "gray") if gray else (3, "rgb24")
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, *args, "-vsync", "passthrough",
                          "-f", "rawvideo", "-pix_fmt", fmt, "-"], stdout=subprocess.PIPE)
    n = w * h * ch
    shape = (h, w) if gray else (h, w, 3)
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(shape)
    finally:
        p.stdout.close()
        p.wait()


class VideoWriter:
    """Write RGB frames to an H.264 MP4 (yuv420p, plays everywhere)."""

    def __init__(self, path: str, width: int, height: int, fps: float, crf: int = 20):
        self.p = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "-",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", path],
            stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        self.p.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise RuntimeError("ffmpeg failed to encode video")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
