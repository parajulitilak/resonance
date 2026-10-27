"""True frame timing from the container, not the nominal frame rate.

Slow-motion phone files are the trap: iPhone originals carry real 240 fps
timestamps but the actual rate can drift (users report 177-219 fps), and
Android slow-motion files are stamped at playback speed with the capture rate
stored in the `com.android.capture.fps` tag. Frequencies are only as good as
the time axis, so every spectrum in Resonance is computed from these times.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass

import numpy as np


@dataclass
class TimeBase:
    t: np.ndarray            # seconds of real (capture) time, starting at 0, one per frame
    fps: float               # median capture rate
    nominal_fps: float       # what the container claims
    jitter: float            # std of frame interval / median interval
    time_scale: float        # factor applied to container time to get capture time
    source: str

    @property
    def variable(self) -> bool:
        return self.jitter > 0.02


def _probe_json(path: str, args: list[str]) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", *args, "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def capture_fps_tag(path: str) -> float | None:
    """Android slow motion: real capture rate stored as a format/stream tag."""
    info = _probe_json(path, ["-show_entries", "format_tags:stream_tags"])
    tags = dict(info.get("format", {}).get("tags", {}))
    for s in info.get("streams", []):
        tags.update(s.get("tags", {}))
    for k, v in tags.items():
        if k.lower() == "com.android.capture.fps":
            try:
                return float(v)
            except ValueError:
                return None
    return None


def frame_times(path: str, capture_fps: float | None = None) -> TimeBase:
    """Presentation timestamps of every video frame, rescaled to capture time.

    `capture_fps` overrides the detected capture rate (use it when a slow-motion
    file was exported at playback speed).
    """
    info = _probe_json(path, ["-select_streams", "v:0", "-show_entries",
                              "packet=pts_time:stream=avg_frame_rate,r_frame_rate"])
    pts = np.array(sorted(float(p["pts_time"]) for p in info.get("packets", [])
                          if p.get("pts_time") not in (None, "N/A")))
    if pts.size < 2:
        raise ValueError(f"no frame timestamps in {path}")
    num, den = info["streams"][0]["avg_frame_rate"].split("/")
    nominal = float(num) / float(den) if float(den) else float("nan")
    t = pts - pts[0]
    dt = np.diff(t)
    container_fps = 1.0 / float(np.median(dt))

    scale, source = 1.0, "container timestamps"
    tag = capture_fps_tag(path)
    if capture_fps:
        scale, source = container_fps / capture_fps, f"override capture_fps={capture_fps:g}"
    elif tag and tag > container_fps * 1.5:
        scale, source = container_fps / tag, f"android capture tag {tag:g} fps"
    t = t * scale
    dt = np.diff(t)
    med = float(np.median(dt))
    return TimeBase(t=t, fps=1.0 / med, nominal_fps=nominal, jitter=float(np.std(dt) / med),
                    time_scale=scale, source=source)
